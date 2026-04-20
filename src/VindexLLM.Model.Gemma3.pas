{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Model.Gemma3;

{$I VindexLLM.Defines.inc}

interface

uses
  VindexLLM.GGUFReader,
  VindexLLM.Model;

type

  { TVdxGemma3Model }
  TVdxGemma3Model = class(TVdxModel)
  public
    class function SupportedArchitectures(): TArray<string>; override;

    function LoadModelConfig(const AReader: TVdxGGUFReader;
      const AMaxContext: Integer): Boolean; override;
    function LoadWeights(): Boolean; override;

    procedure RunLayerForward(const ALayer: Integer;
      const APosition: Integer); override;
    procedure RunLayerForwardBatch(const ALayer: Integer;
      const ANumTokens: UInt32; const AStartPos: UInt32;
      const ABidirectional: Boolean = False); override;

    function GetRoPETheta(const ALayer: Integer): Single; override;
    function FormatPrompt(const APrompt: string): string; override;
    function FormatEmbedding(const AText: string;
      const AIsQuery: Boolean): string; override;
    function GetStopTokenStrings(): TArray<string>; override;
    function SupportsEmbedding(): Boolean; override;
  end;

implementation

uses
  System.SysUtils,
  System.Math,
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.Compute,
  VindexLLM.Model.Registry;

{ TVdxGemma3Model }

class function TVdxGemma3Model.SupportedArchitectures(): TArray<string>;
begin
  Result := ['gemma3', 'gemma-embedding'];
end;

function TVdxGemma3Model.LoadModelConfig(const AReader: TVdxGGUFReader;
  const AMaxContext: Integer): Boolean;
var
  LArch: string;
  LQInfo: TVdxGGUFTensorInfo;
  LModelMax: UInt32;
begin
  Result := False;

  // Base adopts reader and records MaxContext
  if not inherited LoadModelConfig(AReader, AMaxContext) then Exit;

  LArch := FArchitecture;
  Status('Architecture: %s', [LArch]);

  // Read model dimensions from GGUF metadata
  FNumLayers := AReader.GetMetadataUInt32(LArch + '.block_count');
  FHiddenDim := AReader.GetMetadataUInt32(LArch + '.embedding_length');
  FFFNWidth := AReader.GetMetadataUInt32(LArch + '.feed_forward_length');
  FNumQHeads := AReader.GetMetadataUInt32(LArch + '.attention.head_count');
  FNumKVHeads := AReader.GetMetadataUInt32(LArch + '.attention.head_count_kv');

  // Derive head_dim from Q weight tensor shape
  if not AReader.GetTensorInfo('blk.0.attn_q.weight', LQInfo) then
  begin
    FErrors.Add(esFatal, 'CONF',
      'Missing required tensor: "blk.0.attn_q.weight"');
    Exit;
  end;
  FHeadDim := UInt32(LQInfo.Dimensions[1]) div FNumQHeads;
  FWeightType := LQInfo.TensorType;

  // Context length clamp
  if AReader.HasMetadata(LArch + '.context_length') then
    LModelMax := AReader.GetMetadataUInt32(LArch + '.context_length')
  else
    LModelMax := 8192;
  // AMaxContext <= 0 means "use model's native max" (e.g., embeddings)
  if AMaxContext <= 0 then
    FMaxSeqLen := LModelMax
  else
    FMaxSeqLen := Min(UInt32(AMaxContext), LModelMax);
  Status('Context length: %d (model max: %d)', [FMaxSeqLen, LModelMax]);

  // Detect embedding tensor type from tensor info
  if not AReader.GetTensorInfo('token_embd.weight', LQInfo) then
  begin
    FErrors.Add(esFatal, 'CONF',
      'Missing required tensor: "token_embd.weight"');
    Exit;
  end;
  FEmbedType := LQInfo.TensorType;

  Status('Weight type: %s', [VdxGGMLTypeName(FWeightType)]);
  Status('Embedding type: %s', [VdxGGMLTypeName(FEmbedType)]);
  Status('Config: layers=%d hidden=%d ffn=%d heads=%d/%d head_dim=%d',
    [FNumLayers, FHiddenDim, FFFNWidth, FNumQHeads, FNumKVHeads, FHeadDim]);

  Result := True;
end;

function TVdxGemma3Model.LoadWeights(): Boolean;
var
  LLayer: Integer;
begin
  Result := False;
  Status('Uploading weights to GPU...');

  // FFN gate/down index + upload
  if not FFFN.BuildFromGGUF(FReader) then
  begin
    FErrors.Add(esFatal, 'LOAD', 'Failed to build FFN weight index from GGUF');
    Exit;
  end;
  Status('  Uploading gate/down vectors (%d layers)...', [FNumLayers]);
  FFFN.UploadAll(FCompute);

  // Per-layer attention weights (Q/K/V/O)
  Status('  Uploading attention weights (%d layers)...', [FNumLayers]);
  SetLength(FAttnWeights, FNumLayers);
  for LLayer := 0 to Integer(FNumLayers) - 1 do
    FAttn.UploadAttnWeights(FReader, LLayer, FAttnWeights[LLayer]);

  // Per-layer FFN up weights
  Status('  Uploading FFN up weights (%d layers)...', [FNumLayers]);
  SetLength(FUpWeights, FNumLayers);
  for LLayer := 0 to Integer(FNumLayers) - 1 do
    FUpWeights[LLayer] := UploadWeightTensor(
      Format('blk.%d.ffn_up.weight', [LLayer]));

  // Per-layer norm weights (6 per layer)
  Status('  Uploading norm weights (%d layers)...', [FNumLayers]);
  SetLength(FNormWeights, FNumLayers);
  for LLayer := 0 to Integer(FNumLayers) - 1 do
  begin
    FNormWeights[LLayer].AttnNormGpu := UploadNormWeight(
      Format('blk.%d.attn_norm.weight', [LLayer]), FHiddenDim);
    FNormWeights[LLayer].PostAttnNormGpu := UploadNormWeight(
      Format('blk.%d.post_attention_norm.weight', [LLayer]), FHiddenDim);
    FNormWeights[LLayer].FFNNormGpu := UploadNormWeight(
      Format('blk.%d.ffn_norm.weight', [LLayer]), FHiddenDim);
    FNormWeights[LLayer].PostFFNNormGpu := UploadNormWeight(
      Format('blk.%d.post_ffw_norm.weight', [LLayer]), FHiddenDim);
    FNormWeights[LLayer].QNormGpu := UploadNormWeight(
      Format('blk.%d.attn_q_norm.weight', [LLayer]), FHeadDim);
    FNormWeights[LLayer].KNormGpu := UploadNormWeight(
      Format('blk.%d.attn_k_norm.weight', [LLayer]), FHeadDim);
  end;

  // Global output norm
  FOutputNormGpu := UploadNormWeight('output_norm.weight', FHiddenDim);

  // Bail if any weight upload failed
  if FErrors.HasFatal() then Exit;

  // --- Embedding table ---
  FEmbedPtr := PByte(FReader.GetTensorDataPtr('token_embd.weight'));
  if FEmbedPtr = nil then
  begin
    FErrors.Add(esFatal, 'LOAD', 'token_embd.weight not found in GGUF');
    Exit;
  end;
  if (FEmbedType <> gtF16) and (FEmbedType <> gtF32) and
     (FEmbedType <> gtQ8_0) and (FEmbedType <> gtQ4_0) then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Unsupported embedding type: %s (need F16, F32, Q8_0, or Q4_0)',
      [VdxGGMLTypeName(FEmbedType)]);
    Exit;
  end;
  FEmbedScale := Sqrt(Single(FHiddenDim));

  // Upload embedding table to GPU for unembedding + batch embed
  Status('  Uploading embedding table to GPU...');
  FEmbedGpu := UploadWeightTensor('token_embd.weight');
  if FErrors.HasFatal() then Exit;

  // Build batch resources (needs FEmbedGpu valid + FVocabSize set)
  if not BuildBatchResources() then Exit;

  Status('Weights loaded. Embed scale: %.4f', [FEmbedScale]);
  Result := True;
end;

procedure TVdxGemma3Model.RunLayerForward(const ALayer: Integer;
  const APosition: Integer);
var
  LTheta: Single;
  LGeluPush: TVdxGeluMulPush;
  LVecAddPush: TVdxVecAddPush;
begin
  // === Attention branch: x = x + PostAttnNorm(Attn(PreAttnNorm(x))) ===

  // Fused copy+norm: residual → PreAttnNorm → WorkBufA
  FNorm.ApplyCopy(FResidualGpu, FNormWeights[ALayer].AttnNormGpu,
    FWorkBufA, FHiddenDim);
  FCompute.BatchBarrier();

  // Per-layer RoPE theta
  LTheta := GetRoPETheta(ALayer);

  // Full attention: normed input → attention output
  FAttn.Forward(
    FWorkBufA,
    FAttnWeights[ALayer],
    FNormWeights[ALayer].QNormGpu,
    FNormWeights[ALayer].KNormGpu,
    ALayer,
    APosition,
    LTheta,
    FAttnOutBuf);

  // PostAttnNorm on attention output
  FNorm.Apply(FAttnOutBuf,
    FNormWeights[ALayer].PostAttnNormGpu, FHiddenDim);
  FCompute.BatchBarrier();

  // GPU vec_add: residual += attn_output
  LVecAddPush.Count := FHiddenDim;
  FCompute.DispatchComputeWithPush(
    FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
    FVecAddAttnDescSet, @LVecAddPush, SizeOf(LVecAddPush),
    (FHiddenDim + 255) div 256);
  FCompute.BatchBarrier();

  // === FFN branch: x = x + PostFFNNorm(FFN(PreFFNNorm(x))) ===

  // Fused copy+norm: residual → PreFFNNorm → WorkBufA
  FNorm.ApplyCopy(FResidualGpu, FNormWeights[ALayer].FFNNormGpu,
    FWorkBufA, FHiddenDim);
  FCompute.BatchBarrier();

  // gate(x) → FGateBuf [FFNWidth]
  FAttn.TestMatVec(FFFN.GetLayer(ALayer).GateGpuBuffer,
    FWorkBufA, FGateBuf, FHiddenDim, FFFNWidth, FWeightType);

  // up(x) → FUpBuf [FFNWidth]
  FAttn.TestMatVec(FUpWeights[ALayer],
    FWorkBufA, FUpBuf, FHiddenDim, FFFNWidth, FWeightType);
  FCompute.BatchBarrier();

  // GELU(gate) * up → FGateBuf [FFNWidth] in-place
  LGeluPush.Count := FFFNWidth;
  FCompute.DispatchComputeWithPush(
    FGeluMulBundle.Pipeline, FGeluMulBundle.PipelineLayout,
    FGeluMulDescSet, @LGeluPush, SizeOf(LGeluPush),
    (FFFNWidth + 255) div 256);
  FCompute.BatchBarrier();

  // down(hidden) → FFFNOutBuf [HiddenDim]
  FAttn.TestMatVec(FFFN.GetLayer(ALayer).DownGpuBuffer,
    FGateBuf, FFFNOutBuf, FFFNWidth, FHiddenDim, FWeightType);
  FCompute.BatchBarrier();

  // PostFFNNorm on FFN output
  FNorm.Apply(FFFNOutBuf,
    FNormWeights[ALayer].PostFFNNormGpu, FHiddenDim);
  FCompute.BatchBarrier();

  // GPU vec_add: residual += ffn_output
  LVecAddPush.Count := FHiddenDim;
  FCompute.DispatchComputeWithPush(
    FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
    FVecAddFFNDescSet, @LVecAddPush, SizeOf(LVecAddPush),
    (FHiddenDim + 255) div 256);
  FCompute.BatchBarrier();
end;

procedure TVdxGemma3Model.RunLayerForwardBatch(const ALayer: Integer;
  const ANumTokens: UInt32; const AStartPos: UInt32;
  const ABidirectional: Boolean);
var
  LTheta: Single;
  LGeluPush: TVdxGeluMulPush;
  LVecAddPush: TVdxVecAddPush;
begin
  // === Attention branch ===

  // Fused copy+norm batch: FResidualMat → PreAttnNorm → FWorkMat
  FNorm.ApplyCopyBatch(FResidualMat, FNormWeights[ALayer].AttnNormGpu,
    FWorkMat, FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();

  LTheta := GetRoPETheta(ALayer);

  // Full batched attention: FWorkMat → FAttnOutMatBuf
  FAttn.ForwardBatch(FWorkMat, FAttnWeights[ALayer],
    FNormWeights[ALayer].QNormGpu, FNormWeights[ALayer].KNormGpu,
    ALayer, ANumTokens, AStartPos, LTheta,
    FQMat, FKMat, FVMat, FAttnOutMatBuf, ABidirectional);

  // PostAttnNorm batch
  FNorm.ApplyBatch(FAttnOutMatBuf,
    FNormWeights[ALayer].PostAttnNormGpu, FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();

  // Vec-add batch: FResidualMat += FAttnOutMatBuf
  LVecAddPush.Count := ANumTokens * FHiddenDim;
  FCompute.DispatchComputeWithPush(
    FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
    FBatchVecAddAttnDescSet, @LVecAddPush, SizeOf(LVecAddPush),
    (ANumTokens * FHiddenDim + 255) div 256);
  FCompute.BatchBarrier();

  // === FFN branch ===

  // Fused copy+norm batch: FResidualMat → PreFFNNorm → FWorkMat
  FNorm.ApplyCopyBatch(FResidualMat, FNormWeights[ALayer].FFNNormGpu,
    FWorkMat, FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();

  // Gate matmul: FWorkMat → FGateMat
  FAttn.BatchMatMul(FFFN.GetLayer(ALayer).GateGpuBuffer,
    FWorkMat, FGateMat, FHiddenDim, FFFNWidth, ANumTokens, FWeightType);

  // Up matmul: FWorkMat → FUpMatBuf
  FAttn.BatchMatMul(FUpWeights[ALayer],
    FWorkMat, FUpMatBuf, FHiddenDim, FFFNWidth, ANumTokens, FWeightType);
  FCompute.BatchBarrier();

  // GELU-mul batch
  LGeluPush.Count := ANumTokens * FFFNWidth;
  FCompute.DispatchComputeWithPush(
    FGeluMulBundle.Pipeline, FGeluMulBundle.PipelineLayout,
    FBatchGeluMulDescSet, @LGeluPush, SizeOf(LGeluPush),
    (ANumTokens * FFFNWidth + 255) div 256);
  FCompute.BatchBarrier();

  // Down matmul: FGateMat → FFFNOutMat
  FAttn.BatchMatMul(FFFN.GetLayer(ALayer).DownGpuBuffer,
    FGateMat, FFFNOutMat, FFFNWidth, FHiddenDim, ANumTokens, FWeightType);
  FCompute.BatchBarrier();

  // PostFFNNorm batch
  FNorm.ApplyBatch(FFFNOutMat,
    FNormWeights[ALayer].PostFFNNormGpu, FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();

  // Vec-add batch: FResidualMat += FFFNOutMat
  LVecAddPush.Count := ANumTokens * FHiddenDim;
  FCompute.DispatchComputeWithPush(
    FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
    FBatchVecAddFFNDescSet, @LVecAddPush, SizeOf(LVecAddPush),
    (ANumTokens * FHiddenDim + 255) div 256);
  FCompute.BatchBarrier();
end;

function TVdxGemma3Model.GetRoPETheta(const ALayer: Integer): Single;
begin
  if ALayer mod 6 = 5 then
    Result := 1000000.0
  else
    Result := 10000.0;
end;

function TVdxGemma3Model.FormatPrompt(const APrompt: string): string;
begin
  Result := '<start_of_turn>user' + #10 + APrompt + '<end_of_turn>' + #10 +
            '<start_of_turn>model' + #10;
end;

function TVdxGemma3Model.FormatEmbedding(const AText: string;
  const AIsQuery: Boolean): string;
begin
  if AIsQuery then
    Result := 'task: search result | query: ' + AText
  else
    Result := 'title: none | text: ' + AText;
end;

function TVdxGemma3Model.GetStopTokenStrings(): TArray<string>;
begin
  Result := ['<end_of_turn>'];
end;

function TVdxGemma3Model.SupportsEmbedding(): Boolean;
begin
  Result := True;
end;

initialization
  TVdxModelRegistry.RegisterClass(TVdxGemma3Model);

end.
