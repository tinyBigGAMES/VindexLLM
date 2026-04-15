{===============================================================================
  VindexLLM™ - Graph-Walk LLM Inference Engine

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Inference;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Classes,
  System.IOUtils,
  System.Math,
  System.Diagnostics,
  System.Generics.Collections,
  VindexLLM.Utils,
  VindexLLM.GGUFReader,
  VindexLLM.VulkanCompute,
  VindexLLM.LayerNorm,
  VindexLLM.Attention,
  VindexLLM.Vindex,
  VindexLLM.Tokenizer,
  VindexLLM.ChatTemplate,
  VindexLLM.VirtualBuffer,
  VindexLLM.Shaders;

// ============================================================================
//  Push constants for shaders
// ============================================================================

type
  TVdxGeluMulPush = record
    Count: UInt32;
  end;

  TVdxVecAddPush = record
    Count: UInt32;
  end;

  TVdxEmbedLookupPush = record
    TokenId: UInt32;
    DimParam: UInt32;    // hidden_dim/2 for F16, hidden_dim for Q8_0
    EmbedScale: Single;
  end;

  // Batched embedding lookup push constants (Phase 6D)
  TVdxEmbedBatchPush = record
    DimParam: UInt32;    // hidden_dim/2 for F16, hidden_dim for Q8_0
    EmbedScale: Single;
    NumTokens: UInt32;
  end;

// ============================================================================
//  Stop reason — why generation ended
// ============================================================================

  TVdxStopReason = (
    srNone,              // Not yet generated
    srEOS,               // End-of-sequence token
    srStopToken,         // User-defined stop token (e.g., <end_of_turn>)
    srMaxTokens,         // Reached max token limit
    srCallbackStopped    // Token callback returned False
  );

// ============================================================================
//  Inference stats — filled by Generate(), read via GetStats()
// ============================================================================

  TVdxInferenceStats = record
    PrefillTokens: Integer;
    PrefillTimeMs: Double;
    PrefillTokPerSec: Double;
    GeneratedTokens: Integer;
    GenerationTimeMs: Double;
    GenerationTokPerSec: Double;
    TimeToFirstTokenMs: Double;
    TotalTimeMs: Double;
    StopReason: TVdxStopReason;
  end;
  PVdxInferenceStats = ^TVdxInferenceStats;

// ============================================================================
//  Token callback — return True to continue, False to stop generation
// ============================================================================

  TVdxTokenCallback = reference to function(
    const AToken: string;
    const AUserData: Pointer): Boolean;

// ============================================================================
//  TVdxInference — Dense transformer inference engine
// ============================================================================

  { TVdxInference }
  TVdxInference = class(TVdxStatusObject)
  private
    // Subsystem objects
    FReader: TVdxGGUFReader;
    FCompute: TVdxVulkanCompute;
    FNorm: TVdxLayerNorm;
    FAttn: TVdxAttention;
    FVindex: TVdxVindex;
    FTokenizer: TVdxTokenizer;

    // Token callback
    FTokenCallback: TVdxCallback<TVdxTokenCallback>;

    // Model state
    FModelLoaded: Boolean;

    // Model config (read from GGUF metadata)
    FArchitecture: string;
    FNumLayers: UInt32;
    FHiddenDim: UInt32;
    FFFNWidth: UInt32;
    FNumQHeads: UInt32;
    FNumKVHeads: UInt32;
    FHeadDim: UInt32;
    FVocabSize: Integer;
    FMaxSeqLen: UInt32;

    // Stop token IDs — generation stops when any of these are produced
    FStopTokenIds: TList<Integer>;

    // Embedding (mmap'd from GGUF)
    FEmbedPtr: PByte;
    FEmbedScale: Single;
    FEmbedType: TVdxGGMLType;     // embedding tensor format (for unembedding matvec)

    // Per-layer weights
    FAttnWeights: array of TVdxAttnLayerWeights;
    FNormWeights: array of TVdxNormLayerWeights;
    FUpWeights: array of TVdxGpuBuffer;
    FWeightType: TVdxGGMLType;         // weight tensor type (F16 or Q8_0)

    // Global weights
    FOutputNormGpu: TVdxGpuBuffer;

    // GPU residual buffer — stays on GPU between layers (no CPU round-trip)
    FResidualGpu: TVdxGpuBuffer;

    // Work GPU buffers
    FWorkBufA: TVdxGpuBuffer;      // normed residual [HiddenDim] F32
    FAttnOutBuf: TVdxGpuBuffer;    // attention output [HiddenDim] F32
    FGateBuf: TVdxGpuBuffer;       // gate projection [FFNWidth] F32
    FUpBuf: TVdxGpuBuffer;         // up projection [FFNWidth] F32
    FFFNOutBuf: TVdxGpuBuffer;     // FFN output [HiddenDim] F32

    // GELU-mul pipeline
    FGeluMulShader: VkShaderModule;
    FGeluMulBundle: TVdxComputePipelineBundle;
    FGeluMulDescLayout: VkDescriptorSetLayout;
    FGeluMulDescPool: VkDescriptorPool;
    FGeluMulDescSet: VkDescriptorSet;

    // Vec-add pipeline (GPU residual += branch output)
    FVecAddShader: VkShaderModule;
    FVecAddBundle: TVdxComputePipelineBundle;
    FVecAddDescLayout: VkDescriptorSetLayout;
    FVecAddDescPool: VkDescriptorPool;
    FVecAddAttnDescSet: VkDescriptorSet;  // residual += attn output
    FVecAddFFNDescSet: VkDescriptorSet;   // residual += ffn output

    // GPU embedding lookup (eliminates CPU→GPU transfer per token)
    FEmbedF16Shader: VkShaderModule;
    FEmbedQ8Shader: VkShaderModule;
    FEmbedF16Bundle: TVdxComputePipelineBundle;
    FEmbedQ8Bundle: TVdxComputePipelineBundle;
    FEmbedDescLayout: VkDescriptorSetLayout;
    FEmbedDescPool: VkDescriptorPool;
    FEmbedDescSet: VkDescriptorSet;

    // Batched GPU embedding lookup (Phase 6D — prefill batching)
    FEmbedBatchF16Shader: VkShaderModule;
    FEmbedBatchQ8Shader: VkShaderModule;
    FEmbedBatchF16Bundle: TVdxComputePipelineBundle;
    FEmbedBatchQ8Bundle: TVdxComputePipelineBundle;
    FEmbedBatchDescLayout: VkDescriptorSetLayout;  // 3 bindings: table, output, token_ids
    FEmbedBatchDescPool: VkDescriptorPool;
    FEmbedBatchDescSet: VkDescriptorSet;
    FTokenIdsGpu: TVdxGpuBuffer;   // host-visible, holds token IDs for batched embed

    // Batched prefill matrix buffers (Phase 6D)
    FResidualMat: TVdxGpuBuffer;    // [MaxSeq x HiddenDim] F32
    FWorkMat: TVdxGpuBuffer;        // [MaxSeq x HiddenDim] F32
    FQMat: TVdxGpuBuffer;           // [MaxSeq x NumQHeads*HeadDim] F32
    FKMat: TVdxGpuBuffer;           // [MaxSeq x NumKVHeads*HeadDim] F32
    FVMat: TVdxGpuBuffer;           // [MaxSeq x NumKVHeads*HeadDim] F32
    FAttnOutMatBuf: TVdxGpuBuffer;  // [MaxSeq x HiddenDim] F32
    FGateMat: TVdxGpuBuffer;        // [MaxSeq x FFNWidth] F32
    FUpMatBuf: TVdxGpuBuffer;       // [MaxSeq x FFNWidth] F32
    FFFNOutMat: TVdxGpuBuffer;      // [MaxSeq x HiddenDim] F32

    // Batched elementwise descriptor sets (Phase 6D)
    FBatchEWDescPool: VkDescriptorPool;
    FBatchVecAddAttnDescSet: VkDescriptorSet;
    FBatchVecAddFFNDescSet: VkDescriptorSet;
    FBatchGeluMulDescSet: VkDescriptorSet;

    // GPU unembedding (avoids slow CPU F16 scan)
    FEmbedGpu: TVdxGpuBuffer;     // F16 embedding table on GPU
    FLogitsBuf: TVdxGpuBuffer;    // F32 logits output [VocabSize], host-visible

    // Pre-allocated CPU logits buffer (avoids per-token heap allocation)
    FLogitsVBuf: TVdxVirtualBuffer<Single>;

    // CPU-side embedding scratch (for F16→F32 conversion before upload)
    FResidual: array of Single;

    // Stats — filled by Generate()
    FStats: TVdxInferenceStats;

    // Private helpers
    function F16ToF32(const AVal: UInt16): Single;
    function UploadNormWeight(const ATensorName: string;
      const ACount: UInt32): TVdxGpuBuffer;
    function UploadWeightTensor(const ATensorName: string): TVdxGpuBuffer;
    procedure EmbedToken(const ATokenId: Integer);
    procedure EmbedTokensBatch(const ATokenIds: TArray<Integer>;
      const ANumTokens: Integer; const AOutputBuf: TVdxGpuBuffer);
    procedure RunLayerForward(const ALayer: Integer;
      const APosition: Integer);
    procedure RunLayerForwardBatch(const ALayer: Integer;
      const ANumTokens: UInt32);
    function RunUnembedding(): Integer;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    procedure LoadModel(const AGGUFPath: string);

    procedure SetTokenCallback(const ACallback: TVdxTokenCallback;
      const AUserData: Pointer);

    function Generate(const APrompt: string;
      const AMaxTokens: Integer = 256): string;

    procedure UnloadModel();

    procedure AddStopToken(const ATokenId: Integer);
    procedure ClearStopTokens();

    // Stats from last Generate() call — pointer to internal record
    function GetStats(): PVdxInferenceStats;
  end;

implementation

// ============================================================================
//  TVdxInference — Construction / Destruction
// ============================================================================

constructor TVdxInference.Create();
begin
  inherited;

  FReader := nil;
  FCompute := nil;
  FNorm := nil;
  FAttn := nil;
  FVindex := nil;
  FTokenizer := nil;
  FModelLoaded := False;
  FArchitecture := '';
  FStopTokenIds := TList<Integer>.Create();
  FLogitsVBuf := nil;
  FEmbedPtr := nil;
  FEmbedScale := 0.0;
end;

destructor TVdxInference.Destroy();
begin
  if FModelLoaded then
    UnloadModel();

  FreeAndNil(FStopTokenIds);

  inherited;
end;

// ============================================================================
//  F16ToF32 — Convert IEEE 754 half-precision to single-precision
// ============================================================================

function TVdxInference.F16ToF32(const AVal: UInt16): Single;
const
  CSignBit: UInt32 = $80000000;
var
  LExp: UInt32;
  LMant: UInt32;
  LBits: UInt32;
begin
  LExp := (UInt32(AVal) shr 10) and $1F;
  LMant := UInt32(AVal) and $3FF;

  if LExp = 0 then
  begin
    if LMant = 0 then
      LBits := 0
    else
    begin
      LExp := 1;
      while (LMant and $400) = 0 do
      begin
        LMant := LMant shl 1;
        Inc(LExp);
      end;
      LMant := LMant and $3FF;
      LBits := UInt32((113 - LExp) shl 23) or (LMant shl 13);
    end;
  end
  else if LExp = $1F then
    LBits := UInt32($FF shl 23) or (LMant shl 13)
  else
    LBits := UInt32((LExp + 112) shl 23) or (LMant shl 13);

  if (AVal and $8000) <> 0 then
    LBits := LBits or CSignBit;

  Move(LBits, Result, 4);
end;

// ============================================================================
//  UploadNormWeight — Upload F32 norm weight from GGUF to GPU
//  GGUF stores effective weight (Gemma +1 offset already applied)
// ============================================================================

function TVdxInference.UploadNormWeight(const ATensorName: string;
  const ACount: UInt32): TVdxGpuBuffer;
var
  LPtr: Pointer;
  LData: array of Single;
begin
  LPtr := FReader.GetTensorDataPtr(ATensorName);
  SetLength(LData, ACount);
  Move(LPtr^, LData[0], ACount * SizeOf(Single));

  Result := FCompute.CreateGpuBuffer(
    UInt64(ACount) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  FCompute.UploadToBuffer(Result, @LData[0], UInt64(ACount) * SizeOf(Single));
end;

// ============================================================================
//  UploadWeightTensor — Upload any weight tensor (F16/Q4_0) to device-local GPU
// ============================================================================

function TVdxInference.UploadWeightTensor(const ATensorName: string): TVdxGpuBuffer;
var
  LInfo: TVdxGGUFTensorInfo;
  LPtr: Pointer;
  LSize: UInt64;
  LStaging: TVdxGpuBuffer;
begin
  LInfo := FReader.GetTensorInfo(ATensorName);
  LPtr := FReader.GetTensorDataPtr(ATensorName);
  LSize := VdxGGMLTensorBytes(LInfo.TensorType,
    LInfo.Dimensions[0], LInfo.Dimensions[1]);
  TVdxUtils.FailIf(LSize = 0,
    'Unsupported tensor type for %s: %s',
    [ATensorName, VdxGGMLTypeName(LInfo.TensorType)]);

  LStaging := FCompute.CreateGpuBuffer(LSize,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  try
    FCompute.UploadToBuffer(LStaging, LPtr, LSize);
    Result := FCompute.CreateGpuBuffer(LSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    FCompute.CopyBuffer(LStaging, Result, LSize);
  finally
    FCompute.DestroyGpuBuffer(LStaging);
  end;
end;

// ============================================================================
//  EmbedToken — GPU embedding lookup: read from embedding table on GPU,
//  dequantize + scale, write to residual buffer. No CPU→GPU transfer.
//  F32 fallback: CPU conversion + upload (rare, kept for completeness).
//  Must be called inside an active batch (BeginBatch/EndBatch).
// ============================================================================

procedure TVdxInference.EmbedToken(const ATokenId: Integer);
var
  LPush: TVdxEmbedLookupPush;
  LI: Integer;
  LOffset: UInt64;
begin
  if FEmbedType = gtF32 then
  begin
    // F32 fallback: CPU conversion + upload
    LOffset := UInt64(ATokenId) * UInt64(FHiddenDim) * 4;
    for LI := 0 to Integer(FHiddenDim) - 1 do
      FResidual[LI] := PSingle(FEmbedPtr + LOffset + UInt64(LI) * 4)^
        * FEmbedScale;
    FCompute.UploadToBuffer(FResidualGpu, @FResidual[0],
      UInt64(FHiddenDim) * SizeOf(Single));
  end
  else
  begin
    // GPU dispatch: F16 or Q8_0 embedding lookup
    LPush.TokenId := UInt32(ATokenId);
    LPush.EmbedScale := FEmbedScale;

    if FEmbedType = gtQ8_0 then
    begin
      LPush.DimParam := FHiddenDim;
      FCompute.DispatchComputeWithPush(
        FEmbedQ8Bundle.Pipeline, FEmbedQ8Bundle.PipelineLayout,
        FEmbedDescSet, @LPush, SizeOf(LPush), 1);
    end
    else
    begin
      LPush.DimParam := FHiddenDim div 2;
      FCompute.DispatchComputeWithPush(
        FEmbedF16Bundle.Pipeline, FEmbedF16Bundle.PipelineLayout,
        FEmbedDescSet, @LPush, SizeOf(LPush),
        (FHiddenDim div 2 + 255) div 256);
    end;
    FCompute.BatchBarrier(); // ResidualGpu ready for first layer
  end;
end;

// ============================================================================
//  EmbedTokensBatch — GPU batched embedding: look up N tokens at once,
//  dequantize + scale, write to output matrix [N x HiddenDim].
//  Uploads token IDs to GPU, then dispatches batched embed shader.
//  Must be called inside an active batch (BeginBatch/EndBatch).
// ============================================================================

procedure TVdxInference.EmbedTokensBatch(const ATokenIds: TArray<Integer>;
  const ANumTokens: Integer; const AOutputBuf: TVdxGpuBuffer);
var
  LPush: TVdxEmbedBatchPush;
  LIds: array of UInt32;
  LI: Integer;
begin
  // Convert Integer token IDs to UInt32 and upload to GPU
  SetLength(LIds, ANumTokens);
  for LI := 0 to ANumTokens - 1 do
    LIds[LI] := UInt32(ATokenIds[LI]);
  FCompute.UploadToBuffer(FTokenIdsGpu, @LIds[0],
    UInt64(ANumTokens) * SizeOf(UInt32));

  // Rebind descriptor set with current output matrix
  FCompute.UpdateDescriptorSetBuffers(FEmbedBatchDescSet,
    [FEmbedGpu, AOutputBuf, FTokenIdsGpu]);

  LPush.EmbedScale := FEmbedScale;
  LPush.NumTokens := UInt32(ANumTokens);

  if FEmbedType = gtQ8_0 then
  begin
    LPush.DimParam := FHiddenDim;
    FCompute.DispatchComputeWithPush(
      FEmbedBatchQ8Bundle.Pipeline, FEmbedBatchQ8Bundle.PipelineLayout,
      FEmbedBatchDescSet, @LPush, SizeOf(LPush),
      UInt32(ANumTokens));  // one workgroup per token
  end
  else
  begin
    LPush.DimParam := FHiddenDim div 2;
    FCompute.DispatchComputeWithPush(
      FEmbedBatchF16Bundle.Pipeline, FEmbedBatchF16Bundle.PipelineLayout,
      FEmbedBatchDescSet, @LPush, SizeOf(LPush),
      (FHiddenDim div 2 + 255) div 256, UInt32(ANumTokens));  // 2D dispatch
  end;
  FCompute.BatchBarrier(); // Output matrix ready for first layer
end;

// ============================================================================
//  RunLayerForward — Process one transformer layer (attn + FFN)
//  Sandwich norm pattern: PreNorm → Op → PostNorm → residual add
//  All operations recorded into the active batch — no individual submits
// ============================================================================

procedure TVdxInference.RunLayerForward(const ALayer: Integer;
  const APosition: Integer);
var
  LTheta: Single;
  LGeluPush: TVdxGeluMulPush;
  LVecAddPush: TVdxVecAddPush;
begin
  // === Attention branch: x = x + PostAttnNorm(Attn(PreAttnNorm(x))) ===

  // Fused copy+norm: residual → PreAttnNorm → WorkBufA (no separate copy)
  FNorm.ApplyCopy(FResidualGpu, FNormWeights[ALayer].AttnNormGpu,
    FWorkBufA, FHiddenDim);
  FCompute.BatchBarrier(); // WorkBufA normed, ready for attention matvecs

  // Per-layer RoPE theta: full layers use 1M, sliding layers use 10K
  if ALayer mod 6 = 5 then
    LTheta := 1000000.0
  else
    LTheta := 10000.0;

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
  FCompute.BatchBarrier(); // Normed AttnOutBuf ready for vec_add

  // GPU vec_add: residual += attn_output (no CPU round-trip)
  LVecAddPush.Count := FHiddenDim;
  FCompute.DispatchComputeWithPush(
    FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
    FVecAddAttnDescSet, @LVecAddPush, SizeOf(LVecAddPush),
    (FHiddenDim + 255) div 256);
  FCompute.BatchBarrier(); // Residual updated, ready for FFN copy

  // === FFN branch: x = x + PostFFNNorm(FFN(PreFFNNorm(x))) ===

  // Fused copy+norm: residual → PreFFNNorm → WorkBufA (no separate copy)
  FNorm.ApplyCopy(FResidualGpu, FNormWeights[ALayer].FFNNormGpu,
    FWorkBufA, FHiddenDim);
  FCompute.BatchBarrier(); // WorkBufA normed, ready for gate/up matvecs

  // Dense FFN: gate(x), up(x), GELU(gate)*up, down(hidden)

  // gate(x) → FGateBuf [FFNWidth] — reads WorkBufA, writes GateBuf
  FAttn.TestMatVec(FVindex.GetLayer(ALayer).GateGpuBuffer,
    FWorkBufA, FGateBuf, FHiddenDim, FFFNWidth, FWeightType);

  // up(x) → FUpBuf [FFNWidth] — reads WorkBufA, writes UpBuf (independent of gate)
  FAttn.TestMatVec(FUpWeights[ALayer],
    FWorkBufA, FUpBuf, FHiddenDim, FFFNWidth, FWeightType);
  FCompute.BatchBarrier(); // GateBuf/UpBuf ready for GELU-mul

  // GELU(gate) * up → FGateBuf [FFNWidth] in-place
  LGeluPush.Count := FFFNWidth;
  FCompute.DispatchComputeWithPush(
    FGeluMulBundle.Pipeline, FGeluMulBundle.PipelineLayout,
    FGeluMulDescSet, @LGeluPush, SizeOf(LGeluPush),
    (FFFNWidth + 255) div 256);
  FCompute.BatchBarrier(); // GateBuf (GELU result) ready for down matvec

  // down(hidden) → FFFNOutBuf [HiddenDim]
  FAttn.TestMatVec(FVindex.GetLayer(ALayer).DownGpuBuffer,
    FGateBuf, FFFNOutBuf, FFFNWidth, FHiddenDim, FWeightType);
  FCompute.BatchBarrier(); // FFNOutBuf ready for post-norm

  // PostFFNNorm on FFN output
  FNorm.Apply(FFFNOutBuf,
    FNormWeights[ALayer].PostFFNNormGpu, FHiddenDim);
  FCompute.BatchBarrier(); // Normed FFNOutBuf ready for vec_add

  // GPU vec_add: residual += ffn_output (no CPU round-trip)
  LVecAddPush.Count := FHiddenDim;
  FCompute.DispatchComputeWithPush(
    FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
    FVecAddFFNDescSet, @LVecAddPush, SizeOf(LVecAddPush),
    (FHiddenDim + 255) div 256);
  FCompute.BatchBarrier(); // Residual ready for next layer
end;

// ============================================================================
//  RunLayerForwardBatch — Process one transformer layer for N tokens (Phase 6D)
//  Same sandwich norm pattern as RunLayerForward but on matrices.
//  Uses batched matmul, batched norms, and existing vec-add/gelu-mul with N×dim.
// ============================================================================

procedure TVdxInference.RunLayerForwardBatch(const ALayer: Integer;
  const ANumTokens: UInt32);
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

  // Per-layer RoPE theta
  if ALayer mod 6 = 5 then
    LTheta := 1000000.0
  else
    LTheta := 10000.0;

  // Full batched attention: FWorkMat → FAttnOutMatBuf
  FAttn.ForwardBatch(FWorkMat, FAttnWeights[ALayer],
    FNormWeights[ALayer].QNormGpu, FNormWeights[ALayer].KNormGpu,
    ALayer, ANumTokens, LTheta,
    FQMat, FKMat, FVMat, FAttnOutMatBuf);

  // PostAttnNorm batch on attention output
  FNorm.ApplyBatch(FAttnOutMatBuf,
    FNormWeights[ALayer].PostAttnNormGpu, FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();

  // Vec-add batch: FResidualMat += FAttnOutMatBuf (count = N x HiddenDim)
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
  FAttn.BatchMatMul(FVindex.GetLayer(ALayer).GateGpuBuffer,
    FWorkMat, FGateMat, FHiddenDim, FFFNWidth, ANumTokens, FWeightType);

  // Up matmul: FWorkMat → FUpMatBuf
  FAttn.BatchMatMul(FUpWeights[ALayer],
    FWorkMat, FUpMatBuf, FHiddenDim, FFFNWidth, ANumTokens, FWeightType);
  FCompute.BatchBarrier();

  // GELU-mul batch: FGateMat = GELU(FGateMat) * FUpMatBuf (count = N x FFNWidth)
  LGeluPush.Count := ANumTokens * FFFNWidth;
  FCompute.DispatchComputeWithPush(
    FGeluMulBundle.Pipeline, FGeluMulBundle.PipelineLayout,
    FBatchGeluMulDescSet, @LGeluPush, SizeOf(LGeluPush),
    (ANumTokens * FFFNWidth + 255) div 256);
  FCompute.BatchBarrier();

  // Down matmul: FGateMat → FFFNOutMat
  FAttn.BatchMatMul(FVindex.GetLayer(ALayer).DownGpuBuffer,
    FGateMat, FFFNOutMat, FFFNWidth, FHiddenDim, ANumTokens, FWeightType);
  FCompute.BatchBarrier();

  // PostFFNNorm batch on FFN output
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

// ============================================================================
//  RunUnembedding — Apply final norm, GPU matvec for logits, CPU argmax
//  Called inside an active batch — records norm + matvec, then EndBatch
//  is called by the caller before downloading logits
// ============================================================================

function TVdxInference.RunUnembedding(): Integer;
var
  LBestId: Integer;
  LBestScore: Single;
  LLogits: PSingle;
  LI: Integer;
begin

  // Batched: fused copy+norm residual → work, matvec → logits
  FCompute.BeginBatch();

  // Fused copy+norm: residual → OutputNorm → WorkBufA (no separate copy)
  FNorm.ApplyCopy(FResidualGpu, FOutputNormGpu, FWorkBufA, FHiddenDim);
  FCompute.BatchBarrier(); // WorkBufA normed, ready for unembedding matvec
  FAttn.TestMatVec(FEmbedGpu, FWorkBufA, FLogitsBuf,
    FHiddenDim, UInt32(FVocabSize), FEmbedType);

  FCompute.EndBatch();

  // Download logits into pre-allocated VirtualBuffer (no heap allocation)
  FCompute.DownloadFromBuffer(FLogitsBuf, FLogitsVBuf.Memory,
    UInt64(FVocabSize) * SizeOf(Single));

  // Argmax over logits
  LLogits := PSingle(FLogitsVBuf.Memory);
  LBestId := 0;
  LBestScore := LLogits^;
  for LI := 1 to FVocabSize - 1 do
  begin
    if PSingle(PByte(LLogits) + UInt64(LI) * SizeOf(Single))^ > LBestScore then
    begin
      LBestId := LI;
      LBestScore := PSingle(PByte(LLogits) + UInt64(LI) * SizeOf(Single))^;
    end;
  end;

  Result := LBestId;
end;

// ============================================================================
//  LoadModel — Open GGUF, init Vulkan subsystems, upload all weights
// ============================================================================

procedure TVdxInference.LoadModel(const AGGUFPath: string);
var
  LLayer: Integer;
  LBufSize: UInt64;
  LSpvData: TBytes;
  LQInfo: TVdxGGUFTensorInfo;
  LProbeIds: TArray<Integer>;
  LProbeIdx: Integer;
begin
  TVdxUtils.FailIf(FModelLoaded,
    'Model already loaded — call UnloadModel() first', []);

  // Create subsystem objects
  FReader := TVdxGGUFReader.Create();
  FCompute := TVdxVulkanCompute.Create();
  FNorm := TVdxLayerNorm.Create();
  FAttn := TVdxAttention.Create();
  FVindex := TVdxVindex.Create();
  FTokenizer := TVdxTokenizer.Create();

  // Forward status callback to subsystems
  FReader.SetStatusCallback(FStatusCallback.Callback, FStatusCallback.UserData);
  FCompute.SetStatusCallback(FStatusCallback.Callback, FStatusCallback.UserData);
  FNorm.SetStatusCallback(FStatusCallback.Callback, FStatusCallback.UserData);

  // --- Open GGUF and read model config ---
  Status('Opening GGUF: %s', [AGGUFPath]);
  TVdxUtils.FailIf(not FReader.Open(AGGUFPath),
    'Failed to open GGUF: %s', [AGGUFPath]);

  // Detect architecture and read model dimensions from GGUF metadata
  FArchitecture := TVdxChatTemplate.DetectArchitecture(FReader);
  Status('Architecture: %s', [FArchitecture]);

  FNumLayers := FReader.GetMetadataUInt32(FArchitecture + '.block_count');
  FHiddenDim := FReader.GetMetadataUInt32(FArchitecture + '.embedding_length');
  FFFNWidth := FReader.GetMetadataUInt32(FArchitecture + '.feed_forward_length');
  FNumQHeads := FReader.GetMetadataUInt32(FArchitecture + '.attention.head_count');
  FNumKVHeads := FReader.GetMetadataUInt32(FArchitecture + '.attention.head_count_kv');

  // Derive head_dim from Q weight tensor shape
  LQInfo := FReader.GetTensorInfo('blk.0.attn_q.weight');
  FHeadDim := UInt32(LQInfo.Dimensions[1]) div FNumQHeads;
  FMaxSeqLen := 2048;

  // Detect weight tensor type (F16, Q4_0, etc.) from first layer's Q weight
  FWeightType := LQInfo.TensorType;
  Status('Weight type: %s', [VdxGGMLTypeName(FWeightType)]);

  // Detect embedding tensor type
  FEmbedType := FReader.GetTensorInfo('token_embd.weight').TensorType;
  Status('Embedding type: %s', [VdxGGMLTypeName(FEmbedType)]);

  Status('Config: layers=%d hidden=%d ffn=%d heads=%d/%d head_dim=%d',
    [FNumLayers, FHiddenDim, FFFNWidth, FNumQHeads, FNumKVHeads, FHeadDim]);

  // --- Init Vulkan and subsystems ---
  FCompute.Init();
  FNorm.Init(FCompute);
  FAttn.Init(FCompute, FHiddenDim, FNumQHeads, FNumKVHeads,
    FHeadDim, FNumLayers, FMaxSeqLen);
  TVdxUtils.FailIf(not FVindex.BuildFromGGUF(FReader),
    'Failed to build vindex', []);

  // Create work buffers
  LBufSize := UInt64(FHiddenDim) * SizeOf(Single);

  // GPU residual — lives on GPU between layers, host-visible for initial upload
  FResidualGpu := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Work buffer — receives copy of residual for in-place norming
  FWorkBufA := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Attention and FFN output buffers
  FAttnOutBuf := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  FFFNOutBuf := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Dense FFN scratch buffers (device-local for performance)
  FGateBuf := FCompute.CreateGpuBuffer(
    UInt64(FFFNWidth) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  FUpBuf := FCompute.CreateGpuBuffer(
    UInt64(FFFNWidth) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // --- GELU-mul pipeline (GELU with tanh approximation) ---
  LSpvData := VdxLoadShader('GELU_MUL');
  FGeluMulShader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  FGeluMulDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  FGeluMulBundle := FCompute.CreateComputePipelineWithPush(
    FGeluMulShader, 'main', FGeluMulDescLayout, SizeOf(TVdxGeluMulPush));

  FGeluMulDescPool := FCompute.CreateDescriptorPoolForStorage(1, 2);
  FGeluMulDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FGeluMulDescPool, FGeluMulDescLayout, [FGateBuf, FUpBuf]);

  // --- Vec-add pipeline (GPU residual += branch output) ---
  LSpvData := VdxLoadShader('VEC_ADD');
  FVecAddShader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  FVecAddDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  FVecAddBundle := FCompute.CreateComputePipelineWithPush(
    FVecAddShader, 'main', FVecAddDescLayout, SizeOf(TVdxVecAddPush));

  // Pre-allocate two descriptor sets: residual+=attn, residual+=ffn
  FVecAddDescPool := FCompute.CreateDescriptorPoolForStorage(2, 4);
  FVecAddAttnDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FVecAddDescPool, FVecAddDescLayout, [FResidualGpu, FAttnOutBuf]);
  FVecAddFFNDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FVecAddDescPool, FVecAddDescLayout, [FResidualGpu, FFFNOutBuf]);

  // --- Upload all weights to VRAM ---
  Status('Uploading weights to GPU...');

  // Gate + Down vectors (all layers via Vindex)
  Status('  Uploading gate/down vectors (%d layers)...', [FNumLayers]);
  FVindex.UploadAll(FCompute);

  // Attention weights (Q/K/V/O for all layers)
  Status('  Uploading attention weights (%d layers)...', [FNumLayers]);
  SetLength(FAttnWeights, FNumLayers);
  for LLayer := 0 to Integer(FNumLayers) - 1 do
    FAttn.UploadAttnWeights(FReader, LLayer, FAttnWeights[LLayer]);

  // FFN up weights (all layers)
  Status('  Uploading FFN up weights (%d layers)...', [FNumLayers]);
  SetLength(FUpWeights, FNumLayers);
  for LLayer := 0 to Integer(FNumLayers) - 1 do
    FUpWeights[LLayer] := UploadWeightTensor(
      Format('blk.%d.ffn_up.weight', [LLayer]));

  // Norm weights (all 6 per layer)
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

  // Output norm (global, not per-layer)
  FOutputNormGpu := UploadNormWeight('output_norm.weight', FHiddenDim);

  // --- Load tokenizer ---
  TVdxUtils.FailIf(not FTokenizer.LoadFromGGUF(FReader),
    'Failed to load tokenizer from GGUF', []);
  FVocabSize := FTokenizer.GetVocabSize();
  Status('Tokenizer loaded: %d tokens, BOS=%d, EOS=%d',
    [FVocabSize, FTokenizer.GetBosId(), FTokenizer.GetEosId()]);

  // Populate stop tokens from GGUF metadata
  FStopTokenIds.Clear();

  // EOS is always a stop token
  FStopTokenIds.Add(FTokenizer.GetEosId());

  // End-of-turn token from GGUF metadata (if present)
  if FReader.HasMetadata('tokenizer.ggml.eot_token_id') then
    FStopTokenIds.Add(
      Integer(FReader.GetMetadataUInt32('tokenizer.ggml.eot_token_id')));

  // Probe vocab for common end-of-turn tokens across model families
  for LProbeIdx := 0 to 4 do
  begin
    case LProbeIdx of
      0: LProbeIds := FTokenizer.Encode('<end_of_turn>', False);
      1: LProbeIds := FTokenizer.Encode('<|im_end|>', False);
      2: LProbeIds := FTokenizer.Encode('<|eot_id|>', False);
      3: LProbeIds := FTokenizer.Encode('<|end|>', False);
      4: LProbeIds := FTokenizer.Encode('<|endoftext|>', False);
    end;

    // Single-token result means it's a special token in the vocab
    if (Length(LProbeIds) = 1) and (not FStopTokenIds.Contains(LProbeIds[0])) then
    begin
      FStopTokenIds.Add(LProbeIds[0]);
      Status('  Found stop token: id=%d ("%s")',
        [LProbeIds[0], FTokenizer.GetTokenStr(LProbeIds[0])]);
    end;
  end;

  Status('Stop tokens: %d configured', [FStopTokenIds.Count]);

  // --- Get embedding table pointer (mmap'd data) ---
  FEmbedPtr := PByte(FReader.GetTensorDataPtr('token_embd.weight'));
  TVdxUtils.FailIf(FEmbedPtr = nil,
    'token_embd.weight not found in GGUF', []);
  TVdxUtils.FailIf((FEmbedType <> gtF16) and (FEmbedType <> gtF32) and (FEmbedType <> gtQ8_0),
    'Unsupported embedding type: %s (need F16, F32, or Q8_0)',
    [VdxGGMLTypeName(FEmbedType)]);
  FEmbedScale := Sqrt(Single(FHiddenDim));

  // Upload embedding table to GPU for fast unembedding
  Status('  Uploading embedding table to GPU...');
  FEmbedGpu := UploadWeightTensor('token_embd.weight');

  // --- GPU embedding lookup pipeline ---
  LSpvData := VdxLoadShader('EMBED_LOOKUP_F16');
  FEmbedF16Shader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  LSpvData := VdxLoadShader('EMBED_LOOKUP_Q8');
  FEmbedQ8Shader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  FEmbedDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  FEmbedF16Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedF16Shader, 'main', FEmbedDescLayout, SizeOf(TVdxEmbedLookupPush));
  FEmbedQ8Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedQ8Shader, 'main', FEmbedDescLayout, SizeOf(TVdxEmbedLookupPush));

  FEmbedDescPool := FCompute.CreateDescriptorPoolForStorage(1, 2);
  FEmbedDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FEmbedDescPool, FEmbedDescLayout, [FEmbedGpu, FResidualGpu]);

  // --- Batched GPU embedding lookup pipeline (Phase 6D) ---
  LSpvData := VdxLoadShader('EMBED_LOOKUP_BATCH_F16');
  FEmbedBatchF16Shader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  LSpvData := VdxLoadShader('EMBED_LOOKUP_BATCH_Q8');
  FEmbedBatchQ8Shader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  // 3 bindings: embed table, output matrix, token IDs
  FEmbedBatchDescLayout := FCompute.CreateStorageDescriptorSetLayout(3);
  FEmbedBatchF16Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedBatchF16Shader, 'main', FEmbedBatchDescLayout,
    SizeOf(TVdxEmbedBatchPush));
  FEmbedBatchQ8Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedBatchQ8Shader, 'main', FEmbedBatchDescLayout,
    SizeOf(TVdxEmbedBatchPush));

  // Token IDs buffer: host-visible, sized for max sequence length
  FTokenIdsGpu := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * SizeOf(UInt32),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Pre-allocate descriptor pool + set (rebound per call via UpdateDescriptorSetBuffers)
  FEmbedBatchDescPool := FCompute.CreateDescriptorPoolForStorage(1, 3);
  FEmbedBatchDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FEmbedBatchDescPool, FEmbedBatchDescLayout,
    [FEmbedGpu, FResidualGpu, FTokenIdsGpu]);

  // --- Batched prefill matrix buffers (Phase 6D) ---
  Status('  Allocating prefill matrix buffers...');

  FResidualMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FHiddenDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FWorkMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FHiddenDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FQMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FNumQHeads * FHeadDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FKMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FNumKVHeads * FHeadDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FVMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FNumKVHeads * FHeadDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FAttnOutMatBuf := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FHiddenDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FGateMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FFFNWidth * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FUpMatBuf := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FFFNWidth * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FFFNOutMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FHiddenDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Batch elementwise descriptor sets (vec-add + gelu-mul on matrices)
  // 3 sets, total bindings = 2+2+2 = 6
  FBatchEWDescPool := FCompute.CreateDescriptorPoolForStorage(3, 6);
  FBatchVecAddAttnDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FBatchEWDescPool, FVecAddDescLayout, [FResidualMat, FAttnOutMatBuf]);
  FBatchVecAddFFNDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FBatchEWDescPool, FVecAddDescLayout, [FResidualMat, FFFNOutMat]);
  FBatchGeluMulDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FBatchEWDescPool, FGeluMulDescLayout, [FGateMat, FUpMatBuf]);

  // Logits buffer: F32 x VocabSize, host-visible for CPU argmax
  FLogitsBuf := FCompute.CreateGpuBuffer(
    UInt64(FVocabSize) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Pre-allocate CPU logits buffer (reused every token, no heap churn)
  FLogitsVBuf := TVdxVirtualBuffer<Single>.Create(FVocabSize);

  // --- Allocate CPU embedding scratch array ---
  SetLength(FResidual, FHiddenDim);

  FModelLoaded := True;
  Status('Model loaded successfully');
end;

// ============================================================================
//  SetTokenCallback
// ============================================================================

procedure TVdxInference.SetTokenCallback(const ACallback: TVdxTokenCallback;
  const AUserData: Pointer);
begin
  FTokenCallback.Callback := ACallback;
  FTokenCallback.UserData := AUserData;
end;

// ============================================================================
//  AddStopToken / ClearStopTokens
// ============================================================================

procedure TVdxInference.AddStopToken(const ATokenId: Integer);
begin
  if not FStopTokenIds.Contains(ATokenId) then
    FStopTokenIds.Add(ATokenId);
end;

procedure TVdxInference.ClearStopTokens();
begin
  FStopTokenIds.Clear();

  // EOS always remains as a stop token
  if FModelLoaded then
    FStopTokenIds.Add(FTokenizer.GetEosId());
end;

// ============================================================================
//  GetStats — Return pointer to internal stats record
// ============================================================================

function TVdxInference.GetStats(): PVdxInferenceStats;
begin
  Result := @FStats;
end;

// ============================================================================
//  Generate — Tokenize, prefill, then autoregressive token generation
//  Uses batched dispatch: one GPU submit per token for all 34 layers
// ============================================================================

function TVdxInference.Generate(const APrompt: string;
  const AMaxTokens: Integer): string;
var
  LFormatted: string;
  LTokenIds: TArray<Integer>;
  LTokenCount: Integer;
  LLayer: Integer;
  LNextTokenId: Integer;
  LTokenStr: string;
  LResult: TStringBuilder;
  LGenerated: Integer;
  LTotalWatch: TStopwatch;
  LPrefillWatch: TStopwatch;
  LGenWatch: TStopwatch;
begin
  TVdxUtils.FailIf(not FModelLoaded, 'Model not loaded', []);

  // Reset stats
  FillChar(FStats, SizeOf(FStats), 0);
  FStats.StopReason := srNone;

  // Format prompt with chat template
  LFormatted := TVdxChatTemplate.FormatPrompt(FArchitecture, APrompt);

  // Tokenize (with BOS)
  LTokenIds := FTokenizer.Encode(LFormatted, True);
  LTokenCount := Length(LTokenIds);

  LTotalWatch := TStopwatch.StartNew();
  LResult := TStringBuilder.Create();
  try
    // --- Prefill: batch all prompt tokens through the model (Phase 6D) ---
    LPrefillWatch := TStopwatch.StartNew();

    FCompute.BeginBatch();
    EmbedTokensBatch(LTokenIds, LTokenCount, FResidualMat);
    for LLayer := 0 to Integer(FNumLayers) - 1 do
      RunLayerForwardBatch(LLayer, UInt32(LTokenCount));
    FCompute.EndBatch();

    // Copy last token's residual from matrix to vector for generation handoff
    FCompute.CopyBufferRegion(
      FResidualMat,
      UInt64(LTokenCount - 1) * UInt64(FHiddenDim) * SizeOf(Single),
      FResidualGpu, 0,
      UInt64(FHiddenDim) * SizeOf(Single));

    LPrefillWatch.Stop();

    // --- Autoregressive generation ---
    LGenWatch := TStopwatch.StartNew();
    LGenerated := 0;
    while LGenerated < AMaxTokens do
    begin
      LNextTokenId := RunUnembedding();

      // Check for stop tokens
      if FStopTokenIds.Contains(LNextTokenId) then
      begin
        if LNextTokenId = FTokenizer.GetEosId() then
          FStats.StopReason := srEOS
        else
          FStats.StopReason := srStopToken;
        Break;
      end;

      // Decode token and append to result
      LTokenStr := FTokenizer.Decode(
        TArray<Integer>.Create(LNextTokenId));
      LResult.Append(LTokenStr);

      // Call token callback if assigned
      if FTokenCallback.IsAssigned() then
      begin
        if not FTokenCallback.Callback(LTokenStr, FTokenCallback.UserData) then
        begin
          FStats.StopReason := srCallbackStopped;
          Inc(LGenerated);
          Break;
        end;
      end;

      // Feed predicted token back into the model
      Inc(LGenerated);

      FCompute.BeginBatch();
      EmbedToken(LNextTokenId);
      for LLayer := 0 to Integer(FNumLayers) - 1 do
        RunLayerForward(LLayer, LTokenCount + LGenerated - 1);
      FCompute.EndBatch();
    end;
    LGenWatch.Stop();

    // Max tokens reached without a stop token
    if FStats.StopReason = srNone then
      FStats.StopReason := srMaxTokens;

    LTotalWatch.Stop();

    // Fill stats
    FStats.PrefillTokens := LTokenCount;
    FStats.PrefillTimeMs := LPrefillWatch.Elapsed.TotalMilliseconds;
    if FStats.PrefillTimeMs > 0 then
      FStats.PrefillTokPerSec := (LTokenCount * 1000.0) / FStats.PrefillTimeMs;

    FStats.GeneratedTokens := LGenerated;
    FStats.GenerationTimeMs := LGenWatch.Elapsed.TotalMilliseconds;
    if FStats.GenerationTimeMs > 0 then
      FStats.GenerationTokPerSec := (LGenerated * 1000.0) / FStats.GenerationTimeMs;

    FStats.TimeToFirstTokenMs := LPrefillWatch.Elapsed.TotalMilliseconds;
    FStats.TotalTimeMs := LTotalWatch.Elapsed.TotalMilliseconds;

    Result := LResult.ToString();
  finally
    LResult.Free();
  end;
end;

// ============================================================================
//  UnloadModel — Release all GPU resources and destroy subsystem objects
// ============================================================================

procedure TVdxInference.UnloadModel();
var
  LLayer: Integer;
begin
  if not FModelLoaded then
    Exit;

  // Free GELU-mul pipeline
  FCompute.DestroyDescriptorPoolHandle(FGeluMulDescPool);
  FCompute.DestroyComputePipelineBundle(FGeluMulBundle);
  FCompute.DestroyDescriptorSetLayoutHandle(FGeluMulDescLayout);
  FCompute.DestroyShaderModuleHandle(FGeluMulShader);

  // Free vec-add pipeline
  FCompute.DestroyDescriptorPoolHandle(FVecAddDescPool);
  FCompute.DestroyComputePipelineBundle(FVecAddBundle);
  FCompute.DestroyDescriptorSetLayoutHandle(FVecAddDescLayout);
  FCompute.DestroyShaderModuleHandle(FVecAddShader);

  // Free embed lookup pipeline
  FCompute.DestroyDescriptorPoolHandle(FEmbedDescPool);
  FCompute.DestroyComputePipelineBundle(FEmbedF16Bundle);
  FCompute.DestroyComputePipelineBundle(FEmbedQ8Bundle);
  FCompute.DestroyDescriptorSetLayoutHandle(FEmbedDescLayout);
  FCompute.DestroyShaderModuleHandle(FEmbedF16Shader);
  FCompute.DestroyShaderModuleHandle(FEmbedQ8Shader);

  // Free batched embed pipeline (Phase 6D)
  FCompute.DestroyDescriptorPoolHandle(FEmbedBatchDescPool);
  FCompute.DestroyComputePipelineBundle(FEmbedBatchF16Bundle);
  FCompute.DestroyComputePipelineBundle(FEmbedBatchQ8Bundle);
  FCompute.DestroyDescriptorSetLayoutHandle(FEmbedBatchDescLayout);
  FCompute.DestroyShaderModuleHandle(FEmbedBatchF16Shader);
  FCompute.DestroyShaderModuleHandle(FEmbedBatchQ8Shader);
  FCompute.DestroyGpuBuffer(FTokenIdsGpu);

  // Free batched prefill matrix buffers (Phase 6D)
  FCompute.DestroyDescriptorPoolHandle(FBatchEWDescPool);
  FCompute.DestroyGpuBuffer(FResidualMat);
  FCompute.DestroyGpuBuffer(FWorkMat);
  FCompute.DestroyGpuBuffer(FQMat);
  FCompute.DestroyGpuBuffer(FKMat);
  FCompute.DestroyGpuBuffer(FVMat);
  FCompute.DestroyGpuBuffer(FAttnOutMatBuf);
  FCompute.DestroyGpuBuffer(FGateMat);
  FCompute.DestroyGpuBuffer(FUpMatBuf);
  FCompute.DestroyGpuBuffer(FFFNOutMat);

  // Free work buffers
  FCompute.DestroyGpuBuffer(FResidualGpu);
  FCompute.DestroyGpuBuffer(FWorkBufA);
  FCompute.DestroyGpuBuffer(FAttnOutBuf);
  FCompute.DestroyGpuBuffer(FGateBuf);
  FCompute.DestroyGpuBuffer(FUpBuf);
  FCompute.DestroyGpuBuffer(FFFNOutBuf);

  // Free global weights
  FCompute.DestroyGpuBuffer(FOutputNormGpu);

  // Free embedding GPU + logits buffer
  FCompute.DestroyGpuBuffer(FEmbedGpu);
  FCompute.DestroyGpuBuffer(FLogitsBuf);

  // Free pre-allocated CPU logits buffer
  FreeAndNil(FLogitsVBuf);

  // Free per-layer weights
  for LLayer := 0 to Integer(FNumLayers) - 1 do
  begin
    FAttn.FreeAttnWeights(FAttnWeights[LLayer]);

    if FUpWeights[LLayer].Buffer <> VK_NULL_HANDLE then
      FCompute.DestroyGpuBuffer(FUpWeights[LLayer]);

    FCompute.DestroyGpuBuffer(FNormWeights[LLayer].AttnNormGpu);
    FCompute.DestroyGpuBuffer(FNormWeights[LLayer].PostAttnNormGpu);
    FCompute.DestroyGpuBuffer(FNormWeights[LLayer].FFNNormGpu);
    FCompute.DestroyGpuBuffer(FNormWeights[LLayer].PostFFNNormGpu);
    FCompute.DestroyGpuBuffer(FNormWeights[LLayer].QNormGpu);
    FCompute.DestroyGpuBuffer(FNormWeights[LLayer].KNormGpu);
  end;

  // Free gate/down via Vindex
  FVindex.FreeAllGpu(FCompute);

  // Cleanup subsystems
  FAttn.Cleanup();
  FNorm.Cleanup();
  FReader.Close();

  // Destroy subsystem objects
  FreeAndNil(FTokenizer);
  FreeAndNil(FVindex);
  FreeAndNil(FAttn);
  FreeAndNil(FNorm);
  FreeAndNil(FCompute);
  FreeAndNil(FReader);

  // Clear state
  FAttnWeights := nil;
  FNormWeights := nil;
  FUpWeights := nil;
  FResidual := nil;
  FEmbedPtr := nil;
  FModelLoaded := False;

  Status('Model unloaded');
end;

end.
