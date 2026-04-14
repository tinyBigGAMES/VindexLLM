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
  System.Generics.Collections,
  VindexLLM.Utils,
  VindexLLM.GGUFReader,
  VindexLLM.VulkanCompute,
  VindexLLM.LayerNorm,
  VindexLLM.Attention,
  VindexLLM.Vindex,
  VindexLLM.Tokenizer,
  VindexLLM.ChatTemplate;

// ============================================================================
//  Push constant for GELU-mul shader
// ============================================================================

type
  TVdxGeluMulPush = record
    Count: UInt32;
  end;

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
    // Populated from GGUF metadata (EOS + EOT), user can add more
    FStopTokenIds: TList<Integer>;

    // Embedding (mmap'd from GGUF, F16)
    FEmbedPtr: PByte;
    FEmbedScale: Single;

    // Per-layer weights
    FAttnWeights: array of TVdxAttnLayerWeights;
    FNormWeights: array of TVdxNormLayerWeights;
    FUpWeights: array of TVdxGpuBuffer;

    // Global weights
    FOutputNormGpu: TVdxGpuBuffer;

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

    // GPU unembedding (avoids slow CPU F16 scan)
    FEmbedGpu: TVdxGpuBuffer;     // F16 embedding table on GPU [HiddenDim x VocabSize]
    FLogitsBuf: TVdxGpuBuffer;    // F32 logits output [VocabSize], host-visible

    // CPU-side arrays
    FResidual: array of Single;
    FSavedResidual: array of Single;
    FTempVec: array of Single;

    // Private helpers
    function F16ToF32(const AVal: UInt16): Single;
    function UploadNormWeight(const ATensorName: string;
      const ACount: UInt32): TVdxGpuBuffer;
    function UploadF16Tensor(const ATensorName: string): TVdxGpuBuffer;
    procedure EmbedToken(const ATokenId: Integer);
    procedure RunLayerForward(const ALayer: Integer;
      const APosition: Integer);
    function RunUnembedding(): Integer;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Load model from GGUF file — sets up Vulkan, loads all weights
    procedure LoadModel(const AGGUFPath: string);

    // Set token callback (called per generated token, return False to stop)
    procedure SetTokenCallback(const ACallback: TVdxTokenCallback;
      const AUserData: Pointer);

    // Generate response from prompt string
    // Auto-wraps in chat template based on model architecture
    // Returns full generated text
    function Generate(const APrompt: string;
      const AMaxTokens: Integer = 256): string;

    // Release all resources
    procedure UnloadModel();

    // Add a custom stop token ID (checked during generation)
    procedure AddStopToken(const ATokenId: Integer);

    // Clear all user-added stop tokens (EOS/EOT from GGUF remain)
    procedure ClearStopTokens();
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
//  UploadF16Tensor — Upload raw F16 tensor to device-local GPU via staging
// ============================================================================

function TVdxInference.UploadF16Tensor(const ATensorName: string): TVdxGpuBuffer;
var
  LInfo: TVdxGGUFTensorInfo;
  LPtr: Pointer;
  LSize: UInt64;
  LStaging: TVdxGpuBuffer;
begin
  LInfo := FReader.GetTensorInfo(ATensorName);
  LPtr := FReader.GetTensorDataPtr(ATensorName);
  LSize := UInt64(LInfo.Dimensions[0]) * UInt64(LInfo.Dimensions[1]) * 2;

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
//  EmbedToken — Look up F16 embedding row, convert to F32, scale
// ============================================================================

procedure TVdxInference.EmbedToken(const ATokenId: Integer);
var
  LI: Integer;
begin
  for LI := 0 to Integer(FHiddenDim) - 1 do
    FResidual[LI] := F16ToF32(
      PWord(FEmbedPtr +
        UInt64(ATokenId) * UInt64(FHiddenDim) * 2 +
        UInt64(LI) * 2)^
    ) * FEmbedScale;
end;

// ============================================================================
//  RunLayerForward — Process one transformer layer (attn + FFN)
//  Sandwich norm pattern: PreNorm → Op → PostNorm → residual add
// ============================================================================

procedure TVdxInference.RunLayerForward(const ALayer: Integer;
  const APosition: Integer);
var
  LBufSize: UInt64;
  LTheta: Single;
  LGeluPush: TVdxGeluMulPush;
  LI: Integer;
begin
  LBufSize := UInt64(FHiddenDim) * SizeOf(Single);

  // === Attention branch: x = x + PostAttnNorm(Attn(PreAttnNorm(x))) ===

  // Save residual for the residual connection
  Move(FResidual[0], FSavedResidual[0], LBufSize);

  // Upload residual to GPU, apply PreAttnNorm in-place
  FCompute.UploadToBuffer(FWorkBufA, @FResidual[0], LBufSize);
  FNorm.Apply(FWorkBufA, FNormWeights[ALayer].AttnNormGpu, FHiddenDim);

  // Full layers [5,11,17,23,29] use theta=1M, sliding layers use theta=10K
  if ALayer mod 6 = 5 then
    LTheta := 1000000.0
  else
    LTheta := 10000.0;

  // Run full attention: normed input → attention output
  FAttn.Forward(
    FWorkBufA,
    FAttnWeights[ALayer],
    FNormWeights[ALayer].QNormGpu,
    FNormWeights[ALayer].KNormGpu,
    ALayer,
    APosition,
    LTheta,
    FAttnOutBuf);

  // Apply PostAttnNorm on attention output in-place
  FNorm.Apply(FAttnOutBuf,
    FNormWeights[ALayer].PostAttnNormGpu, FHiddenDim);

  // Download post-normed attention output, add to saved residual
  FCompute.DownloadFromBuffer(FAttnOutBuf, @FTempVec[0], LBufSize);
  for LI := 0 to Integer(FHiddenDim) - 1 do
    FResidual[LI] := FSavedResidual[LI] + FTempVec[LI];

  // === FFN branch: x = x + PostFFNNorm(FFN(PreFFNNorm(x))) ===

  // Save residual for the residual connection
  Move(FResidual[0], FSavedResidual[0], LBufSize);

  // Upload residual to GPU, apply PreFFNNorm in-place
  FCompute.UploadToBuffer(FWorkBufA, @FResidual[0], LBufSize);
  FNorm.Apply(FWorkBufA, FNormWeights[ALayer].FFNNormGpu, FHiddenDim);

  // Dense FFN: gate(x), up(x), GELU(gate)*up, down(hidden)

  // gate(x) → FGateBuf [FFNWidth]
  FAttn.TestMatVec(FVindex.GetLayer(ALayer).GateGpuBuffer,
    FWorkBufA, FGateBuf, FHiddenDim, FFFNWidth);

  // up(x) → FUpBuf [FFNWidth]
  FAttn.TestMatVec(FUpWeights[ALayer],
    FWorkBufA, FUpBuf, FHiddenDim, FFFNWidth);

  // GELU(gate) * up → FGateBuf [FFNWidth] in-place
  LGeluPush.Count := FFFNWidth;
  FCompute.DispatchComputeWithPush(
    FGeluMulBundle.Pipeline, FGeluMulBundle.PipelineLayout,
    FGeluMulDescSet, @LGeluPush, SizeOf(LGeluPush),
    (FFFNWidth + 255) div 256);

  // down(hidden) → FFFNOutBuf [HiddenDim]
  FAttn.TestMatVec(FVindex.GetLayer(ALayer).DownGpuBuffer,
    FGateBuf, FFFNOutBuf, FFFNWidth, FHiddenDim);

  // Apply PostFFNNorm to FFN output
  FNorm.Apply(FFFNOutBuf,
    FNormWeights[ALayer].PostFFNNormGpu, FHiddenDim);

  // Download post-normed FFN output, add to saved residual
  FCompute.DownloadFromBuffer(FFFNOutBuf, @FTempVec[0], LBufSize);
  for LI := 0 to Integer(FHiddenDim) - 1 do
    FResidual[LI] := FSavedResidual[LI] + FTempVec[LI];
end;

// ============================================================================
//  RunUnembedding — Apply final norm, scan vocab for argmax (greedy)
//  Returns the token ID with the highest logit score
// ============================================================================

function TVdxInference.RunUnembedding(): Integer;
var
  LBufSize: UInt64;
  LBestId: Integer;
  LBestScore: Single;
  LLogits: array of Single;
  LI: Integer;
begin
  LBufSize := UInt64(FHiddenDim) * SizeOf(Single);

  // Apply final RMSNorm (output_norm) to the residual
  FCompute.UploadToBuffer(FWorkBufA, @FResidual[0], LBufSize);
  FNorm.Apply(FWorkBufA, FOutputNormGpu, FHiddenDim);

  // GPU matvec: embed_table[HiddenDim x VocabSize] × normed_residual[HiddenDim]
  // → logits[VocabSize] in one dispatch
  FAttn.TestMatVec(FEmbedGpu, FWorkBufA, FLogitsBuf,
    FHiddenDim, UInt32(FVocabSize));

  // Download logits and find argmax (simple F32 scan)
  SetLength(LLogits, FVocabSize);
  FCompute.DownloadFromBuffer(FLogitsBuf, @LLogits[0],
    UInt64(FVocabSize) * SizeOf(Single));

  LBestId := 0;
  LBestScore := LLogits[0];
  for LI := 1 to FVocabSize - 1 do
  begin
    if LLogits[LI] > LBestScore then
    begin
      LBestId := LI;
      LBestScore := LLogits[LI];
    end;
  end;

  // Download normed residual back for next iteration's embedding overwrite
  FCompute.DownloadFromBuffer(FWorkBufA, @FResidual[0], LBufSize);

  Result := LBestId;
end;

// ============================================================================
//  LoadModel — Open GGUF, init Vulkan subsystems, upload all weights
// ============================================================================

procedure TVdxInference.LoadModel(const AGGUFPath: string);
var
  LLayer: Integer;
  LBufSize: UInt64;
  LSpvPath: string;
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

  // Derive head_dim from Q weight tensor shape: [hidden_dim, num_q_heads * head_dim]
  LQInfo := FReader.GetTensorInfo('blk.0.attn_q.weight');
  FHeadDim := UInt32(LQInfo.Dimensions[1]) div FNumQHeads;

  // Max sequence length for KV cache allocation
  FMaxSeqLen := 2048;

  Status('Config: layers=%d hidden=%d ffn=%d heads=%d/%d head_dim=%d',
    [FNumLayers, FHiddenDim, FFFNWidth, FNumQHeads, FNumKVHeads, FHeadDim]);

  // --- Init Vulkan and subsystems ---
  FCompute.Init();
  FNorm.Init(FCompute);
  FAttn.Init(FCompute, FHiddenDim, FNumQHeads, FNumKVHeads,
    FHeadDim, FNumLayers, FMaxSeqLen);
  TVdxUtils.FailIf(not FVindex.BuildFromGGUF(FReader),
    'Failed to build vindex', []);

  // Create work buffers (host-visible coherent for upload/download)
  LBufSize := UInt64(FHiddenDim) * SizeOf(Single);
  FWorkBufA := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
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
  LSpvPath := TPath.Combine(TPath.GetDirectoryName(ParamStr(0)),
    '..\shaders\gelu_mul.spv');
  LSpvPath := TPath.GetFullPath(LSpvPath);
  LSpvData := TFile.ReadAllBytes(LSpvPath);
  FGeluMulShader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  FGeluMulDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  FGeluMulBundle := FCompute.CreateComputePipelineWithPush(
    FGeluMulShader, 'main', FGeluMulDescLayout, SizeOf(TVdxGeluMulPush));

  // Pre-allocate GELU descriptor set (buffers are fixed)
  FGeluMulDescPool := FCompute.CreateDescriptorPoolForStorage(1, 2);
  FGeluMulDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FGeluMulDescPool, FGeluMulDescLayout, [FGateBuf, FUpBuf]);

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
    FUpWeights[LLayer] := UploadF16Tensor(
      Format('blk.%d.ffn_up.weight', [LLayer]));

  // Norm weights (all 6 per layer: pre/post attn, pre/post FFN, QK-norm)
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
  // Data-driven: only adds tokens that actually exist in this model's vocab
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

  // --- Get embedding table pointer (mmap'd F16 data) ---
  FEmbedPtr := PByte(FReader.GetTensorDataPtr('token_embd.weight'));
  TVdxUtils.FailIf(FEmbedPtr = nil,
    'token_embd.weight not found in GGUF', []);
  FEmbedScale := Sqrt(Single(FHiddenDim));

  // Upload embedding table to GPU for fast unembedding (F16, ~1.28 GB)
  Status('  Uploading embedding table to GPU...');
  FEmbedGpu := UploadF16Tensor('token_embd.weight');

  // Logits buffer: F32 x VocabSize, host-visible for CPU argmax
  FLogitsBuf := FCompute.CreateGpuBuffer(
    UInt64(FVocabSize) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // --- Allocate CPU arrays ---
  SetLength(FResidual, FHiddenDim);
  SetLength(FSavedResidual, FHiddenDim);
  SetLength(FTempVec, FHiddenDim);

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
//  Generate — Tokenize, prefill, then autoregressive token generation
// ============================================================================

function TVdxInference.Generate(const APrompt: string;
  const AMaxTokens: Integer): string;
var
  LFormatted: string;
  LTokenIds: TArray<Integer>;
  LTokenCount: Integer;
  LPos: Integer;
  LLayer: Integer;
  LNextTokenId: Integer;
  LTokenStr: string;
  LResult: TStringBuilder;
  LGenerated: Integer;
begin
  TVdxUtils.FailIf(not FModelLoaded, 'Model not loaded', []);

  // Format prompt with chat template
  LFormatted := TVdxChatTemplate.FormatPrompt(FArchitecture, APrompt);

  // Tokenize (with BOS)
  LTokenIds := FTokenizer.Encode(LFormatted, True);
  LTokenCount := Length(LTokenIds);
  Status('Prompt: %d tokens, generating up to %d tokens',
    [LTokenCount, AMaxTokens]);

  LResult := TStringBuilder.Create();
  try
    // --- Prefill: process all prompt tokens through the model ---
    for LPos := 0 to LTokenCount - 1 do
    begin
      EmbedToken(LTokenIds[LPos]);
      for LLayer := 0 to Integer(FNumLayers) - 1 do
        RunLayerForward(LLayer, LPos);
    end;

    // --- Autoregressive generation ---
    LGenerated := 0;
    while LGenerated < AMaxTokens do
    begin
      // Unembedding: find top-1 token (greedy)
      LNextTokenId := RunUnembedding();

      // Check for stop tokens (EOS, EOT, user-defined)
      if FStopTokenIds.Contains(LNextTokenId) then
        Break;

      // Decode token and append to result
      LTokenStr := FTokenizer.Decode(
        TArray<Integer>.Create(LNextTokenId));
      LResult.Append(LTokenStr);

      // Call token callback if assigned
      if FTokenCallback.IsAssigned() then
      begin
        if not FTokenCallback.Callback(LTokenStr, FTokenCallback.UserData) then
          Break;
      end;

      // Feed predicted token back into the model
      Inc(LGenerated);
      EmbedToken(LNextTokenId);
      for LLayer := 0 to Integer(FNumLayers) - 1 do
        RunLayerForward(LLayer, LTokenCount + LGenerated - 1);
    end;

    Status('Generated %d tokens', [LGenerated]);
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

  // Free work buffers
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
  FSavedResidual := nil;
  FTempVec := nil;
  FEmbedPtr := nil;
  FModelLoaded := False;

  Status('Model unloaded');
end;

end.
