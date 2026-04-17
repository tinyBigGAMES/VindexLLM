{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Attention;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  VindexLLM.Utils,
  VindexLLM.GGUFReader,
  VindexLLM.Vulkan,
  VindexLLM.Compute;

type

  { TVdxMatVecF16Push }
  TVdxMatVecF16Push = record
    InDimHalf: UInt32;
    OutDim: UInt32;
  end;

  { TVdxQKNormPush }
  TVdxQKNormPush = record
    HeadDim: UInt32;
    NumHeads: UInt32;
    Eps: Single;
  end;

  { TVdxRoPEPush }
  TVdxRoPEPush = record
    HeadDim: UInt32;
    NumHeads: UInt32;
    Position: UInt32;
    ThetaBase: Single;
  end;

  { TVdxAttnScoresMHPush }
  TVdxAttnScoresMHPush = record
    HeadDim: UInt32;
    SeqLen: UInt32;
    MaxSeq: UInt32;
    Scale: Single;
    NumQHeads: UInt32;
    GqaRatio: UInt32;
  end;

  { TVdxAttnScoresMHTQ3Push }
  TVdxAttnScoresMHTQ3Push = record
    HeadDim: UInt32;
    SeqLen: UInt32;
    MaxSeq: UInt32;
    Scale: Single;
    NumQHeads: UInt32;
    GqaRatio: UInt32;
    BlocksPerHead: UInt32;
  end;

  { TVdxSoftmaxMHPush }
  TVdxSoftmaxMHPush = record
    SeqLen: UInt32;
    MaxSeq: UInt32;
    NumQHeads: UInt32;
  end;

  { TVdxAttnValueMHPush }
  TVdxAttnValueMHPush = record
    HeadDim: UInt32;
    SeqLen: UInt32;
    MaxSeq: UInt32;
    NumQHeads: UInt32;
    GqaRatio: UInt32;
  end;

  { TVdxKVCacheStorePush }
  TVdxKVCacheStorePush = record
    HeadDim: UInt32;
    MaxSeq: UInt32;
    Position: UInt32;
    NumKVHeads: UInt32;
  end;

  { TVdxMatMulPush }
  TVdxMatMulPush = record
    InDimParam: UInt32;   // in_dim/2 for F16, in_dim for Q8_0
    OutDim: UInt32;
    NumTokens: UInt32;
  end;

  { TVdxRoPEBatchPush }
  TVdxRoPEBatchPush = record
    HeadDim: UInt32;
    NumHeads: UInt32;
    NumTokens: UInt32;
    ThetaBase: Single;
    StartPos: UInt32;
  end;

  { TVdxKVCacheStoreBatchPush }
  TVdxKVCacheStoreBatchPush = record
    HeadDim: UInt32;
    MaxSeq: UInt32;
    NumKVHeads: UInt32;
    NumTokens: UInt32;
    StartPos: UInt32;
  end;

  { TVdxTQ3KVQuantPush }
  TVdxTQ3KVQuantPush = record
    BlocksPerHead: UInt32;
    MaxSeq: UInt32;
    Position: UInt32;
    NumHeads: UInt32;
  end;

  { TVdxTQ3KVDequantPush }
  TVdxTQ3KVDequantPush = record
    BlocksPerHead: UInt32;
    MaxSeq: UInt32;
    SeqLen: UInt32;
    NumHeads: UInt32;
  end;

  { TVdxKVStoreBatchTQ3Push }
  TVdxKVStoreBatchTQ3Push = record
    HeadDim: UInt32;
    MaxSeq: UInt32;
    NumHeads: UInt32;
    NumTokens: UInt32;
    StartPos: UInt32;
  end;

  { TVdxAttnScoresPrefillPush }
  TVdxAttnScoresPrefillPush = record
    HeadDim: UInt32;
    NumTokens: UInt32;
    MaxSeq: UInt32;
    Scale: Single;
    NumQHeads: UInt32;
    GqaRatio: UInt32;
    StartPos: UInt32;
    SeqLen: UInt32;
  end;

  { TVdxSoftmaxPrefillPush }
  TVdxSoftmaxPrefillPush = record
    NumTokens: UInt32;
    NumQHeads: UInt32;
    MaxSeq: UInt32;
    SeqLen: UInt32;
    StartPos: UInt32;
  end;

  { TVdxAttnValuePrefillPush }
  TVdxAttnValuePrefillPush = record
    HeadDim: UInt32;
    NumTokens: UInt32;
    MaxSeq: UInt32;
    NumQHeads: UInt32;
    GqaRatio: UInt32;
    StartPos: UInt32;
    SeqLen: UInt32;
  end;

  { TVdxAttnLayerWeights }
  TVdxAttnLayerWeights = record
    QWeightGpu: TVdxGpuBuffer;   // F16 or Q4_0 [2560 x 2048] = Q projection
    KWeightGpu: TVdxGpuBuffer;   // F16 or Q4_0 [2560 x 1024] = K projection
    VWeightGpu: TVdxGpuBuffer;   // F16 or Q4_0 [2560 x 1024] = V projection
    OWeightGpu: TVdxGpuBuffer;   // F16 or Q4_0 [2048 x 2560] = output projection
    WeightType: TVdxGGMLType;    // tensor format (gtF16 or gtQ4_0)
  end;

  { TVdxAttention }
  TVdxAttention = class(TVdxBaseObject)
  private
    FCompute: TVdxVulkanCompute;

    // Shader modules
    FMatVecShader: VkShaderModule;
    FMatVecQ8Shader: VkShaderModule;
    FMatVecQ4Shader: VkShaderModule;
    FQKNormShader: VkShaderModule;
    FRoPEShader: VkShaderModule;
    FAttnScoresMHShader: VkShaderModule;
    FSoftmaxMHShader: VkShaderModule;
    FAttnValueMHShader: VkShaderModule;
    // Pipeline bundles
    FMatVecBundle: TVdxComputePipelineBundle;
    FMatVecQ8Bundle: TVdxComputePipelineBundle;
    FMatVecQ4Bundle: TVdxComputePipelineBundle;
    FQKNormBundle: TVdxComputePipelineBundle;
    FRoPEBundle: TVdxComputePipelineBundle;
    FAttnScoresMHBundle: TVdxComputePipelineBundle;
    FSoftmaxMHBundle: TVdxComputePipelineBundle;
    FAttnValueMHBundle: TVdxComputePipelineBundle;

    // Descriptor set layouts
    FMatVecDescLayout: VkDescriptorSetLayout;      // 3 bindings
    FQKNormDescLayout: VkDescriptorSetLayout;      // 2 bindings
    FRoPEDescLayout: VkDescriptorSetLayout;        // 1 binding
    FAttnScoresDescLayout: VkDescriptorSetLayout;  // 3 bindings
    FSoftmaxDescLayout: VkDescriptorSetLayout;     // 1 binding
    FAttnValueDescLayout: VkDescriptorSetLayout;   // 3 bindings

    // Pre-allocated descriptor pool + reusable sets (eliminates per-dispatch churn)
    FDescPool: VkDescriptorPool;
    FMatVecDescSet: VkDescriptorSet;
    FQKNormDescSet: VkDescriptorSet;
    FRoPEDescSet: VkDescriptorSet;
    FScoresDescSet: VkDescriptorSet;
    FSoftmaxDescSet: VkDescriptorSet;
    FValueDescSet: VkDescriptorSet;

    // KV cache store shader (replaces per-head CopyBufferRegion calls)
    FKVStoreShader: VkShaderModule;
    FKVStoreBundle: TVdxComputePipelineBundle;
    FKVStoreDescLayout: VkDescriptorSetLayout;  // 4 bindings
    FKVStoreDescPool: VkDescriptorPool;
    FKVStoreDescSet: VkDescriptorSet;

    // Batch matmul shaders + pipelines (prefill batching — Phase 6D)
    FMatMulF16Shader: VkShaderModule;
    FMatMulQ8Shader: VkShaderModule;
    FMatMulQ4Shader: VkShaderModule;
    FMatMulF16Bundle: TVdxComputePipelineBundle;
    FMatMulQ8Bundle: TVdxComputePipelineBundle;
    FMatMulQ4Bundle: TVdxComputePipelineBundle;

    // Prefill attention shaders + pipelines (Phase 6D)
    FRoPEBatchShader: VkShaderModule;
    FKVStoreBatchShader: VkShaderModule;
    FScoresPrefillShader: VkShaderModule;
    FSoftmaxPrefillShader: VkShaderModule;
    FValuePrefillShader: VkShaderModule;
    FRoPEBatchBundle: TVdxComputePipelineBundle;
    FKVStoreBatchBundle: TVdxComputePipelineBundle;
    FScoresPrefillBundle: TVdxComputePipelineBundle;
    FSoftmaxPrefillBundle: TVdxComputePipelineBundle;
    FValuePrefillBundle: TVdxComputePipelineBundle;

    // Prefill descriptor pool + reusable sets (separate from single-token pool)
    FPrefillDescPool: VkDescriptorPool;
    FPrefillRoPEDescSet: VkDescriptorSet;       // 1 binding (data)
    FPrefillKVStoreDescSet: VkDescriptorSet;    // 4 bindings (K, V, KCache, VCache)
    FPrefillScoresDescSet: VkDescriptorSet;     // 3 bindings (Q, KCache, Scores)
    FPrefillSoftmaxDescSet: VkDescriptorSet;    // 1 binding (Scores)
    FPrefillValueDescSet: VkDescriptorSet;      // 3 bindings (Scores, VCache, Output)

    // Pre-allocated prefill scores buffer [NumQHeads x MaxSeq x MaxSeq] F32
    FPrefillScoresBuf: TVdxGpuBuffer;

    // TQ3 KV cache compression (Phase 2)
    FTQ3KVQuantShader: VkShaderModule;
    FTQ3KVDequantShader: VkShaderModule;
    FTQ3KVQuantBundle: TVdxComputePipelineBundle;
    FTQ3KVDequantBundle: TVdxComputePipelineBundle;
    FTQ3KVDescLayout: VkDescriptorSetLayout;     // 2 bindings: source + dest
    FTQ3KVDescPool: VkDescriptorPool;
    FTQ3KVQuantDescSet: VkDescriptorSet;          // for quantize dispatch
    FTQ3KVDequantDescSet: VkDescriptorSet;        // for dequantize dispatch

    // Fused KV store + TQ3 quantize for batch prefill (Phase 3)
    FKVStoreBatchTQ3Shader: VkShaderModule;
    FKVStoreBatchTQ3Bundle: TVdxComputePipelineBundle;
    FKVStoreBatchTQ3DescLayout: VkDescriptorSetLayout;  // 3 bindings: src, decode, TQ3
    FKVStoreBatchTQ3DescPool: VkDescriptorPool;
    FKVStoreBatchTQ3DescSet: VkDescriptorSet;

    // Fused attention scores on TQ3 keys (Phase 4 — eliminates K dequant)
    FAttnScoresMHTQ3Shader: VkShaderModule;
    FAttnScoresMHTQ3Bundle: TVdxComputePipelineBundle;

    // TQ3 compressed KV caches (per layer, replaces F32 caches)
    FKCacheTQ3: array of TVdxGpuBuffer;
    FVCacheTQ3: array of TVdxGpuBuffer;

    // Shared F32 decode buffers (one pair, reused across layers)
    FKDecodeF32: TVdxGpuBuffer;
    FVDecodeF32: TVdxGpuBuffer;

    // Scratch buffers (reused every Forward call)
    FQBuf: TVdxGpuBuffer;        // [NumQHeads * HeadDim] F32
    FKBuf: TVdxGpuBuffer;        // [NumKVHeads * HeadDim] F32
    FVBuf: TVdxGpuBuffer;        // [NumKVHeads * HeadDim] F32
    FScoresBuf: TVdxGpuBuffer;   // [MaxSeqLen] F32
    FAttnOutBuf: TVdxGpuBuffer;  // [NumQHeads * HeadDim] F32
    // Model dimensions
    FHiddenDim: UInt32;
    FNumQHeads: UInt32;
    FNumKVHeads: UInt32;
    FHeadDim: UInt32;
    FMaxSeqLen: UInt32;
    FNumLayers: UInt32;

    // Internal helpers
    function LoadShader(const AFileName: string): VkShaderModule;
    procedure DispatchMatVec(const AWeightBuf: TVdxGpuBuffer;
      const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
      const AInDim: UInt32; const AOutDim: UInt32;
      const ATensorType: TVdxGGMLType);
    procedure DispatchBatchMatMul(const AWeightBuf: TVdxGpuBuffer;
      const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
      const AInDim: UInt32; const AOutDim: UInt32;
      const ANumTokens: UInt32; const ATensorType: TVdxGGMLType);

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Initialize all shaders, pipelines, scratch buffers, and KV cache
    procedure Init(const ACompute: TVdxVulkanCompute;
      const AHiddenDim: UInt32; const ANumQHeads: UInt32;
      const ANumKVHeads: UInt32; const AHeadDim: UInt32;
      const ANumLayers: UInt32; const AMaxSeqLen: UInt32;
      const AFFNWidth: UInt32);

    // Release all GPU resources
    procedure Cleanup();

    // Upload Q/K/V/O weight tensors from GGUF to GPU for one layer
    procedure UploadAttnWeights(const AReader: TVdxGGUFReader;
      const ALayerIndex: Integer; out AWeights: TVdxAttnLayerWeights);
    // Free GPU buffers for one layer's attention weights
    procedure FreeAttnWeights(var AWeights: TVdxAttnLayerWeights);

    // Run full attention for one layer at one position
    // AInputBuf: pre-normed residual [HiddenDim] F32
    // AQNormBuf/AKNormBuf: QK-norm weights [HeadDim] F32
    // AOutputBuf: attention output [HiddenDim] F32 (caller adds to residual)
    procedure Forward(const AInputBuf: TVdxGpuBuffer;
      const AWeights: TVdxAttnLayerWeights;
      const AQNormBuf: TVdxGpuBuffer;
      const AKNormBuf: TVdxGpuBuffer;
      const ALayerIndex: Integer;
      const APosition: Integer;
      const AThetaBase: Single;
      const AOutputBuf: TVdxGpuBuffer);

    // Expose for testing: run a single matvec F16 dispatch
    procedure TestMatVec(const AWeightBuf: TVdxGpuBuffer;
      const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
      const AInDim: UInt32; const AOutDim: UInt32;
      const ATensorType: TVdxGGMLType = gtF16);

    // Batch matmul: W[OutDim x InDim] x Input[NumTokens x InDim] -> Out[NumTokens x OutDim]
    // Used during batched prefill (Phase 6D). Must be called inside an active batch.
    procedure BatchMatMul(const AWeightBuf: TVdxGpuBuffer;
      const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
      const AInDim: UInt32; const AOutDim: UInt32;
      const ANumTokens: UInt32;
      const ATensorType: TVdxGGMLType = gtF16);

    // Batched attention for prefill: processes all N tokens through one layer.
    // QMat/KMat/VMat: pre-normed projections [NumTokens x Dim], already allocated.
    // AAttnOutMat: output [NumTokens x HiddenDim], caller adds to residual.
    // AStartPos: absolute KV slot at which to begin writing this batch.
    //   For a fresh prefill, pass 0. For continuation after a loaded KV
    //   cache, pass the saved position.
    // Must be called inside an active batch (BeginBatch/EndBatch).
    procedure ForwardBatch(const AInputMat: TVdxGpuBuffer;
      const AWeights: TVdxAttnLayerWeights;
      const AQNormBuf: TVdxGpuBuffer;
      const AKNormBuf: TVdxGpuBuffer;
      const ALayerIndex: Integer;
      const ANumTokens: UInt32;
      const AStartPos: UInt32;
      const AThetaBase: Single;
      const AQMat: TVdxGpuBuffer;
      const AKMat: TVdxGpuBuffer;
      const AVMat: TVdxGpuBuffer;
      const AAttnOutMat: TVdxGpuBuffer);

    // Diagnostic read-only access to internal buffers (for debugging tests)
    property ScoresBuf: TVdxGpuBuffer read FScoresBuf;
    property QBuf: TVdxGpuBuffer read FQBuf;
    property KBuf: TVdxGpuBuffer read FKBuf;
    property VBuf: TVdxGpuBuffer read FVBuf;
    property AttnOutBufInternal: TVdxGpuBuffer read FAttnOutBuf;
    function GetKCache(const ALayerIndex: Integer): TVdxGpuBuffer;
    function GetVCache(const ALayerIndex: Integer): TVdxGpuBuffer;

    // TQ3-compressed KV cache accessors (for save/load and diagnostics).
    // Returns the per-layer persistent cache buffer, NOT the shared decode
    // buffer that GetKCache/GetVCache return.
    function GetLayerKCacheTQ3(const ALayerIndex: Integer): TVdxGpuBuffer;
    function GetLayerVCacheTQ3(const ALayerIndex: Integer): TVdxGpuBuffer;

    // Byte size of one K or V layer's TQ3 cache buffer. Identical for K
    // and V (same shape), identical across all layers.
    function GetLayerKVCacheTQ3Bytes(): UInt64;
  end;

implementation

uses
  System.IOUtils,
  VindexLLM.Shaders;

{ TVdxAttention }
constructor TVdxAttention.Create();
begin
  inherited;

  FCompute := nil;
  FMatVecShader := VK_NULL_HANDLE;
  FMatVecQ8Shader := VK_NULL_HANDLE;
  FMatVecQ4Shader := VK_NULL_HANDLE;
  FQKNormShader := VK_NULL_HANDLE;
  FRoPEShader := VK_NULL_HANDLE;
  FAttnScoresMHShader := VK_NULL_HANDLE;
  FSoftmaxMHShader := VK_NULL_HANDLE;
  FAttnValueMHShader := VK_NULL_HANDLE;

  FMatVecBundle.Pipeline := VK_NULL_HANDLE;
  FMatVecQ8Bundle.Pipeline := VK_NULL_HANDLE;
  FMatVecQ4Bundle.Pipeline := VK_NULL_HANDLE;
  FQKNormBundle.Pipeline := VK_NULL_HANDLE;
  FRoPEBundle.Pipeline := VK_NULL_HANDLE;
  FAttnScoresMHBundle.Pipeline := VK_NULL_HANDLE;
  FSoftmaxMHBundle.Pipeline := VK_NULL_HANDLE;
  FAttnValueMHBundle.Pipeline := VK_NULL_HANDLE;

  FMatVecDescLayout := VK_NULL_HANDLE;
  FQKNormDescLayout := VK_NULL_HANDLE;
  FRoPEDescLayout := VK_NULL_HANDLE;
  FAttnScoresDescLayout := VK_NULL_HANDLE;
  FSoftmaxDescLayout := VK_NULL_HANDLE;
  FAttnValueDescLayout := VK_NULL_HANDLE;

  FDescPool := VK_NULL_HANDLE;

  FKVStoreShader := VK_NULL_HANDLE;
  FKVStoreBundle.Pipeline := VK_NULL_HANDLE;
  FKVStoreBundle.PipelineLayout := VK_NULL_HANDLE;
  FKVStoreDescLayout := VK_NULL_HANDLE;
  FKVStoreDescPool := VK_NULL_HANDLE;
  FKVStoreDescSet := VK_NULL_HANDLE;

  FMatMulF16Shader := VK_NULL_HANDLE;
  FMatMulQ8Shader := VK_NULL_HANDLE;
  FMatMulQ4Shader := VK_NULL_HANDLE;
  FMatMulF16Bundle.Pipeline := VK_NULL_HANDLE;
  FMatMulF16Bundle.PipelineLayout := VK_NULL_HANDLE;
  FMatMulQ8Bundle.Pipeline := VK_NULL_HANDLE;
  FMatMulQ8Bundle.PipelineLayout := VK_NULL_HANDLE;
  FMatMulQ4Bundle.Pipeline := VK_NULL_HANDLE;
  FMatMulQ4Bundle.PipelineLayout := VK_NULL_HANDLE;

  FRoPEBatchShader := VK_NULL_HANDLE;
  FKVStoreBatchShader := VK_NULL_HANDLE;
  FScoresPrefillShader := VK_NULL_HANDLE;
  FSoftmaxPrefillShader := VK_NULL_HANDLE;
  FValuePrefillShader := VK_NULL_HANDLE;
  FRoPEBatchBundle.Pipeline := VK_NULL_HANDLE;
  FKVStoreBatchBundle.Pipeline := VK_NULL_HANDLE;
  FScoresPrefillBundle.Pipeline := VK_NULL_HANDLE;
  FSoftmaxPrefillBundle.Pipeline := VK_NULL_HANDLE;
  FValuePrefillBundle.Pipeline := VK_NULL_HANDLE;
  FPrefillDescPool := VK_NULL_HANDLE;

  FTQ3KVQuantShader := VK_NULL_HANDLE;
  FTQ3KVDequantShader := VK_NULL_HANDLE;
  FTQ3KVQuantBundle.Pipeline := VK_NULL_HANDLE;
  FTQ3KVQuantBundle.PipelineLayout := VK_NULL_HANDLE;
  FTQ3KVDequantBundle.Pipeline := VK_NULL_HANDLE;
  FTQ3KVDequantBundle.PipelineLayout := VK_NULL_HANDLE;
  FTQ3KVDescLayout := VK_NULL_HANDLE;
  FTQ3KVDescPool := VK_NULL_HANDLE;

  FKVStoreBatchTQ3Shader := VK_NULL_HANDLE;
  FKVStoreBatchTQ3Bundle.Pipeline := VK_NULL_HANDLE;
  FKVStoreBatchTQ3Bundle.PipelineLayout := VK_NULL_HANDLE;
  FKVStoreBatchTQ3DescLayout := VK_NULL_HANDLE;
  FKVStoreBatchTQ3DescPool := VK_NULL_HANDLE;
  FKVStoreBatchTQ3DescSet := VK_NULL_HANDLE;

  FAttnScoresMHTQ3Shader := VK_NULL_HANDLE;
  FAttnScoresMHTQ3Bundle.Pipeline := VK_NULL_HANDLE;
  FAttnScoresMHTQ3Bundle.PipelineLayout := VK_NULL_HANDLE;
end;

destructor TVdxAttention.Destroy();
begin
  if FCompute <> nil then
    Cleanup();

  inherited;
end;

function TVdxAttention.LoadShader(const AFileName: string): VkShaderModule;
var
  LSpvData: TBytes;
begin
  LSpvData := VdxLoadShader(AFileName);
  Result := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
end;

procedure TVdxAttention.Init(const ACompute: TVdxVulkanCompute;
  const AHiddenDim: UInt32; const ANumQHeads: UInt32;
  const ANumKVHeads: UInt32; const AHeadDim: UInt32;
  const ANumLayers: UInt32; const AMaxSeqLen: UInt32;
  const AFFNWidth: UInt32);
var
  LI: Integer;
  LCacheSize: UInt64;
  LDummyBuf: TVdxGpuBuffer;
  LMaxQ8Blocks: UInt32;
begin
  FCompute := ACompute;
  FHiddenDim := AHiddenDim;
  FNumQHeads := ANumQHeads;
  FNumKVHeads := ANumKVHeads;
  FHeadDim := AHeadDim;
  FNumLayers := ANumLayers;
  FMaxSeqLen := AMaxSeqLen;

  // Max Q8_0 blocks per row across all matvec uses (attention + FFN).
  // FFN-down has the largest in_dim (AFFNWidth), so it determines the ceiling.
  LMaxQ8Blocks := AFFNWidth div 32;

  // Load all shaders
  FMatVecShader := LoadShader('MATVEC_F16');
  FMatVecQ8Shader := LoadShader('MATVEC_Q8_0');
  FMatVecQ4Shader := LoadShader('MATVEC_Q4_0');
  FQKNormShader := LoadShader('QK_NORM');
  FRoPEShader := LoadShader('ROPE');
  FAttnScoresMHShader := LoadShader('ATTN_SCORES_MH');
  FSoftmaxMHShader := LoadShader('SOFTMAX_MH');
  FAttnValueMHShader := LoadShader('ATTN_VALUE_MH');
  // Create descriptor set layouts
  FMatVecDescLayout := FCompute.CreateStorageDescriptorSetLayout(3);
  FQKNormDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  FRoPEDescLayout := FCompute.CreateStorageDescriptorSetLayout(1);
  FAttnScoresDescLayout := FCompute.CreateStorageDescriptorSetLayout(3);
  FSoftmaxDescLayout := FCompute.CreateStorageDescriptorSetLayout(1);
  FAttnValueDescLayout := FCompute.CreateStorageDescriptorSetLayout(3);

  // Create pipelines with push constants
  FMatVecBundle := FCompute.CreateComputePipelineWithPush(
    FMatVecShader, 'main', FMatVecDescLayout, SizeOf(TVdxMatVecF16Push));
  FMatVecQ8Bundle := FCompute.CreateComputePipelineWithPushAndSpec(
    FMatVecQ8Shader, 'main', FMatVecDescLayout, SizeOf(TVdxMatVecF16Push),
    LMaxQ8Blocks);
  FMatVecQ4Bundle := FCompute.CreateComputePipelineWithPush(
    FMatVecQ4Shader, 'main', FMatVecDescLayout, SizeOf(TVdxMatVecF16Push));
  FQKNormBundle := FCompute.CreateComputePipelineWithPush(
    FQKNormShader, 'main', FQKNormDescLayout, SizeOf(TVdxQKNormPush));
  FRoPEBundle := FCompute.CreateComputePipelineWithPush(
    FRoPEShader, 'main', FRoPEDescLayout, SizeOf(TVdxRoPEPush));

  // Multi-head attention pipelines (reuse existing descriptor set layouts)
  FAttnScoresMHBundle := FCompute.CreateComputePipelineWithPush(
    FAttnScoresMHShader, 'main', FAttnScoresDescLayout,
    SizeOf(TVdxAttnScoresMHPush));
  FSoftmaxMHBundle := FCompute.CreateComputePipelineWithPush(
    FSoftmaxMHShader, 'main', FSoftmaxDescLayout,
    SizeOf(TVdxSoftmaxMHPush));
  FAttnValueMHBundle := FCompute.CreateComputePipelineWithPush(
    FAttnValueMHShader, 'main', FAttnValueDescLayout,
    SizeOf(TVdxAttnValueMHPush));

  // Pre-allocate one descriptor pool + 6 reusable sets (no per-dispatch churn)
  // Total descriptors: 3+2+1+3+1+3 = 13 storage buffers across 6 sets
  FDescPool := FCompute.CreateDescriptorPoolForStorage(6, 13);
  FMatVecDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FDescPool, FMatVecDescLayout, [FQBuf, FQBuf, FQBuf]);
  FQKNormDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FDescPool, FQKNormDescLayout, [FQBuf, FQBuf]);
  FRoPEDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FDescPool, FRoPEDescLayout, [FQBuf]);
  FScoresDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FDescPool, FAttnScoresDescLayout, [FQBuf, FQBuf, FQBuf]);
  FSoftmaxDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FDescPool, FSoftmaxDescLayout, [FQBuf]);
  FValueDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FDescPool, FAttnValueDescLayout, [FQBuf, FQBuf, FQBuf]);

  // Allocate scratch buffers (device-local, storage + transfer)
  FQBuf := FCompute.CreateGpuBuffer(
    UInt64(FNumQHeads) * FHeadDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FKBuf := FCompute.CreateGpuBuffer(
    UInt64(FNumKVHeads) * FHeadDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FVBuf := FCompute.CreateGpuBuffer(
    UInt64(FNumKVHeads) * FHeadDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FScoresBuf := FCompute.CreateGpuBuffer(
    UInt64(FNumQHeads) * FMaxSeqLen * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FAttnOutBuf := FCompute.CreateGpuBuffer(
    UInt64(FNumQHeads) * FHeadDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  // Allocate TQ3 compressed KV caches per layer
  // TQ3 layout: [NumKVHeads * MaxSeqLen * BlocksPerHead * 4] uint32 per layer
  // BlocksPerHead = HeadDim / 32 = 8, each block = 4 uint32 = 16 bytes
  LCacheSize := UInt64(FNumKVHeads) * FMaxSeqLen * (FHeadDim div 32) * 16;
  SetLength(FKCacheTQ3, FNumLayers);
  SetLength(FVCacheTQ3, FNumLayers);

  for LI := 0 to FNumLayers - 1 do
  begin
    FKCacheTQ3[LI] := FCompute.CreateGpuBuffer(
      LCacheSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT
        or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    FVCacheTQ3[LI] := FCompute.CreateGpuBuffer(
      LCacheSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT
        or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  end;

  // Shared F32 decode buffers (one pair reused across all layers)
  // Same layout as old per-layer cache: [NumKVHeads * MaxSeqLen * HeadDim] F32
  LCacheSize := UInt64(FNumKVHeads) * FMaxSeqLen * FHeadDim * SizeOf(Single);
  FKDecodeF32 := FCompute.CreateGpuBuffer(
    LCacheSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT
      or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  FVDecodeF32 := FCompute.CreateGpuBuffer(
    LCacheSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT
      or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // KV cache store shader — replaces per-head CopyBufferRegion with single dispatch
  FKVStoreShader := LoadShader('KV_CACHE_STORE');
  FKVStoreDescLayout := FCompute.CreateStorageDescriptorSetLayout(4);
  FKVStoreBundle := FCompute.CreateComputePipelineWithPush(
    FKVStoreShader, 'main', FKVStoreDescLayout, SizeOf(TVdxKVCacheStorePush));

  // Pre-allocate descriptor pool + set (4 bindings: KBuf, VBuf, KCache, VCache)
  LDummyBuf := Default(TVdxGpuBuffer);
  FKVStoreDescPool := FCompute.CreateDescriptorPoolForStorage(1, 4);
  FKVStoreDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FKVStoreDescPool, FKVStoreDescLayout,
    [LDummyBuf, LDummyBuf, LDummyBuf, LDummyBuf]);

  // Batch matmul shaders + pipelines (Phase 6D — prefill batching)
  // Reuse FMatVecDescLayout (3 storage bindings: weight, input, output)
  FMatMulF16Shader := LoadShader('MATMUL_F16');
  FMatMulQ8Shader := LoadShader('MATMUL_Q8_0');
  FMatMulQ4Shader := LoadShader('MATMUL_Q4_0');
  FMatMulF16Bundle := FCompute.CreateComputePipelineWithPush(
    FMatMulF16Shader, 'main', FMatVecDescLayout, SizeOf(TVdxMatMulPush));
  FMatMulQ8Bundle := FCompute.CreateComputePipelineWithPushAndSpec(
    FMatMulQ8Shader, 'main', FMatVecDescLayout, SizeOf(TVdxMatMulPush),
    LMaxQ8Blocks);
  FMatMulQ4Bundle := FCompute.CreateComputePipelineWithPush(
    FMatMulQ4Shader, 'main', FMatVecDescLayout, SizeOf(TVdxMatMulPush));

  // --- Prefill attention shaders + pipelines (Phase 6D) ---
  FRoPEBatchShader := LoadShader('ROPE_BATCH');
  FKVStoreBatchShader := LoadShader('KV_CACHE_STORE_BATCH');
  FScoresPrefillShader := LoadShader('ATTN_SCORES_PREFILL');
  FSoftmaxPrefillShader := LoadShader('SOFTMAX_PREFILL');
  FValuePrefillShader := LoadShader('ATTN_VALUE_PREFILL');

  // Reuse existing descriptor set layouts (same binding counts)
  FRoPEBatchBundle := FCompute.CreateComputePipelineWithPush(
    FRoPEBatchShader, 'main', FRoPEDescLayout,
    SizeOf(TVdxRoPEBatchPush));
  FKVStoreBatchBundle := FCompute.CreateComputePipelineWithPush(
    FKVStoreBatchShader, 'main', FKVStoreDescLayout,
    SizeOf(TVdxKVCacheStoreBatchPush));
  FScoresPrefillBundle := FCompute.CreateComputePipelineWithPush(
    FScoresPrefillShader, 'main', FAttnScoresDescLayout,
    SizeOf(TVdxAttnScoresPrefillPush));
  FSoftmaxPrefillBundle := FCompute.CreateComputePipelineWithPush(
    FSoftmaxPrefillShader, 'main', FSoftmaxDescLayout,
    SizeOf(TVdxSoftmaxPrefillPush));
  FValuePrefillBundle := FCompute.CreateComputePipelineWithPush(
    FValuePrefillShader, 'main', FAttnValueDescLayout,
    SizeOf(TVdxAttnValuePrefillPush));

  // Prefill descriptor pool: 5 sets, total bindings = 1+4+3+1+3 = 12
  FPrefillDescPool := FCompute.CreateDescriptorPoolForStorage(5, 12);
  FPrefillRoPEDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FPrefillDescPool, FRoPEDescLayout, [FQBuf]);
  FPrefillKVStoreDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FPrefillDescPool, FKVStoreDescLayout,
    [LDummyBuf, LDummyBuf, LDummyBuf, LDummyBuf]);
  FPrefillScoresDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FPrefillDescPool, FAttnScoresDescLayout,
    [LDummyBuf, LDummyBuf, LDummyBuf]);
  FPrefillSoftmaxDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FPrefillDescPool, FSoftmaxDescLayout, [LDummyBuf]);
  FPrefillValueDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FPrefillDescPool, FAttnValueDescLayout,
    [LDummyBuf, LDummyBuf, LDummyBuf]);

  // Pre-allocated prefill scores buffer [NumQHeads x MaxSeq x MaxSeq] F32
  FPrefillScoresBuf := FCompute.CreateGpuBuffer(
    UInt64(FNumQHeads) * FMaxSeqLen * FMaxSeqLen * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // --- TQ3 KV cache compression pipelines ---
  FTQ3KVQuantShader := LoadShader('TQ3_KV_QUANTIZE');
  FTQ3KVDequantShader := LoadShader('TQ3_KV_DEQUANTIZE');
  FTQ3KVDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  FTQ3KVQuantBundle := FCompute.CreateComputePipelineWithPush(
    FTQ3KVQuantShader, 'main', FTQ3KVDescLayout, SizeOf(TVdxTQ3KVQuantPush));
  FTQ3KVDequantBundle := FCompute.CreateComputePipelineWithPush(
    FTQ3KVDequantShader, 'main', FTQ3KVDescLayout, SizeOf(TVdxTQ3KVDequantPush));

  // Fused KV store + TQ3 quantize pipeline (Phase 3 — 3 bindings: src, decode, TQ3)
  FKVStoreBatchTQ3Shader := LoadShader('KV_CACHE_STORE_BATCH_TQ3');
  FKVStoreBatchTQ3DescLayout := FCompute.CreateStorageDescriptorSetLayout(3);
  FKVStoreBatchTQ3Bundle := FCompute.CreateComputePipelineWithPush(
    FKVStoreBatchTQ3Shader, 'main', FKVStoreBatchTQ3DescLayout,
    SizeOf(TVdxKVStoreBatchTQ3Push));
  FKVStoreBatchTQ3DescPool := FCompute.CreateDescriptorPoolForStorage(1, 3);
  FKVStoreBatchTQ3DescSet := FCompute.AllocateDescriptorSetForBuffers(
    FKVStoreBatchTQ3DescPool, FKVStoreBatchTQ3DescLayout,
    [LDummyBuf, LDummyBuf, LDummyBuf]);

  // TQ3 KV descriptor pool: 2 sets (quant + dequant), 2 bindings each = 4 total
  FTQ3KVDescPool := FCompute.CreateDescriptorPoolForStorage(2, 4);
  FTQ3KVQuantDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FTQ3KVDescPool, FTQ3KVDescLayout, [LDummyBuf, LDummyBuf]);
  FTQ3KVDequantDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FTQ3KVDescPool, FTQ3KVDescLayout, [LDummyBuf, LDummyBuf]);

  // --- Fused TQ3 attention scores pipeline (Phase 4) ---
  // Reuses FAttnScoresDescLayout (3 bindings) and FScoresDescSet
  FAttnScoresMHTQ3Shader := LoadShader('ATTN_SCORES_MH_TQ3');
  FAttnScoresMHTQ3Bundle := FCompute.CreateComputePipelineWithPush(
    FAttnScoresMHTQ3Shader, 'main', FAttnScoresDescLayout,
    SizeOf(TVdxAttnScoresMHTQ3Push));
end;

procedure TVdxAttention.Cleanup();
var
  LI: Integer;
begin
  if FCompute = nil then
    Exit;

  // Free TQ3 KV caches
  for LI := 0 to Length(FKCacheTQ3) - 1 do
  begin
    if FKCacheTQ3[LI].Buffer <> VK_NULL_HANDLE then
      FCompute.DestroyGpuBuffer(FKCacheTQ3[LI]);
    if FVCacheTQ3[LI].Buffer <> VK_NULL_HANDLE then
      FCompute.DestroyGpuBuffer(FVCacheTQ3[LI]);
  end;
  SetLength(FKCacheTQ3, 0);
  SetLength(FVCacheTQ3, 0);

  // Free shared decode buffers
  if FKDecodeF32.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(FKDecodeF32);
  if FVDecodeF32.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(FVDecodeF32);

  // Free scratch buffers
  if FQBuf.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(FQBuf);
  if FKBuf.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(FKBuf);
  if FVBuf.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(FVBuf);
  if FScoresBuf.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(FScoresBuf);
  if FAttnOutBuf.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(FAttnOutBuf);
  // Destroy pipelines
  FCompute.DestroyComputePipelineBundle(FMatVecBundle);
  FCompute.DestroyComputePipelineBundle(FMatVecQ8Bundle);
  FCompute.DestroyComputePipelineBundle(FMatVecQ4Bundle);
  FCompute.DestroyComputePipelineBundle(FQKNormBundle);
  FCompute.DestroyComputePipelineBundle(FRoPEBundle);
  FCompute.DestroyComputePipelineBundle(FAttnScoresMHBundle);
  FCompute.DestroyComputePipelineBundle(FSoftmaxMHBundle);
  FCompute.DestroyComputePipelineBundle(FAttnValueMHBundle);

  // Destroy KV cache store resources
  FCompute.DestroyComputePipelineBundle(FKVStoreBundle);
  if FKVStoreDescPool <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorPoolHandle(FKVStoreDescPool);
  FCompute.DestroyDescriptorSetLayoutHandle(FKVStoreDescLayout);
  FCompute.DestroyShaderModuleHandle(FKVStoreShader);

  // Destroy batch matmul pipelines + shaders (Phase 6D)
  FCompute.DestroyComputePipelineBundle(FMatMulF16Bundle);
  FCompute.DestroyComputePipelineBundle(FMatMulQ8Bundle);
  FCompute.DestroyComputePipelineBundle(FMatMulQ4Bundle);
  FCompute.DestroyShaderModuleHandle(FMatMulF16Shader);
  FCompute.DestroyShaderModuleHandle(FMatMulQ8Shader);
  FCompute.DestroyShaderModuleHandle(FMatMulQ4Shader);

  // Destroy prefill attention resources (Phase 6D)
  if FPrefillScoresBuf.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(FPrefillScoresBuf);
  if FPrefillDescPool <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorPoolHandle(FPrefillDescPool);
  FCompute.DestroyComputePipelineBundle(FRoPEBatchBundle);
  FCompute.DestroyComputePipelineBundle(FKVStoreBatchBundle);
  FCompute.DestroyComputePipelineBundle(FScoresPrefillBundle);
  FCompute.DestroyComputePipelineBundle(FSoftmaxPrefillBundle);
  FCompute.DestroyComputePipelineBundle(FValuePrefillBundle);
  FCompute.DestroyShaderModuleHandle(FRoPEBatchShader);
  FCompute.DestroyShaderModuleHandle(FKVStoreBatchShader);
  FCompute.DestroyShaderModuleHandle(FScoresPrefillShader);
  FCompute.DestroyShaderModuleHandle(FSoftmaxPrefillShader);
  FCompute.DestroyShaderModuleHandle(FValuePrefillShader);

  // Destroy pre-allocated descriptor pool (frees all 6 sets automatically)
  if FDescPool <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorPoolHandle(FDescPool);

  // Destroy descriptor set layouts
  FCompute.DestroyDescriptorSetLayoutHandle(FMatVecDescLayout);
  FCompute.DestroyDescriptorSetLayoutHandle(FQKNormDescLayout);
  FCompute.DestroyDescriptorSetLayoutHandle(FRoPEDescLayout);
  FCompute.DestroyDescriptorSetLayoutHandle(FAttnScoresDescLayout);
  FCompute.DestroyDescriptorSetLayoutHandle(FSoftmaxDescLayout);
  FCompute.DestroyDescriptorSetLayoutHandle(FAttnValueDescLayout);

  // Destroy shader modules
  FCompute.DestroyShaderModuleHandle(FMatVecShader);
  FCompute.DestroyShaderModuleHandle(FMatVecQ8Shader);
  FCompute.DestroyShaderModuleHandle(FMatVecQ4Shader);
  FCompute.DestroyShaderModuleHandle(FQKNormShader);
  FCompute.DestroyShaderModuleHandle(FRoPEShader);
  FCompute.DestroyShaderModuleHandle(FAttnScoresMHShader);
  FCompute.DestroyShaderModuleHandle(FSoftmaxMHShader);
  FCompute.DestroyShaderModuleHandle(FAttnValueMHShader);

  // Destroy TQ3 KV cache compression resources
  if FTQ3KVDescPool <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorPoolHandle(FTQ3KVDescPool);
  FCompute.DestroyComputePipelineBundle(FTQ3KVQuantBundle);
  FCompute.DestroyComputePipelineBundle(FTQ3KVDequantBundle);
  if FTQ3KVDescLayout <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorSetLayoutHandle(FTQ3KVDescLayout);
  if FTQ3KVQuantShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FTQ3KVQuantShader);
  if FTQ3KVDequantShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FTQ3KVDequantShader);

  // Destroy fused KV store + TQ3 quantize resources (Phase 3)
  if FKVStoreBatchTQ3DescPool <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorPoolHandle(FKVStoreBatchTQ3DescPool);
  FCompute.DestroyComputePipelineBundle(FKVStoreBatchTQ3Bundle);
  if FKVStoreBatchTQ3DescLayout <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorSetLayoutHandle(FKVStoreBatchTQ3DescLayout);
  if FKVStoreBatchTQ3Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FKVStoreBatchTQ3Shader);

  // Destroy fused TQ3 attention scores resources (Phase 4)
  FCompute.DestroyComputePipelineBundle(FAttnScoresMHTQ3Bundle);
  if FAttnScoresMHTQ3Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FAttnScoresMHTQ3Shader);

  FCompute := nil;
end;

procedure TVdxAttention.DispatchMatVec(const AWeightBuf: TVdxGpuBuffer;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ATensorType: TVdxGGMLType);
var
  LPush: TVdxMatVecF16Push;
  LGroups: UInt32;
  LPipeline: VkPipeline;
  LPipelineLayout: VkPipelineLayout;
begin
  // Rebind buffers to pre-allocated descriptor set (no pool create/destroy)
  FCompute.UpdateDescriptorSetBuffers(FMatVecDescSet,
    [AWeightBuf, AInputBuf, AOutputBuf]);

  // Select pipeline: Q4_0 and Q8_0 use full in_dim, F16 uses in_dim/2
  if ATensorType = gtQ4_0 then
  begin
    LPush.InDimHalf := AInDim;
    LPipeline := FMatVecQ4Bundle.Pipeline;
    LPipelineLayout := FMatVecQ4Bundle.PipelineLayout;
  end
  else if ATensorType = gtQ8_0 then
  begin
    LPush.InDimHalf := AInDim;
    LPipeline := FMatVecQ8Bundle.Pipeline;
    LPipelineLayout := FMatVecQ8Bundle.PipelineLayout;
  end
  else
  begin
    LPush.InDimHalf := AInDim div 2;
    LPipeline := FMatVecBundle.Pipeline;
    LPipelineLayout := FMatVecBundle.PipelineLayout;
  end;

  LPush.OutDim := AOutDim;

  // Tiled: one workgroup (256 threads) per output row
  LGroups := AOutDim;

  FCompute.DispatchComputeWithPush(
    LPipeline,
    LPipelineLayout,
    FMatVecDescSet,
    @LPush,
    SizeOf(LPush),
    LGroups);
end;

procedure TVdxAttention.TestMatVec(const AWeightBuf: TVdxGpuBuffer;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ATensorType: TVdxGGMLType);
begin
  DispatchMatVec(AWeightBuf, AInputBuf, AOutputBuf, AInDim, AOutDim, ATensorType);
end;

procedure TVdxAttention.DispatchBatchMatMul(const AWeightBuf: TVdxGpuBuffer;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ANumTokens: UInt32; const ATensorType: TVdxGGMLType);
var
  LPush: TVdxMatMulPush;
  LPipeline: VkPipeline;
  LPipelineLayout: VkPipelineLayout;
begin
  // Rebind buffers to pre-allocated descriptor set (same 3-binding layout as matvec)
  FCompute.UpdateDescriptorSetBuffers(FMatVecDescSet,
    [AWeightBuf, AInputBuf, AOutputBuf]);

  // Select pipeline and set dimension parameter
  if ATensorType = gtQ4_0 then
  begin
    LPush.InDimParam := AInDim;
    LPipeline := FMatMulQ4Bundle.Pipeline;
    LPipelineLayout := FMatMulQ4Bundle.PipelineLayout;
  end
  else if ATensorType = gtQ8_0 then
  begin
    LPush.InDimParam := AInDim;
    LPipeline := FMatMulQ8Bundle.Pipeline;
    LPipelineLayout := FMatMulQ8Bundle.PipelineLayout;
  end
  else
  begin
    LPush.InDimParam := AInDim div 2;
    LPipeline := FMatMulF16Bundle.Pipeline;
    LPipelineLayout := FMatMulF16Bundle.PipelineLayout;
  end;

  LPush.OutDim := AOutDim;
  LPush.NumTokens := ANumTokens;

  // 2D dispatch: X = output rows (OutDim), Y = tokens (NumTokens)
  FCompute.DispatchComputeWithPush(
    LPipeline, LPipelineLayout,
    FMatVecDescSet, @LPush, SizeOf(LPush),
    AOutDim, ANumTokens);
end;

procedure TVdxAttention.BatchMatMul(const AWeightBuf: TVdxGpuBuffer;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ANumTokens: UInt32; const ATensorType: TVdxGGMLType);
begin
  DispatchBatchMatMul(AWeightBuf, AInputBuf, AOutputBuf,
    AInDim, AOutDim, ANumTokens, ATensorType);
end;

procedure TVdxAttention.ForwardBatch(const AInputMat: TVdxGpuBuffer;
  const AWeights: TVdxAttnLayerWeights;
  const AQNormBuf: TVdxGpuBuffer;
  const AKNormBuf: TVdxGpuBuffer;
  const ALayerIndex: Integer;
  const ANumTokens: UInt32;
  const AStartPos: UInt32;
  const AThetaBase: Single;
  const AQMat: TVdxGpuBuffer;
  const AKMat: TVdxGpuBuffer;
  const AVMat: TVdxGpuBuffer;
  const AAttnOutMat: TVdxGpuBuffer);
var
  LSeqLen: UInt32;
  LQKNormPush: TVdxQKNormPush;
  LRoPEPush: TVdxRoPEBatchPush;
  LKVStoreTQ3Push: TVdxKVStoreBatchTQ3Push;
  LTQ3DequantPush: TVdxTQ3KVDequantPush;
  LScoresPush: TVdxAttnScoresPrefillPush;
  LSoftmaxPush: TVdxSoftmaxPrefillPush;
  LValuePush: TVdxAttnValuePrefillPush;
begin
  // Total keys in the filled cache after this batch writes its tokens.
  // Prefill dispatches and shader indexing must cover this range, not
  // just the current batch's own tokens.
  LSeqLen := AStartPos + ANumTokens;

  // ---- Step 1: Q/K/V projections (batch matmul) ----
  DispatchBatchMatMul(AWeights.QWeightGpu, AInputMat, AQMat,
    FHiddenDim, FNumQHeads * FHeadDim, ANumTokens, AWeights.WeightType);
  DispatchBatchMatMul(AWeights.KWeightGpu, AInputMat, AKMat,
    FHiddenDim, FNumKVHeads * FHeadDim, ANumTokens, AWeights.WeightType);
  DispatchBatchMatMul(AWeights.VWeightGpu, AInputMat, AVMat,
    FHiddenDim, FNumKVHeads * FHeadDim, ANumTokens, AWeights.WeightType);
  FCompute.BatchBarrier(); // Q/K/V matrices ready

  // ---- Step 2: QK-norm (reuse existing shader with NumHeads * NumTokens) ----
  // QMat is [N x NumQHeads x HeadDim] flat = [N*NumQHeads x HeadDim]
  LQKNormPush.HeadDim := FHeadDim;
  LQKNormPush.Eps := 1e-6;

  LQKNormPush.NumHeads := FNumQHeads * ANumTokens;
  FCompute.UpdateDescriptorSetBuffers(FQKNormDescSet, [AQMat, AQNormBuf]);
  FCompute.DispatchComputeWithPush(
    FQKNormBundle.Pipeline, FQKNormBundle.PipelineLayout,
    FQKNormDescSet, @LQKNormPush, SizeOf(LQKNormPush),
    FNumQHeads * ANumTokens);

  LQKNormPush.NumHeads := FNumKVHeads * ANumTokens;
  FCompute.UpdateDescriptorSetBuffers(FQKNormDescSet, [AKMat, AKNormBuf]);
  FCompute.DispatchComputeWithPush(
    FQKNormBundle.Pipeline, FQKNormBundle.PipelineLayout,
    FQKNormDescSet, @LQKNormPush, SizeOf(LQKNormPush),
    FNumKVHeads * ANumTokens);
  FCompute.BatchBarrier(); // Q/K normed

  // ---- Step 3: Batched RoPE (per-token positions start_pos..start_pos+N-1) ----
  LRoPEPush.HeadDim := FHeadDim;
  LRoPEPush.ThetaBase := AThetaBase;
  LRoPEPush.StartPos := AStartPos;

  // RoPE on Q: dispatch 2D (NumQHeads, NumTokens)
  LRoPEPush.NumHeads := FNumQHeads;
  LRoPEPush.NumTokens := ANumTokens;
  FCompute.UpdateDescriptorSetBuffers(FPrefillRoPEDescSet, [AQMat]);
  FCompute.DispatchComputeWithPush(
    FRoPEBatchBundle.Pipeline, FRoPEBatchBundle.PipelineLayout,
    FPrefillRoPEDescSet, @LRoPEPush, SizeOf(LRoPEPush),
    FNumQHeads, ANumTokens);

  // RoPE on K: dispatch 2D (NumKVHeads, NumTokens)
  LRoPEPush.NumHeads := FNumKVHeads;
  FCompute.UpdateDescriptorSetBuffers(FPrefillRoPEDescSet, [AKMat]);
  FCompute.DispatchComputeWithPush(
    FRoPEBatchBundle.Pipeline, FRoPEBatchBundle.PipelineLayout,
    FPrefillRoPEDescSet, @LRoPEPush, SizeOf(LRoPEPush),
    FNumKVHeads, ANumTokens);
  FCompute.BatchBarrier(); // Q/K with RoPE applied

  // ---- Step 4: Fused store + TQ3 quantize (writes decode buffer AND TQ3 cache) ----
  LKVStoreTQ3Push.HeadDim := FHeadDim;
  LKVStoreTQ3Push.MaxSeq := FMaxSeqLen;
  LKVStoreTQ3Push.NumHeads := FNumKVHeads;
  LKVStoreTQ3Push.NumTokens := ANumTokens;
  LKVStoreTQ3Push.StartPos := AStartPos;

  // Fused K: projection → decode buffer + TQ3 cache
  FCompute.UpdateDescriptorSetBuffers(FKVStoreBatchTQ3DescSet,
    [AKMat, FKDecodeF32, FKCacheTQ3[ALayerIndex]]);
  FCompute.DispatchComputeWithPush(
    FKVStoreBatchTQ3Bundle.Pipeline, FKVStoreBatchTQ3Bundle.PipelineLayout,
    FKVStoreBatchTQ3DescSet, @LKVStoreTQ3Push, SizeOf(LKVStoreTQ3Push),
    (FHeadDim div 32) * FNumKVHeads, ANumTokens);

  // Fused V: projection → decode buffer + TQ3 cache
  FCompute.UpdateDescriptorSetBuffers(FKVStoreBatchTQ3DescSet,
    [AVMat, FVDecodeF32, FVCacheTQ3[ALayerIndex]]);
  FCompute.DispatchComputeWithPush(
    FKVStoreBatchTQ3Bundle.Pipeline, FKVStoreBatchTQ3Bundle.PipelineLayout,
    FKVStoreBatchTQ3DescSet, @LKVStoreTQ3Push, SizeOf(LKVStoreTQ3Push),
    (FHeadDim div 32) * FNumKVHeads, ANumTokens);
  FCompute.BatchBarrier(); // Decode buffers + TQ3 caches ready

  // ---- Step 4.5: Dequantize full cache range for continuation prefill ----
  //
  // When AStartPos > 0 the decode buffers only hold this batch's keys and
  // values at positions [AStartPos .. AStartPos + ANumTokens). Positions
  // [0 .. AStartPos) hold leftover junk from the previous call's last
  // layer. The prefill scores and value shaders read from FKDecodeF32 /
  // FVDecodeF32 over the full SeqLen range, so we must populate them from
  // the persistent per-layer TQ3 caches before attention runs.
  //
  // When AStartPos = 0 the decode buffers are already populated correctly
  // by the Step 4 store, straight from the live projections (no TQ3 round-
  // trip). Skipping the dequant here preserves bit-exact behavior for
  // single-call (fresh-session) prefill.
  if AStartPos > 0 then
  begin
    LTQ3DequantPush.BlocksPerHead := FHeadDim div 32;
    LTQ3DequantPush.MaxSeq := FMaxSeqLen;
    LTQ3DequantPush.SeqLen := LSeqLen;
    LTQ3DequantPush.NumHeads := FNumKVHeads;

    // K dequant: FKCacheTQ3[layer] → FKDecodeF32, full [0, SeqLen) range
    FCompute.UpdateDescriptorSetBuffers(FTQ3KVDequantDescSet,
      [FKCacheTQ3[ALayerIndex], FKDecodeF32]);
    FCompute.DispatchComputeWithPush(
      FTQ3KVDequantBundle.Pipeline, FTQ3KVDequantBundle.PipelineLayout,
      FTQ3KVDequantDescSet, @LTQ3DequantPush, SizeOf(LTQ3DequantPush),
      (FHeadDim div 32) * FNumKVHeads * LSeqLen);

    // V dequant: FVCacheTQ3[layer] → FVDecodeF32, full [0, SeqLen) range
    FCompute.UpdateDescriptorSetBuffers(FTQ3KVDequantDescSet,
      [FVCacheTQ3[ALayerIndex], FVDecodeF32]);
    FCompute.DispatchComputeWithPush(
      FTQ3KVDequantBundle.Pipeline, FTQ3KVDequantBundle.PipelineLayout,
      FTQ3KVDequantDescSet, @LTQ3DequantPush, SizeOf(LTQ3DequantPush),
      (FHeadDim div 32) * FNumKVHeads * LSeqLen);
    FCompute.BatchBarrier(); // Decode buffers hold full [0, SeqLen) range
  end;

  // ---- Step 5: Prefill attention (scores + softmax + value) ----

  // 5a. Causal attention scores: 3D dispatch (keys, heads, queries)
  //   X dimension must cover SeqLen keys, not NumTokens — every new query
  //   token attends over the full filled cache (start_pos + num_tokens).
  LScoresPush.HeadDim := FHeadDim;
  LScoresPush.NumTokens := ANumTokens;
  LScoresPush.MaxSeq := FMaxSeqLen;
  LScoresPush.Scale := 1.0 / Sqrt(Single(FHeadDim));
  LScoresPush.NumQHeads := FNumQHeads;
  LScoresPush.GqaRatio := FNumQHeads div FNumKVHeads;
  LScoresPush.StartPos := AStartPos;
  LScoresPush.SeqLen := LSeqLen;
  FCompute.UpdateDescriptorSetBuffers(FPrefillScoresDescSet,
    [AQMat, FKDecodeF32, FPrefillScoresBuf]);
  FCompute.DispatchComputeWithPush(
    FScoresPrefillBundle.Pipeline, FScoresPrefillBundle.PipelineLayout,
    FPrefillScoresDescSet, @LScoresPush, SizeOf(LScoresPush),
    (LSeqLen + 255) div 256, FNumQHeads, ANumTokens);
  FCompute.BatchBarrier(); // Scores ready for softmax

  // 5b. Softmax: 2D dispatch (heads, queries)
  LSoftmaxPush.NumTokens := ANumTokens;
  LSoftmaxPush.NumQHeads := FNumQHeads;
  LSoftmaxPush.MaxSeq := FMaxSeqLen;
  LSoftmaxPush.SeqLen := LSeqLen;
  LSoftmaxPush.StartPos := AStartPos;
  FCompute.UpdateDescriptorSetBuffers(FPrefillSoftmaxDescSet,
    [FPrefillScoresBuf]);
  FCompute.DispatchComputeWithPush(
    FSoftmaxPrefillBundle.Pipeline, FSoftmaxPrefillBundle.PipelineLayout,
    FPrefillSoftmaxDescSet, @LSoftmaxPush, SizeOf(LSoftmaxPush),
    FNumQHeads, ANumTokens);
  FCompute.BatchBarrier(); // Attention weights ready

  // 5c. Value weighted sum: 3D dispatch (dim, heads, queries)
  LValuePush.HeadDim := FHeadDim;
  LValuePush.NumTokens := ANumTokens;
  LValuePush.MaxSeq := FMaxSeqLen;
  LValuePush.NumQHeads := FNumQHeads;
  LValuePush.GqaRatio := FNumQHeads div FNumKVHeads;
  LValuePush.StartPos := AStartPos;
  LValuePush.SeqLen := LSeqLen;
  FCompute.UpdateDescriptorSetBuffers(FPrefillValueDescSet,
    [FPrefillScoresBuf, FVDecodeF32, AQMat]);
  FCompute.DispatchComputeWithPush(
    FValuePrefillBundle.Pipeline, FValuePrefillBundle.PipelineLayout,
    FPrefillValueDescSet, @LValuePush, SizeOf(LValuePush),
    (FHeadDim + 255) div 256, FNumQHeads, ANumTokens);
  FCompute.BatchBarrier(); // AQMat has value output [N x QDim]

  // ---- Step 6: Output projection (batch matmul) ----
  // O: W[QDim x HiddenDim] x AQMat[N x QDim] -> AAttnOutMat[N x HiddenDim]
  DispatchBatchMatMul(AWeights.OWeightGpu, AQMat, AAttnOutMat,
    FNumQHeads * FHeadDim, FHiddenDim, ANumTokens, AWeights.WeightType);
  FCompute.BatchBarrier(); // AAttnOutMat ready for caller
end;

procedure TVdxAttention.UploadAttnWeights(const AReader: TVdxGGUFReader;
  const ALayerIndex: Integer; out AWeights: TVdxAttnLayerWeights);

  // Upload one weight tensor (F16 or Q4_0) from GGUF to device-local GPU
  function UploadOneTensor(const ATensorName: string): TVdxGpuBuffer;
  var
    LInfo: TVdxGGUFTensorInfo;
    LPtr: Pointer;
    LSize: UInt64;
    LStaging: TVdxGpuBuffer;
  begin
    TVdxUtils.FailIf(not AReader.HasTensor(ATensorName),
      'Attention: tensor not found: %s', [ATensorName]);

    LInfo := AReader.GetTensorInfo(ATensorName);
    LPtr := AReader.GetTensorDataPtr(ATensorName);

    // Compute byte size — works for F16, Q4_0, and other types
    LSize := VdxGGMLTensorBytes(LInfo.TensorType,
      LInfo.Dimensions[0], LInfo.Dimensions[1]);
    TVdxUtils.FailIf(LSize = 0,
      'Attention: unsupported tensor type for %s: %s',
      [ATensorName, VdxGGMLTypeName(LInfo.TensorType)]);

    LStaging := FCompute.CreateGpuBuffer(
      LSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    try
      FCompute.UploadToBuffer(LStaging, LPtr, LSize);

      Result := FCompute.CreateGpuBuffer(
        LSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

      FCompute.CopyBuffer(LStaging, Result, LSize);
    finally
      FCompute.DestroyGpuBuffer(LStaging);
    end;
  end;

var
  LQInfo: TVdxGGUFTensorInfo;
begin
  AWeights := Default(TVdxAttnLayerWeights);

  // Detect weight type from Q tensor (all attn weights share the same type)
  LQInfo := AReader.GetTensorInfo(
    Format('blk.%d.attn_q.weight', [ALayerIndex]));
  AWeights.WeightType := LQInfo.TensorType;

  AWeights.QWeightGpu := UploadOneTensor(
    Format('blk.%d.attn_q.weight', [ALayerIndex]));
  AWeights.KWeightGpu := UploadOneTensor(
    Format('blk.%d.attn_k.weight', [ALayerIndex]));
  AWeights.VWeightGpu := UploadOneTensor(
    Format('blk.%d.attn_v.weight', [ALayerIndex]));
  AWeights.OWeightGpu := UploadOneTensor(
    Format('blk.%d.attn_output.weight', [ALayerIndex]));
end;

procedure TVdxAttention.FreeAttnWeights(var AWeights: TVdxAttnLayerWeights);
begin
  if AWeights.QWeightGpu.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(AWeights.QWeightGpu);
  if AWeights.KWeightGpu.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(AWeights.KWeightGpu);
  if AWeights.VWeightGpu.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(AWeights.VWeightGpu);
  if AWeights.OWeightGpu.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(AWeights.OWeightGpu);
end;

function TVdxAttention.GetKCache(const ALayerIndex: Integer): TVdxGpuBuffer;
begin
  Result := FKDecodeF32;  // shared decode buffer (diagnostic only)
end;

function TVdxAttention.GetVCache(const ALayerIndex: Integer): TVdxGpuBuffer;
begin
  Result := FVDecodeF32;  // shared decode buffer (diagnostic only)
end;

function TVdxAttention.GetLayerKCacheTQ3(const ALayerIndex: Integer): TVdxGpuBuffer;
begin
  Result := FKCacheTQ3[ALayerIndex];
end;

function TVdxAttention.GetLayerVCacheTQ3(const ALayerIndex: Integer): TVdxGpuBuffer;
begin
  Result := FVCacheTQ3[ALayerIndex];
end;

function TVdxAttention.GetLayerKVCacheTQ3Bytes(): UInt64;
begin
  // TQ3 layout: [NumKVHeads x MaxSeq x BlocksPerHead x 4] uint32 per layer.
  // Each block = 4 uint32 = 16 bytes. Matches the allocation in Init().
  Result := UInt64(FNumKVHeads) * FMaxSeqLen * (FHeadDim div 32) * 16;
end;

procedure TVdxAttention.Forward(const AInputBuf: TVdxGpuBuffer;
  const AWeights: TVdxAttnLayerWeights;
  const AQNormBuf: TVdxGpuBuffer;
  const AKNormBuf: TVdxGpuBuffer;
  const ALayerIndex: Integer;
  const APosition: Integer;
  const AThetaBase: Single;
  const AOutputBuf: TVdxGpuBuffer);
var
  LSeqLen: UInt32;
  LQKNormPush: TVdxQKNormPush;
  LRoPEPush: TVdxRoPEPush;
  LSoftmaxMHPush: TVdxSoftmaxMHPush;
  LValueMHPush: TVdxAttnValueMHPush;
  LKVStorePush: TVdxKVCacheStorePush;
  LTQ3QuantPush: TVdxTQ3KVQuantPush;
  LTQ3DequantPush: TVdxTQ3KVDequantPush;
  LScoresMHTQ3Push: TVdxAttnScoresMHTQ3Push;
begin
  LSeqLen := UInt32(APosition) + 1;

  // ---- Step 1: Q/K/V projections (matvec — F16 or Q4_0) ----
  // All three read AInputBuf and write separate output buffers → independent
  DispatchMatVec(AWeights.QWeightGpu, AInputBuf, FQBuf,
    FHiddenDim, FNumQHeads * FHeadDim, AWeights.WeightType);
  DispatchMatVec(AWeights.KWeightGpu, AInputBuf, FKBuf,
    FHiddenDim, FNumKVHeads * FHeadDim, AWeights.WeightType);
  DispatchMatVec(AWeights.VWeightGpu, AInputBuf, FVBuf,
    FHiddenDim, FNumKVHeads * FHeadDim, AWeights.WeightType);
  FCompute.BatchBarrier(); // Q/K/VBuf ready for QK-norm

  // ---- Step 2: QK-norm on Q (8 heads) and K (4 heads) ----
  LQKNormPush.HeadDim := FHeadDim;
  LQKNormPush.Eps := 1e-6;

  // QK-norm on Q
  LQKNormPush.NumHeads := FNumQHeads;
  FCompute.UpdateDescriptorSetBuffers(FQKNormDescSet, [FQBuf, AQNormBuf]);
  FCompute.DispatchComputeWithPush(
    FQKNormBundle.Pipeline, FQKNormBundle.PipelineLayout,
    FQKNormDescSet, @LQKNormPush, SizeOf(LQKNormPush), FNumQHeads);

  // QK-norm on K (independent of Q — writes different buffer)
  LQKNormPush.NumHeads := FNumKVHeads;
  FCompute.UpdateDescriptorSetBuffers(FQKNormDescSet, [FKBuf, AKNormBuf]);
  FCompute.DispatchComputeWithPush(
    FQKNormBundle.Pipeline, FQKNormBundle.PipelineLayout,
    FQKNormDescSet, @LQKNormPush, SizeOf(LQKNormPush), FNumKVHeads);
  FCompute.BatchBarrier(); // Q/KBuf normed, ready for RoPE
  // ---- Step 3: RoPE on Q and K ----
  LRoPEPush.HeadDim := FHeadDim;
  LRoPEPush.Position := UInt32(APosition);
  LRoPEPush.ThetaBase := AThetaBase;

  // RoPE on Q (8 heads)
  LRoPEPush.NumHeads := FNumQHeads;
  FCompute.UpdateDescriptorSetBuffers(FRoPEDescSet, [FQBuf]);
  FCompute.DispatchComputeWithPush(
    FRoPEBundle.Pipeline, FRoPEBundle.PipelineLayout,
    FRoPEDescSet, @LRoPEPush, SizeOf(LRoPEPush), FNumQHeads);

  // RoPE on K (4 heads — independent of Q, writes different buffer)
  LRoPEPush.NumHeads := FNumKVHeads;
  FCompute.UpdateDescriptorSetBuffers(FRoPEDescSet, [FKBuf]);
  FCompute.DispatchComputeWithPush(
    FRoPEBundle.Pipeline, FRoPEBundle.PipelineLayout,
    FRoPEDescSet, @LRoPEPush, SizeOf(LRoPEPush), FNumKVHeads);
  FCompute.BatchBarrier(); // Q/KBuf with RoPE applied, ready for KV cache + attn
  // ---- Step 4a: Store K and V into shared decode buffers ----
  LKVStorePush.HeadDim := FHeadDim;
  LKVStorePush.MaxSeq := FMaxSeqLen;
  LKVStorePush.Position := UInt32(APosition);
  LKVStorePush.NumKVHeads := FNumKVHeads;
  FCompute.UpdateDescriptorSetBuffers(FKVStoreDescSet,
    [FKBuf, FVBuf, FKDecodeF32, FVDecodeF32]);
  FCompute.DispatchComputeWithPush(
    FKVStoreBundle.Pipeline, FKVStoreBundle.PipelineLayout,
    FKVStoreDescSet, @LKVStorePush, SizeOf(LKVStorePush),
    FNumKVHeads);
  FCompute.BatchBarrier(); // Decode buffers have new position written

  // ---- Step 4b: Quantize current position from decode buffer → TQ3 cache ----
  LTQ3QuantPush.BlocksPerHead := FHeadDim div 32;
  LTQ3QuantPush.MaxSeq := FMaxSeqLen;
  LTQ3QuantPush.Position := UInt32(APosition);
  LTQ3QuantPush.NumHeads := FNumKVHeads;

  // Quantize K
  FCompute.UpdateDescriptorSetBuffers(FTQ3KVQuantDescSet,
    [FKDecodeF32, FKCacheTQ3[ALayerIndex]]);
  FCompute.DispatchComputeWithPush(
    FTQ3KVQuantBundle.Pipeline, FTQ3KVQuantBundle.PipelineLayout,
    FTQ3KVQuantDescSet, @LTQ3QuantPush, SizeOf(LTQ3QuantPush),
    (FHeadDim div 32) * FNumKVHeads);

  // Quantize V
  FCompute.UpdateDescriptorSetBuffers(FTQ3KVQuantDescSet,
    [FVDecodeF32, FVCacheTQ3[ALayerIndex]]);
  FCompute.DispatchComputeWithPush(
    FTQ3KVQuantBundle.Pipeline, FTQ3KVQuantBundle.PipelineLayout,
    FTQ3KVQuantDescSet, @LTQ3QuantPush, SizeOf(LTQ3QuantPush),
    (FHeadDim div 32) * FNumKVHeads);
  FCompute.BatchBarrier(); // TQ3 cache updated with new position

  // ---- Step 4c: Dequantize V positions from TQ3 cache → V decode buffer ----
  // K dequant eliminated by Phase 4 fused attention (reads TQ3 K directly)
  LTQ3DequantPush.BlocksPerHead := FHeadDim div 32;
  LTQ3DequantPush.MaxSeq := FMaxSeqLen;
  LTQ3DequantPush.SeqLen := LSeqLen;
  LTQ3DequantPush.NumHeads := FNumKVHeads;

  // Dequantize V only (K is read directly from TQ3 cache by fused attention)
  FCompute.UpdateDescriptorSetBuffers(FTQ3KVDequantDescSet,
    [FVCacheTQ3[ALayerIndex], FVDecodeF32]);
  FCompute.DispatchComputeWithPush(
    FTQ3KVDequantBundle.Pipeline, FTQ3KVDequantBundle.PipelineLayout,
    FTQ3KVDequantDescSet, @LTQ3DequantPush, SizeOf(LTQ3DequantPush),
    (FHeadDim div 32) * FNumKVHeads * LSeqLen);
  FCompute.BatchBarrier(); // V decode buffer has all positions dequantized

  // ---- Step 5: Multi-head attention ----

  // 5a. Fused TQ3 attention scores — reads Q + TQ3 K cache directly
  //     Applies WHT to Q on the fly, dots against packed TQ3 centroids
  LScoresMHTQ3Push.HeadDim := FHeadDim;
  LScoresMHTQ3Push.SeqLen := LSeqLen;
  LScoresMHTQ3Push.MaxSeq := FMaxSeqLen;
  LScoresMHTQ3Push.Scale := 1.0 / Sqrt(Single(FHeadDim));
  LScoresMHTQ3Push.NumQHeads := FNumQHeads;
  LScoresMHTQ3Push.GqaRatio := FNumQHeads div FNumKVHeads;
  LScoresMHTQ3Push.BlocksPerHead := FHeadDim div 32;

  FCompute.UpdateDescriptorSetBuffers(FScoresDescSet,
    [FQBuf, FKCacheTQ3[ALayerIndex], FScoresBuf]);
  FCompute.DispatchComputeWithPush(
    FAttnScoresMHTQ3Bundle.Pipeline, FAttnScoresMHTQ3Bundle.PipelineLayout,
    FScoresDescSet, @LScoresMHTQ3Push, SizeOf(LScoresMHTQ3Push),
    (LSeqLen + 255) div 256, FNumQHeads);
  FCompute.BatchBarrier(); // ScoresBuf ready for softmax

  // 5b. All heads' softmax in one dispatch (one workgroup per head)
  LSoftmaxMHPush.SeqLen := LSeqLen;
  LSoftmaxMHPush.MaxSeq := FMaxSeqLen;
  LSoftmaxMHPush.NumQHeads := FNumQHeads;

  FCompute.UpdateDescriptorSetBuffers(FSoftmaxDescSet, [FScoresBuf]);
  FCompute.DispatchComputeWithPush(
    FSoftmaxMHBundle.Pipeline, FSoftmaxMHBundle.PipelineLayout,
    FSoftmaxDescSet, @LSoftmaxMHPush, SizeOf(LSoftmaxMHPush),
    FNumQHeads);
  FCompute.BatchBarrier(); // Softmax weights ready for value

  // 5c. All heads' weighted V sum in one 2D dispatch
  LValueMHPush.HeadDim := FHeadDim;
  LValueMHPush.SeqLen := LSeqLen;
  LValueMHPush.MaxSeq := FMaxSeqLen;
  LValueMHPush.NumQHeads := FNumQHeads;
  LValueMHPush.GqaRatio := FNumQHeads div FNumKVHeads;

  FCompute.UpdateDescriptorSetBuffers(FValueDescSet,
    [FScoresBuf, FVDecodeF32, FAttnOutBuf]);
  FCompute.DispatchComputeWithPush(
    FAttnValueMHBundle.Pipeline, FAttnValueMHBundle.PipelineLayout,
    FValueDescSet, @LValueMHPush, SizeOf(LValueMHPush),
    (FHeadDim + 255) div 256, FNumQHeads);
  FCompute.BatchBarrier(); // AttnOutBuf ready for O matvec
  // ---- Step 6: Output projection ----
  // O: W[NumQHeads*HeadDim × HiddenDim] × attn_out → output
  DispatchMatVec(AWeights.OWeightGpu, FAttnOutBuf, AOutputBuf,
    FNumQHeads * FHeadDim, FHiddenDim, AWeights.WeightType);
  FCompute.BatchBarrier(); // AOutputBuf ready for caller
end;

end.