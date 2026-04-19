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
  VindexLLM.Resources,
  VindexLLM.Vulkan,
  VindexLLM.Compute,
  VindexLLM.Shaders,
  VindexLLM.GGUFReader;

const

  //--------------------------------------------------------------------------
  // Error Codes
  //--------------------------------------------------------------------------
  VDX_ERROR_ATTN_COMPUTE_NIL          = 'AT01';
  VDX_ERROR_ATTN_ALREADY_INIT         = 'AT02';
  VDX_ERROR_ATTN_NOT_INIT             = 'AT03';
  VDX_ERROR_ATTN_INIT_EXCEPTION       = 'AT04';
  VDX_ERROR_ATTN_RESOLVE_EXCEPTION    = 'AT05';
  VDX_ERROR_ATTN_TENSOR_NOT_FOUND     = 'AT06';
  VDX_ERROR_ATTN_UNSUPPORTED_TYPE     = 'AT07';

type

  //==========================================================================
  // Push-constant records — match the layout of the corresponding SPIR-V
  // shaders byte-for-byte. Do not reorder fields or change widths; any
  // divergence breaks the shader binding.
  //==========================================================================

  { TVdxMatVecF16Push }
  TVdxMatVecF16Push = record
    InDimHalf: UInt32;
    OutDim:    UInt32;
  end;

  { TVdxQKNormPush }
  TVdxQKNormPush = record
    HeadDim:  UInt32;
    NumHeads: UInt32;
    Eps:      Single;
  end;

  { TVdxRoPEPush }
  TVdxRoPEPush = record
    HeadDim:   UInt32;
    NumHeads:  UInt32;
    Position:  UInt32;
    ThetaBase: Single;
  end;

  { TVdxAttnScoresMHPush }
  TVdxAttnScoresMHPush = record
    HeadDim:   UInt32;
    SeqLen:    UInt32;
    MaxSeq:    UInt32;
    Scale:     Single;
    NumQHeads: UInt32;
    GqaRatio:  UInt32;
  end;

  { TVdxAttnScoresMHTQ3Push }
  TVdxAttnScoresMHTQ3Push = record
    HeadDim:       UInt32;
    SeqLen:        UInt32;
    MaxSeq:        UInt32;
    Scale:         Single;
    NumQHeads:     UInt32;
    GqaRatio:      UInt32;
    BlocksPerHead: UInt32;
  end;

  { TVdxSoftmaxMHPush }
  TVdxSoftmaxMHPush = record
    SeqLen:    UInt32;
    MaxSeq:    UInt32;
    NumQHeads: UInt32;
  end;

  { TVdxAttnValueMHPush }
  TVdxAttnValueMHPush = record
    HeadDim:   UInt32;
    SeqLen:    UInt32;
    MaxSeq:    UInt32;
    NumQHeads: UInt32;
    GqaRatio:  UInt32;
  end;

  { TVdxKVCacheStorePush }
  TVdxKVCacheStorePush = record
    HeadDim:    UInt32;
    MaxSeq:     UInt32;
    Position:   UInt32;
    NumKVHeads: UInt32;
  end;

  { TVdxMatMulPush }
  TVdxMatMulPush = record
    InDimParam: UInt32;  // in_dim/2 for F16, in_dim for Q8_0/Q4_0
    OutDim:     UInt32;
    NumTokens:  UInt32;
  end;

  { TVdxRoPEBatchPush }
  TVdxRoPEBatchPush = record
    HeadDim:   UInt32;
    NumHeads:  UInt32;
    NumTokens: UInt32;
    ThetaBase: Single;
    StartPos:  UInt32;
  end;

  { TVdxKVCacheStoreBatchPush }
  TVdxKVCacheStoreBatchPush = record
    HeadDim:    UInt32;
    MaxSeq:     UInt32;
    NumKVHeads: UInt32;
    NumTokens:  UInt32;
    StartPos:   UInt32;
  end;

  { TVdxTQ3KVQuantPush }
  TVdxTQ3KVQuantPush = record
    BlocksPerHead: UInt32;
    MaxSeq:        UInt32;
    Position:      UInt32;
    NumHeads:      UInt32;
  end;

  { TVdxTQ3KVDequantPush }
  TVdxTQ3KVDequantPush = record
    BlocksPerHead: UInt32;
    MaxSeq:        UInt32;
    SeqLen:        UInt32;
    NumHeads:      UInt32;
  end;

  { TVdxKVStoreBatchTQ3Push }
  TVdxKVStoreBatchTQ3Push = record
    HeadDim:   UInt32;
    MaxSeq:    UInt32;
    NumHeads:  UInt32;
    NumTokens: UInt32;
    StartPos:  UInt32;
  end;

  { TVdxAttnScoresPrefillPush }
  TVdxAttnScoresPrefillPush = record
    HeadDim:   UInt32;
    NumTokens: UInt32;
    MaxSeq:    UInt32;
    Scale:     Single;
    NumQHeads: UInt32;
    GqaRatio:  UInt32;
    StartPos:  UInt32;
    SeqLen:    UInt32;
  end;

  { TVdxSoftmaxPrefillPush }
  TVdxSoftmaxPrefillPush = record
    NumTokens: UInt32;
    NumQHeads: UInt32;
    MaxSeq:    UInt32;
    SeqLen:    UInt32;
    StartPos:  UInt32;
  end;

  { TVdxAttnValuePrefillPush }
  TVdxAttnValuePrefillPush = record
    HeadDim:   UInt32;
    NumTokens: UInt32;
    MaxSeq:    UInt32;
    NumQHeads: UInt32;
    GqaRatio:  UInt32;
    StartPos:  UInt32;
    SeqLen:    UInt32;
  end;

  { TVdxAttnLayerWeights }
  // Streaming weight reference for one attention layer. Holds mmap
  // pointers into the GGUF tensor region — NOT permanent GPU buffers.
  // Populated by TVdxAttention.ResolveAttnWeights; consumed by
  // TVdxAttention.Forward / ForwardBatch, which copy the slice into
  // the shared staging pool owned by TVdxCompute before each
  // dispatch. Valid only while the source TVdxGGUFReader is open.
  TVdxAttnLayerWeights = record
    QWeightPtr:   PByte;
    KWeightPtr:   PByte;
    VWeightPtr:   PByte;
    OWeightPtr:   PByte;
    QWeightBytes: UInt64;
    KWeightBytes: UInt64;
    VWeightBytes: UInt64;
    OWeightBytes: UInt64;
    WeightType:   TVdxGGMLType;
  end;

  { TVdxAttention }
  // Streaming attention compute — Q/K/V/O weights are NOT permanently
  // uploaded to per-layer VRAM. Each forward call copies the slice it
  // needs from the mmap'd GGUF into the shared staging buffer pair
  // owned by TVdxCompute, then dispatches. See
  // .claude/tasks/TASK-REFACTOR.md Phase 7 for the full design.
  //
  // Phase 7B: Init builds all shaders, pipelines, descriptor pools,
  // scratch buffers, and per-layer KV caches. ResolveAttnWeights
  // records mmap pointers for one layer. Forward / ForwardBatch
  // land in Phase 7C / 7D.
  TVdxAttention = class(TVdxBaseObject)
  private
    FCompute:     TVdxCompute;
    FInitialized: Boolean;

    // Model dimensions — stashed by Init, consumed by everything else.
    FHiddenDim:  UInt32;
    FNumQHeads:  UInt32;
    FNumKVHeads: UInt32;
    FHeadDim:    UInt32;
    FMaxSeqLen:  UInt32;
    FNumLayers:  UInt32;
    FFFNWidth:   UInt32;

    // Shader modules — single-token decode path
    FMatVecShader:       VkShaderModule;
    FMatVecQ8Shader:     VkShaderModule;
    FMatVecQ4Shader:     VkShaderModule;
    FQKNormShader:       VkShaderModule;
    FRoPEShader:         VkShaderModule;
    FAttnScoresMHShader: VkShaderModule;
    FSoftmaxMHShader:    VkShaderModule;
    FAttnValueMHShader:  VkShaderModule;
    FKVStoreShader:      VkShaderModule;

    // Shader modules — batch matmul (prefill projections)
    FMatMulF16Shader: VkShaderModule;
    FMatMulQ8Shader:  VkShaderModule;
    FMatMulQ4Shader:  VkShaderModule;

    // Shader modules — prefill attention
    FRoPEBatchShader:          VkShaderModule;
    FKVStoreBatchShader:       VkShaderModule;
    FScoresPrefillShader:      VkShaderModule;
    FScoresPrefillBidirShader: VkShaderModule;
    FSoftmaxPrefillShader:     VkShaderModule;
    FValuePrefillShader:       VkShaderModule;

    // Shader modules — TQ3 KV cache compression
    FTQ3KVQuantShader:      VkShaderModule;
    FTQ3KVDequantShader:    VkShaderModule;
    FKVStoreBatchTQ3Shader: VkShaderModule;
    FAttnScoresMHTQ3Shader: VkShaderModule;

    // Pipeline bundles — single-token decode path
    FMatVecBundle:       TVdxComputePipelineBundle;
    FMatVecQ8Bundle:     TVdxComputePipelineBundle;
    FMatVecQ4Bundle:     TVdxComputePipelineBundle;
    FQKNormBundle:       TVdxComputePipelineBundle;
    FRoPEBundle:         TVdxComputePipelineBundle;
    FAttnScoresMHBundle: TVdxComputePipelineBundle;
    FSoftmaxMHBundle:    TVdxComputePipelineBundle;
    FAttnValueMHBundle:  TVdxComputePipelineBundle;
    FKVStoreBundle:      TVdxComputePipelineBundle;

    // Pipeline bundles — batch matmul
    FMatMulF16Bundle: TVdxComputePipelineBundle;
    FMatMulQ8Bundle:  TVdxComputePipelineBundle;
    FMatMulQ4Bundle:  TVdxComputePipelineBundle;

    // Pipeline bundles — prefill attention
    FRoPEBatchBundle:          TVdxComputePipelineBundle;
    FKVStoreBatchBundle:       TVdxComputePipelineBundle;
    FScoresPrefillBundle:      TVdxComputePipelineBundle;
    FScoresPrefillBidirBundle: TVdxComputePipelineBundle;
    FSoftmaxPrefillBundle:     TVdxComputePipelineBundle;
    FValuePrefillBundle:       TVdxComputePipelineBundle;

    // Pipeline bundles — TQ3
    FTQ3KVQuantBundle:      TVdxComputePipelineBundle;
    FTQ3KVDequantBundle:    TVdxComputePipelineBundle;
    FKVStoreBatchTQ3Bundle: TVdxComputePipelineBundle;
    FAttnScoresMHTQ3Bundle: TVdxComputePipelineBundle;

    // Descriptor set layouts
    FMatVecDescLayout:          VkDescriptorSetLayout;  // 3 bindings
    FQKNormDescLayout:          VkDescriptorSetLayout;  // 2 bindings
    FRoPEDescLayout:            VkDescriptorSetLayout;  // 1 binding
    FAttnScoresDescLayout:      VkDescriptorSetLayout;  // 3 bindings
    FSoftmaxDescLayout:         VkDescriptorSetLayout;  // 1 binding
    FAttnValueDescLayout:       VkDescriptorSetLayout;  // 3 bindings
    FKVStoreDescLayout:         VkDescriptorSetLayout;  // 4 bindings
    FTQ3KVDescLayout:           VkDescriptorSetLayout;  // 2 bindings
    FKVStoreBatchTQ3DescLayout: VkDescriptorSetLayout;  // 3 bindings

    // Descriptor pools + pre-allocated sets — single-token path
    FDescPool:       VkDescriptorPool;
    FMatVecDescSet:  VkDescriptorSet;
    FQKNormDescSet:  VkDescriptorSet;
    FRoPEDescSet:    VkDescriptorSet;
    FScoresDescSet:  VkDescriptorSet;
    FSoftmaxDescSet: VkDescriptorSet;
    FValueDescSet:   VkDescriptorSet;

    // Descriptor pools + pre-allocated sets — other paths
    FKVStoreDescPool:         VkDescriptorPool;
    FKVStoreDescSet:          VkDescriptorSet;
    FPrefillDescPool:         VkDescriptorPool;
    FPrefillRoPEDescSet:      VkDescriptorSet;
    FPrefillKVStoreDescSet:   VkDescriptorSet;
    FPrefillScoresDescSet:    VkDescriptorSet;
    FPrefillSoftmaxDescSet:   VkDescriptorSet;
    FPrefillValueDescSet:     VkDescriptorSet;
    FTQ3KVDescPool:           VkDescriptorPool;
    FTQ3KVQuantDescSet:       VkDescriptorSet;
    FTQ3KVDequantDescSet:     VkDescriptorSet;
    FKVStoreBatchTQ3DescPool: VkDescriptorPool;
    FKVStoreBatchTQ3DescSet:  VkDescriptorSet;

    // Scratch buffers (reused every Forward call)
    FQBuf:             TVdxGpuBuffer;  // [NumQHeads * HeadDim] F32
    FKBuf:             TVdxGpuBuffer;  // [NumKVHeads * HeadDim] F32
    FVBuf:             TVdxGpuBuffer;  // [NumKVHeads * HeadDim] F32
    FScoresBuf:        TVdxGpuBuffer;  // [NumQHeads * MaxSeq] F32
    FAttnOutBuf:       TVdxGpuBuffer;  // [NumQHeads * HeadDim] F32
    FPrefillScoresBuf: TVdxGpuBuffer;  // [NumQHeads * MaxSeq * MaxSeq] F32

    // Per-layer TQ3 compressed KV caches — permanent residents (not
    // streamed; caches are written across the sequence and must
    // persist).
    FKCacheTQ3: array of TVdxGpuBuffer;
    FVCacheTQ3: array of TVdxGpuBuffer;

    // Shared F32 decode buffers (one pair reused across all layers)
    FKDecodeF32: TVdxGpuBuffer;
    FVDecodeF32: TVdxGpuBuffer;

    // Helpers
    function  LoadShader(const AName: string): VkShaderModule;
    procedure DispatchMatVec(const AWeightBuf: TVdxGpuBuffer;
      const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
      const AInDim: UInt32; const AOutDim: UInt32;
      const ATensorType: TVdxGGMLType);
    procedure StreamAndDispatchMatVec(const AStagingIndex: UInt32;
      const AWeightPtr: PByte; const AWeightBytes: UInt64;
      const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
      const AInDim: UInt32; const AOutDim: UInt32;
      const ATensorType: TVdxGGMLType);
    procedure DispatchBatchMatMul(const AWeightBuf: TVdxGpuBuffer;
      const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
      const AInDim: UInt32; const AOutDim: UInt32;
      const ANumTokens: UInt32; const ATensorType: TVdxGGMLType);
    procedure StreamAndDispatchBatchMatMul(const AStagingIndex: UInt32;
      const AWeightPtr: PByte; const AWeightBytes: UInt64;
      const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
      const AInDim: UInt32; const AOutDim: UInt32;
      const ANumTokens: UInt32; const ATensorType: TVdxGGMLType);
  public
    constructor Create(); override;
    destructor  Destroy(); override;

    // Initialize all shaders, pipelines, scratch buffers, per-layer
    // KV caches, and grow the shared streaming-staging pool on
    // TVdxCompute to the largest slice this layer might ever upload
    // (sized to the F16 upper bound — smaller quant types fit
    // trivially). Returns False with FErrors populated on failure;
    // partially-constructed state is rolled back via Cleanup.
    function Init(const ACompute: TVdxCompute;
      const AHiddenDim: UInt32;
      const ANumQHeads: UInt32;
      const ANumKVHeads: UInt32;
      const AHeadDim: UInt32;
      const ANumLayers: UInt32;
      const AMaxSeqLen: UInt32;
      const AFFNWidth: UInt32): Boolean;

    // Release all GPU resources. Safe to call on an uninitialized,
    // partially-initialized, or already-cleaned-up instance.
    procedure Cleanup();

    // Resolve mmap pointers + byte sizes for one layer's Q/K/V/O
    // attention weight tensors. Zero GPU allocation — the returned
    // record just references the GGUF mapping. Pointers are valid
    // until AReader is closed.
    function ResolveAttnWeights(const AReader: TVdxGGUFReader;
      const ALayerIndex: Integer;
      out AWeights: TVdxAttnLayerWeights): Boolean;

    // Run full attention for one layer at one position. Must be
    // called inside an active batch (FCompute.BeginBatch/EndBatch) —
    // streaming the Q/K/V/O weight slices into the staging pool
    // depends on all CopyBuffer + dispatch commands being recorded
    // into a single submit, so a CPU memcpy into host[i] never
    // overwrites bytes the GPU hasn't yet consumed from pair[i].
    // Caller wraps; this routine does not open or close a batch.
    //
    // AInputBuf:       pre-normed residual [HiddenDim] F32
    // AWeights:        mmap pointers + byte sizes from ResolveAttnWeights
    // AQNormBuf:       QK-norm weights for Q heads [HeadDim] F32
    // AKNormBuf:       QK-norm weights for K heads [HeadDim] F32
    // ALayerIndex:     per-layer KV cache slot
    // APosition:       absolute sequence position (0-based)
    // AThetaBase:      RoPE theta for this layer (Gemma 3: 10k sliding,
    //                  1M for global-attention layers 5/11/17/23/29)
    // AOutputBuf:      attention output [HiddenDim] F32 — caller adds
    //                  this to the residual stream
    procedure Forward(const AInputBuf: TVdxGpuBuffer;
      const AWeights: TVdxAttnLayerWeights;
      const AQNormBuf: TVdxGpuBuffer;
      const AKNormBuf: TVdxGpuBuffer;
      const ALayerIndex: Integer;
      const APosition: Integer;
      const AThetaBase: Single;
      const AOutputBuf: TVdxGpuBuffer);

    // Batched prefill path for N tokens through one layer. Same
    // batch-mode contract as Forward — caller wraps in
    // BeginBatch/EndBatch. Streams Q/K/V/O weight slices through
    // staging pairs 0/1/2/3, same as single-token Forward.
    //
    // AInputMat:       pre-normed residual [NumTokens x HiddenDim] F32
    // AQMat/KMat/VMat: workspace matrices, pre-allocated by caller
    //                  [NumTokens x (Num*Heads * HeadDim)] F32. Reused
    //                  across layers to avoid per-layer allocation.
    // AAttnOutMat:     output [NumTokens x HiddenDim] F32 — caller adds
    //                  to residual.
    // AStartPos:       absolute KV slot at which to begin writing this
    //                  batch. 0 for fresh prefill, non-zero for
    //                  continuation (reloaded session, incremental
    //                  prefill).
    // ABidirectional:  True skips the causal mask (encoder-style
    //                  attention, EmbeddingGemma). Defaults to False
    //                  for Gemma 3 decoder.
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
      const AAttnOutMat: TVdxGpuBuffer;
      const ABidirectional: Boolean = False);

    // Non-streaming batch matmul — for callers that already have a
    // weight tensor resident in a GPU buffer (e.g. the logit
    // projection in the Model class). Not used by Forward /
    // ForwardBatch themselves; those stream.
    procedure BatchMatMul(const AWeightBuf: TVdxGpuBuffer;
      const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
      const AInDim: UInt32; const AOutDim: UInt32;
      const ANumTokens: UInt32;
      const ATensorType: TVdxGGMLType = gtF16);

    // Non-streaming matvec — diagnostic passthrough for tests and
    // for callers that already own a weight buffer.
    procedure TestMatVec(const AWeightBuf: TVdxGpuBuffer;
      const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
      const AInDim: UInt32; const AOutDim: UInt32;
      const ATensorType: TVdxGGMLType = gtF16);

    property Initialized: Boolean read FInitialized;
  end;

implementation

{ TVdxAttention }

constructor TVdxAttention.Create();
begin
  inherited;

  FCompute     := nil;
  FInitialized := False;

  FHiddenDim  := 0;
  FNumQHeads  := 0;
  FNumKVHeads := 0;
  FHeadDim    := 0;
  FMaxSeqLen  := 0;
  FNumLayers  := 0;
  FFFNWidth   := 0;

  // Shader modules
  FMatVecShader             := VK_NULL_HANDLE;
  FMatVecQ8Shader           := VK_NULL_HANDLE;
  FMatVecQ4Shader           := VK_NULL_HANDLE;
  FQKNormShader             := VK_NULL_HANDLE;
  FRoPEShader               := VK_NULL_HANDLE;
  FAttnScoresMHShader       := VK_NULL_HANDLE;
  FSoftmaxMHShader          := VK_NULL_HANDLE;
  FAttnValueMHShader        := VK_NULL_HANDLE;
  FKVStoreShader            := VK_NULL_HANDLE;
  FMatMulF16Shader          := VK_NULL_HANDLE;
  FMatMulQ8Shader           := VK_NULL_HANDLE;
  FMatMulQ4Shader           := VK_NULL_HANDLE;
  FRoPEBatchShader          := VK_NULL_HANDLE;
  FKVStoreBatchShader       := VK_NULL_HANDLE;
  FScoresPrefillShader      := VK_NULL_HANDLE;
  FScoresPrefillBidirShader := VK_NULL_HANDLE;
  FSoftmaxPrefillShader     := VK_NULL_HANDLE;
  FValuePrefillShader       := VK_NULL_HANDLE;
  FTQ3KVQuantShader         := VK_NULL_HANDLE;
  FTQ3KVDequantShader       := VK_NULL_HANDLE;
  FKVStoreBatchTQ3Shader    := VK_NULL_HANDLE;
  FAttnScoresMHTQ3Shader    := VK_NULL_HANDLE;

  // Pipeline bundles
  FMatVecBundle             := Default(TVdxComputePipelineBundle);
  FMatVecQ8Bundle           := Default(TVdxComputePipelineBundle);
  FMatVecQ4Bundle           := Default(TVdxComputePipelineBundle);
  FQKNormBundle             := Default(TVdxComputePipelineBundle);
  FRoPEBundle               := Default(TVdxComputePipelineBundle);
  FAttnScoresMHBundle       := Default(TVdxComputePipelineBundle);
  FSoftmaxMHBundle          := Default(TVdxComputePipelineBundle);
  FAttnValueMHBundle        := Default(TVdxComputePipelineBundle);
  FKVStoreBundle            := Default(TVdxComputePipelineBundle);
  FMatMulF16Bundle          := Default(TVdxComputePipelineBundle);
  FMatMulQ8Bundle           := Default(TVdxComputePipelineBundle);
  FMatMulQ4Bundle           := Default(TVdxComputePipelineBundle);
  FRoPEBatchBundle          := Default(TVdxComputePipelineBundle);
  FKVStoreBatchBundle       := Default(TVdxComputePipelineBundle);
  FScoresPrefillBundle      := Default(TVdxComputePipelineBundle);
  FScoresPrefillBidirBundle := Default(TVdxComputePipelineBundle);
  FSoftmaxPrefillBundle     := Default(TVdxComputePipelineBundle);
  FValuePrefillBundle       := Default(TVdxComputePipelineBundle);
  FTQ3KVQuantBundle         := Default(TVdxComputePipelineBundle);
  FTQ3KVDequantBundle       := Default(TVdxComputePipelineBundle);
  FKVStoreBatchTQ3Bundle    := Default(TVdxComputePipelineBundle);
  FAttnScoresMHTQ3Bundle    := Default(TVdxComputePipelineBundle);

  // Descriptor layouts
  FMatVecDescLayout          := VK_NULL_HANDLE;
  FQKNormDescLayout          := VK_NULL_HANDLE;
  FRoPEDescLayout            := VK_NULL_HANDLE;
  FAttnScoresDescLayout      := VK_NULL_HANDLE;
  FSoftmaxDescLayout         := VK_NULL_HANDLE;
  FAttnValueDescLayout       := VK_NULL_HANDLE;
  FKVStoreDescLayout         := VK_NULL_HANDLE;
  FTQ3KVDescLayout           := VK_NULL_HANDLE;
  FKVStoreBatchTQ3DescLayout := VK_NULL_HANDLE;

  // Descriptor pools + sets
  FDescPool                := VK_NULL_HANDLE;
  FMatVecDescSet           := VK_NULL_HANDLE;
  FQKNormDescSet           := VK_NULL_HANDLE;
  FRoPEDescSet             := VK_NULL_HANDLE;
  FScoresDescSet           := VK_NULL_HANDLE;
  FSoftmaxDescSet          := VK_NULL_HANDLE;
  FValueDescSet            := VK_NULL_HANDLE;
  FKVStoreDescPool         := VK_NULL_HANDLE;
  FKVStoreDescSet          := VK_NULL_HANDLE;
  FPrefillDescPool         := VK_NULL_HANDLE;
  FPrefillRoPEDescSet      := VK_NULL_HANDLE;
  FPrefillKVStoreDescSet   := VK_NULL_HANDLE;
  FPrefillScoresDescSet    := VK_NULL_HANDLE;
  FPrefillSoftmaxDescSet   := VK_NULL_HANDLE;
  FPrefillValueDescSet     := VK_NULL_HANDLE;
  FTQ3KVDescPool           := VK_NULL_HANDLE;
  FTQ3KVQuantDescSet       := VK_NULL_HANDLE;
  FTQ3KVDequantDescSet     := VK_NULL_HANDLE;
  FKVStoreBatchTQ3DescPool := VK_NULL_HANDLE;
  FKVStoreBatchTQ3DescSet  := VK_NULL_HANDLE;

  // Scratch buffers + shared decode buffers
  FQBuf             := Default(TVdxGpuBuffer);
  FKBuf             := Default(TVdxGpuBuffer);
  FVBuf             := Default(TVdxGpuBuffer);
  FScoresBuf        := Default(TVdxGpuBuffer);
  FAttnOutBuf       := Default(TVdxGpuBuffer);
  FPrefillScoresBuf := Default(TVdxGpuBuffer);
  FKDecodeF32       := Default(TVdxGpuBuffer);
  FVDecodeF32       := Default(TVdxGpuBuffer);

  SetLength(FKCacheTQ3, 0);
  SetLength(FVCacheTQ3, 0);
end;

destructor TVdxAttention.Destroy();
begin
  Cleanup();
  inherited;
end;

function TVdxAttention.LoadShader(const AName: string): VkShaderModule;
var
  LSpv: TBytes;
begin
  LSpv := VdxLoadShader(AName);
  Result := FCompute.CreateShaderModule(@LSpv[0], NativeUInt(Length(LSpv)));
end;

procedure TVdxAttention.Cleanup();
var
  LI: Integer;
begin
  // Cleanup is safe on partially-constructed state. Every handle is
  // checked against its null sentinel before destroy. FCompute may
  // still be nil if Init never ran — guard at the top.
  if FCompute = nil then
  begin
    FInitialized := False;
    Exit;
  end;

  // Per-layer TQ3 caches
  for LI := 0 to Length(FKCacheTQ3) - 1 do
  begin
    if FKCacheTQ3[LI].Buffer <> VK_NULL_HANDLE then
      FCompute.DestroyGpuBuffer(FKCacheTQ3[LI]);
    if FVCacheTQ3[LI].Buffer <> VK_NULL_HANDLE then
      FCompute.DestroyGpuBuffer(FVCacheTQ3[LI]);
  end;
  SetLength(FKCacheTQ3, 0);
  SetLength(FVCacheTQ3, 0);

  // Shared F32 decode buffers
  if FKDecodeF32.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(FKDecodeF32);
  if FVDecodeF32.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(FVDecodeF32);

  // Scratch buffers
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
  if FPrefillScoresBuf.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(FPrefillScoresBuf);

  // Single-token pipelines
  FCompute.DestroyComputePipelineBundle(FMatVecBundle);
  FCompute.DestroyComputePipelineBundle(FMatVecQ8Bundle);
  FCompute.DestroyComputePipelineBundle(FMatVecQ4Bundle);
  FCompute.DestroyComputePipelineBundle(FQKNormBundle);
  FCompute.DestroyComputePipelineBundle(FRoPEBundle);
  FCompute.DestroyComputePipelineBundle(FAttnScoresMHBundle);
  FCompute.DestroyComputePipelineBundle(FSoftmaxMHBundle);
  FCompute.DestroyComputePipelineBundle(FAttnValueMHBundle);

  // KV store (single-token)
  FCompute.DestroyComputePipelineBundle(FKVStoreBundle);
  if FKVStoreDescPool <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorPoolHandle(FKVStoreDescPool);
  if FKVStoreDescLayout <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorSetLayoutHandle(FKVStoreDescLayout);
  if FKVStoreShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FKVStoreShader);

  // Batch matmul
  FCompute.DestroyComputePipelineBundle(FMatMulF16Bundle);
  FCompute.DestroyComputePipelineBundle(FMatMulQ8Bundle);
  FCompute.DestroyComputePipelineBundle(FMatMulQ4Bundle);
  if FMatMulF16Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FMatMulF16Shader);
  if FMatMulQ8Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FMatMulQ8Shader);
  if FMatMulQ4Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FMatMulQ4Shader);

  // Prefill attention
  if FPrefillDescPool <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorPoolHandle(FPrefillDescPool);
  FCompute.DestroyComputePipelineBundle(FRoPEBatchBundle);
  FCompute.DestroyComputePipelineBundle(FKVStoreBatchBundle);
  FCompute.DestroyComputePipelineBundle(FScoresPrefillBundle);
  FCompute.DestroyComputePipelineBundle(FScoresPrefillBidirBundle);
  FCompute.DestroyComputePipelineBundle(FSoftmaxPrefillBundle);
  FCompute.DestroyComputePipelineBundle(FValuePrefillBundle);
  if FRoPEBatchShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FRoPEBatchShader);
  if FKVStoreBatchShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FKVStoreBatchShader);
  if FScoresPrefillShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FScoresPrefillShader);
  if FScoresPrefillBidirShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FScoresPrefillBidirShader);
  if FSoftmaxPrefillShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FSoftmaxPrefillShader);
  if FValuePrefillShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FValuePrefillShader);

  // Single-token descriptor pool (owns 6 reusable sets)
  if FDescPool <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorPoolHandle(FDescPool);

  // Descriptor layouts — tear down after pools that reference them
  if FMatVecDescLayout <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorSetLayoutHandle(FMatVecDescLayout);
  if FQKNormDescLayout <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorSetLayoutHandle(FQKNormDescLayout);
  if FRoPEDescLayout <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorSetLayoutHandle(FRoPEDescLayout);
  if FAttnScoresDescLayout <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorSetLayoutHandle(FAttnScoresDescLayout);
  if FSoftmaxDescLayout <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorSetLayoutHandle(FSoftmaxDescLayout);
  if FAttnValueDescLayout <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorSetLayoutHandle(FAttnValueDescLayout);

  // Shader modules — single-token
  if FMatVecShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FMatVecShader);
  if FMatVecQ8Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FMatVecQ8Shader);
  if FMatVecQ4Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FMatVecQ4Shader);
  if FQKNormShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FQKNormShader);
  if FRoPEShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FRoPEShader);
  if FAttnScoresMHShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FAttnScoresMHShader);
  if FSoftmaxMHShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FSoftmaxMHShader);
  if FAttnValueMHShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FAttnValueMHShader);

  // TQ3 resources
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

  if FKVStoreBatchTQ3DescPool <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorPoolHandle(FKVStoreBatchTQ3DescPool);
  FCompute.DestroyComputePipelineBundle(FKVStoreBatchTQ3Bundle);
  if FKVStoreBatchTQ3DescLayout <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorSetLayoutHandle(FKVStoreBatchTQ3DescLayout);
  if FKVStoreBatchTQ3Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FKVStoreBatchTQ3Shader);

  FCompute.DestroyComputePipelineBundle(FAttnScoresMHTQ3Bundle);
  if FAttnScoresMHTQ3Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FAttnScoresMHTQ3Shader);

  FInitialized := False;
  FCompute     := nil;
end;

function TVdxAttention.Init(const ACompute: TVdxCompute;
  const AHiddenDim: UInt32;
  const ANumQHeads: UInt32;
  const ANumKVHeads: UInt32;
  const AHeadDim: UInt32;
  const ANumLayers: UInt32;
  const AMaxSeqLen: UInt32;
  const AFFNWidth: UInt32): Boolean;
var
  LDummyBuf:    TVdxGpuBuffer;
  LI:           Integer;
  LCacheSize:   UInt64;
  LMaxQ8Blocks: UInt32;
  LMaxSliceF16: UInt64;
  LQDim:        UInt64;
  LKVDim:       UInt64;
begin
  Result := False;

  if FInitialized then
  begin
    FErrors.Add(esError, VDX_ERROR_ATTN_ALREADY_INIT, RSAttnAlreadyInit);
    Exit;
  end;

  if ACompute = nil then
  begin
    FErrors.Add(esFatal, VDX_ERROR_ATTN_COMPUTE_NIL, RSAttnComputeNil);
    Exit;
  end;

  FCompute    := ACompute;
  FHiddenDim  := AHiddenDim;
  FNumQHeads  := ANumQHeads;
  FNumKVHeads := ANumKVHeads;
  FHeadDim    := AHeadDim;
  FNumLayers  := ANumLayers;
  FMaxSeqLen  := AMaxSeqLen;
  FFFNWidth   := AFFNWidth;

  // Max Q8_0 blocks per row across all matvec uses (attention + FFN).
  // FFN-down has the largest in_dim (AFFNWidth), so it determines the
  // ceiling. Passed as spec-constant to the MATVEC_Q8_0 / MATMUL_Q8_0
  // pipelines so they can declare a shared-memory array of the right
  // size.
  LMaxQ8Blocks := AFFNWidth div 32;

  LDummyBuf := Default(TVdxGpuBuffer);

  Status('Attention: Init (H=%d Q=%d KV=%d hd=%d L=%d S=%d F=%d)',
    [AHiddenDim, ANumQHeads, ANumKVHeads, AHeadDim,
     ANumLayers, AMaxSeqLen, AFFNWidth]);

  // Outer try/finally guarantees Cleanup runs on any early-exit.
  // Inner try/except converts raises from VdxLoadShader or any other
  // RTL call into esFatal errors without propagating past this
  // boundary.
  try
    try
      //------------------------------------------------------------------
      // Shaders — single-token decode path
      //------------------------------------------------------------------
      FMatVecShader       := LoadShader('MATVEC_F16');
      if FErrors.HasFatal() then Exit;
      FMatVecQ8Shader     := LoadShader('MATVEC_Q8_0');
      if FErrors.HasFatal() then Exit;
      FMatVecQ4Shader     := LoadShader('MATVEC_Q4_0');
      if FErrors.HasFatal() then Exit;
      FQKNormShader       := LoadShader('QK_NORM');
      if FErrors.HasFatal() then Exit;
      FRoPEShader         := LoadShader('ROPE');
      if FErrors.HasFatal() then Exit;
      FAttnScoresMHShader := LoadShader('ATTN_SCORES_MH');
      if FErrors.HasFatal() then Exit;
      FSoftmaxMHShader    := LoadShader('SOFTMAX_MH');
      if FErrors.HasFatal() then Exit;
      FAttnValueMHShader  := LoadShader('ATTN_VALUE_MH');
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Descriptor set layouts
      //------------------------------------------------------------------
      FMatVecDescLayout     := FCompute.CreateStorageDescriptorSetLayout(3);
      if FErrors.HasFatal() then Exit;
      FQKNormDescLayout     := FCompute.CreateStorageDescriptorSetLayout(2);
      if FErrors.HasFatal() then Exit;
      FRoPEDescLayout       := FCompute.CreateStorageDescriptorSetLayout(1);
      if FErrors.HasFatal() then Exit;
      FAttnScoresDescLayout := FCompute.CreateStorageDescriptorSetLayout(3);
      if FErrors.HasFatal() then Exit;
      FSoftmaxDescLayout    := FCompute.CreateStorageDescriptorSetLayout(1);
      if FErrors.HasFatal() then Exit;
      FAttnValueDescLayout  := FCompute.CreateStorageDescriptorSetLayout(3);
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Pipelines — single-token decode path
      //------------------------------------------------------------------
      FMatVecBundle := FCompute.CreateComputePipelineWithPush(
        FMatVecShader, 'main', FMatVecDescLayout,
        SizeOf(TVdxMatVecF16Push));
      if FErrors.HasFatal() then Exit;
      FMatVecQ8Bundle := FCompute.CreateComputePipelineWithPushAndSpec(
        FMatVecQ8Shader, 'main', FMatVecDescLayout,
        SizeOf(TVdxMatVecF16Push), LMaxQ8Blocks);
      if FErrors.HasFatal() then Exit;
      FMatVecQ4Bundle := FCompute.CreateComputePipelineWithPush(
        FMatVecQ4Shader, 'main', FMatVecDescLayout,
        SizeOf(TVdxMatVecF16Push));
      if FErrors.HasFatal() then Exit;
      FQKNormBundle := FCompute.CreateComputePipelineWithPush(
        FQKNormShader, 'main', FQKNormDescLayout,
        SizeOf(TVdxQKNormPush));
      if FErrors.HasFatal() then Exit;
      FRoPEBundle := FCompute.CreateComputePipelineWithPush(
        FRoPEShader, 'main', FRoPEDescLayout,
        SizeOf(TVdxRoPEPush));
      if FErrors.HasFatal() then Exit;
      FAttnScoresMHBundle := FCompute.CreateComputePipelineWithPush(
        FAttnScoresMHShader, 'main', FAttnScoresDescLayout,
        SizeOf(TVdxAttnScoresMHPush));
      if FErrors.HasFatal() then Exit;
      FSoftmaxMHBundle := FCompute.CreateComputePipelineWithPush(
        FSoftmaxMHShader, 'main', FSoftmaxDescLayout,
        SizeOf(TVdxSoftmaxMHPush));
      if FErrors.HasFatal() then Exit;
      FAttnValueMHBundle := FCompute.CreateComputePipelineWithPush(
        FAttnValueMHShader, 'main', FAttnValueDescLayout,
        SizeOf(TVdxAttnValueMHPush));
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Scratch buffers (allocated before desc pool so sets can bind
      // real handles — though per-dispatch UpdateDescriptorSetBuffers
      // rebinds them anyway)
      //------------------------------------------------------------------
      FQBuf := FCompute.CreateGpuBuffer(
        UInt64(FNumQHeads) * FHeadDim * SizeOf(Single),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vcBuffer);
      if FErrors.HasFatal() then Exit;

      FKBuf := FCompute.CreateGpuBuffer(
        UInt64(FNumKVHeads) * FHeadDim * SizeOf(Single),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vcBuffer);
      if FErrors.HasFatal() then Exit;

      FVBuf := FCompute.CreateGpuBuffer(
        UInt64(FNumKVHeads) * FHeadDim * SizeOf(Single),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vcBuffer);
      if FErrors.HasFatal() then Exit;

      FScoresBuf := FCompute.CreateGpuBuffer(
        UInt64(FNumQHeads) * FMaxSeqLen * SizeOf(Single),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vcBuffer);
      if FErrors.HasFatal() then Exit;

      FAttnOutBuf := FCompute.CreateGpuBuffer(
        UInt64(FNumQHeads) * FHeadDim * SizeOf(Single),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vcBuffer);
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Pre-allocated descriptor pool + 6 reusable sets (single-token)
      // Total descriptors: 3+2+1+3+1+3 = 13 storage buffers across 6 sets
      //------------------------------------------------------------------
      FDescPool := FCompute.CreateDescriptorPoolForStorage(6, 13);
      if FErrors.HasFatal() then Exit;

      FMatVecDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FDescPool, FMatVecDescLayout, [LDummyBuf, LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;
      FQKNormDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FDescPool, FQKNormDescLayout, [LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;
      FRoPEDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FDescPool, FRoPEDescLayout, [LDummyBuf]);
      if FErrors.HasFatal() then Exit;
      FScoresDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FDescPool, FAttnScoresDescLayout, [LDummyBuf, LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;
      FSoftmaxDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FDescPool, FSoftmaxDescLayout, [LDummyBuf]);
      if FErrors.HasFatal() then Exit;
      FValueDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FDescPool, FAttnValueDescLayout, [LDummyBuf, LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // TQ3 compressed KV caches (per layer, permanent residents)
      // Layout: [NumKVHeads * MaxSeqLen * BlocksPerHead * 4] uint32
      // BlocksPerHead = HeadDim / 32, each block = 4 uint32 = 16 bytes
      //------------------------------------------------------------------
      LCacheSize := UInt64(FNumKVHeads) * FMaxSeqLen *
        (FHeadDim div 32) * 16;
      SetLength(FKCacheTQ3, FNumLayers);
      SetLength(FVCacheTQ3, FNumLayers);
      for LI := 0 to Integer(FNumLayers) - 1 do
      begin
        FKCacheTQ3[LI] := FCompute.CreateGpuBuffer(
          LCacheSize,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          vcCache);
        if FErrors.HasFatal() then Exit;
        FVCacheTQ3[LI] := FCompute.CreateGpuBuffer(
          LCacheSize,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          vcCache);
        if FErrors.HasFatal() then Exit;
      end;

      //------------------------------------------------------------------
      // Shared F32 decode buffers (one pair reused across all layers)
      // Layout: [NumKVHeads * MaxSeqLen * HeadDim] F32
      //------------------------------------------------------------------
      LCacheSize := UInt64(FNumKVHeads) * FMaxSeqLen * FHeadDim *
        SizeOf(Single);
      FKDecodeF32 := FCompute.CreateGpuBuffer(
        LCacheSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vcBuffer);
      if FErrors.HasFatal() then Exit;
      FVDecodeF32 := FCompute.CreateGpuBuffer(
        LCacheSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vcBuffer);
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // KV cache store shader (single-token) — 4 bindings
      //------------------------------------------------------------------
      FKVStoreShader := LoadShader('KV_CACHE_STORE');
      if FErrors.HasFatal() then Exit;
      FKVStoreDescLayout := FCompute.CreateStorageDescriptorSetLayout(4);
      if FErrors.HasFatal() then Exit;
      FKVStoreBundle := FCompute.CreateComputePipelineWithPush(
        FKVStoreShader, 'main', FKVStoreDescLayout,
        SizeOf(TVdxKVCacheStorePush));
      if FErrors.HasFatal() then Exit;
      FKVStoreDescPool := FCompute.CreateDescriptorPoolForStorage(1, 4);
      if FErrors.HasFatal() then Exit;
      FKVStoreDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FKVStoreDescPool, FKVStoreDescLayout,
        [LDummyBuf, LDummyBuf, LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Batch matmul shaders + pipelines (reuse FMatVecDescLayout)
      //------------------------------------------------------------------
      FMatMulF16Shader := LoadShader('MATMUL_F16');
      if FErrors.HasFatal() then Exit;
      FMatMulQ8Shader  := LoadShader('MATMUL_Q8_0');
      if FErrors.HasFatal() then Exit;
      FMatMulQ4Shader  := LoadShader('MATMUL_Q4_0');
      if FErrors.HasFatal() then Exit;
      FMatMulF16Bundle := FCompute.CreateComputePipelineWithPush(
        FMatMulF16Shader, 'main', FMatVecDescLayout,
        SizeOf(TVdxMatMulPush));
      if FErrors.HasFatal() then Exit;
      FMatMulQ8Bundle := FCompute.CreateComputePipelineWithPushAndSpec(
        FMatMulQ8Shader, 'main', FMatVecDescLayout,
        SizeOf(TVdxMatMulPush), LMaxQ8Blocks);
      if FErrors.HasFatal() then Exit;
      FMatMulQ4Bundle := FCompute.CreateComputePipelineWithPush(
        FMatMulQ4Shader, 'main', FMatVecDescLayout,
        SizeOf(TVdxMatMulPush));
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Prefill attention shaders + pipelines
      //------------------------------------------------------------------
      FRoPEBatchShader          := LoadShader('ROPE_BATCH');
      if FErrors.HasFatal() then Exit;
      FKVStoreBatchShader       := LoadShader('KV_CACHE_STORE_BATCH');
      if FErrors.HasFatal() then Exit;
      FScoresPrefillShader      := LoadShader('ATTN_SCORES_PREFILL');
      if FErrors.HasFatal() then Exit;
      FScoresPrefillBidirShader := LoadShader('ATTN_SCORES_PREFILL_BIDIR');
      if FErrors.HasFatal() then Exit;
      FSoftmaxPrefillShader     := LoadShader('SOFTMAX_PREFILL');
      if FErrors.HasFatal() then Exit;
      FValuePrefillShader       := LoadShader('ATTN_VALUE_PREFILL');
      if FErrors.HasFatal() then Exit;

      FRoPEBatchBundle := FCompute.CreateComputePipelineWithPush(
        FRoPEBatchShader, 'main', FRoPEDescLayout,
        SizeOf(TVdxRoPEBatchPush));
      if FErrors.HasFatal() then Exit;
      FKVStoreBatchBundle := FCompute.CreateComputePipelineWithPush(
        FKVStoreBatchShader, 'main', FKVStoreDescLayout,
        SizeOf(TVdxKVCacheStoreBatchPush));
      if FErrors.HasFatal() then Exit;
      FScoresPrefillBundle := FCompute.CreateComputePipelineWithPush(
        FScoresPrefillShader, 'main', FAttnScoresDescLayout,
        SizeOf(TVdxAttnScoresPrefillPush));
      if FErrors.HasFatal() then Exit;
      // Bidir variant uses identical bindings + push struct — same
      // shader with the causal mask block removed, dispatch-compatible.
      FScoresPrefillBidirBundle := FCompute.CreateComputePipelineWithPush(
        FScoresPrefillBidirShader, 'main', FAttnScoresDescLayout,
        SizeOf(TVdxAttnScoresPrefillPush));
      if FErrors.HasFatal() then Exit;
      FSoftmaxPrefillBundle := FCompute.CreateComputePipelineWithPush(
        FSoftmaxPrefillShader, 'main', FSoftmaxDescLayout,
        SizeOf(TVdxSoftmaxPrefillPush));
      if FErrors.HasFatal() then Exit;
      FValuePrefillBundle := FCompute.CreateComputePipelineWithPush(
        FValuePrefillShader, 'main', FAttnValueDescLayout,
        SizeOf(TVdxAttnValuePrefillPush));
      if FErrors.HasFatal() then Exit;

      // Prefill descriptor pool: 5 sets, total bindings = 1+4+3+1+3 = 12
      FPrefillDescPool := FCompute.CreateDescriptorPoolForStorage(5, 12);
      if FErrors.HasFatal() then Exit;
      FPrefillRoPEDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FPrefillDescPool, FRoPEDescLayout, [LDummyBuf]);
      if FErrors.HasFatal() then Exit;
      FPrefillKVStoreDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FPrefillDescPool, FKVStoreDescLayout,
        [LDummyBuf, LDummyBuf, LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;
      FPrefillScoresDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FPrefillDescPool, FAttnScoresDescLayout,
        [LDummyBuf, LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;
      FPrefillSoftmaxDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FPrefillDescPool, FSoftmaxDescLayout, [LDummyBuf]);
      if FErrors.HasFatal() then Exit;
      FPrefillValueDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FPrefillDescPool, FAttnValueDescLayout,
        [LDummyBuf, LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;

      // Pre-allocated prefill scores buffer [NumQHeads x MaxSeq x MaxSeq] F32
      FPrefillScoresBuf := FCompute.CreateGpuBuffer(
        UInt64(FNumQHeads) * FMaxSeqLen * FMaxSeqLen * SizeOf(Single),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vcBuffer);
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // TQ3 KV cache compression pipelines
      //------------------------------------------------------------------
      FTQ3KVQuantShader   := LoadShader('TQ3_KV_QUANTIZE');
      if FErrors.HasFatal() then Exit;
      FTQ3KVDequantShader := LoadShader('TQ3_KV_DEQUANTIZE');
      if FErrors.HasFatal() then Exit;
      FTQ3KVDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
      if FErrors.HasFatal() then Exit;
      FTQ3KVQuantBundle := FCompute.CreateComputePipelineWithPush(
        FTQ3KVQuantShader, 'main', FTQ3KVDescLayout,
        SizeOf(TVdxTQ3KVQuantPush));
      if FErrors.HasFatal() then Exit;
      FTQ3KVDequantBundle := FCompute.CreateComputePipelineWithPush(
        FTQ3KVDequantShader, 'main', FTQ3KVDescLayout,
        SizeOf(TVdxTQ3KVDequantPush));
      if FErrors.HasFatal() then Exit;

      // Fused KV store + TQ3 quantize (batch path) — 3 bindings
      FKVStoreBatchTQ3Shader := LoadShader('KV_CACHE_STORE_BATCH_TQ3');
      if FErrors.HasFatal() then Exit;
      FKVStoreBatchTQ3DescLayout :=
        FCompute.CreateStorageDescriptorSetLayout(3);
      if FErrors.HasFatal() then Exit;
      FKVStoreBatchTQ3Bundle := FCompute.CreateComputePipelineWithPush(
        FKVStoreBatchTQ3Shader, 'main', FKVStoreBatchTQ3DescLayout,
        SizeOf(TVdxKVStoreBatchTQ3Push));
      if FErrors.HasFatal() then Exit;
      FKVStoreBatchTQ3DescPool :=
        FCompute.CreateDescriptorPoolForStorage(1, 3);
      if FErrors.HasFatal() then Exit;
      FKVStoreBatchTQ3DescSet := FCompute.AllocateDescriptorSetForBuffers(
        FKVStoreBatchTQ3DescPool, FKVStoreBatchTQ3DescLayout,
        [LDummyBuf, LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;

      // TQ3 KV descriptor pool: 2 sets (quant + dequant), 2 bindings each
      FTQ3KVDescPool := FCompute.CreateDescriptorPoolForStorage(2, 4);
      if FErrors.HasFatal() then Exit;
      FTQ3KVQuantDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FTQ3KVDescPool, FTQ3KVDescLayout, [LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;
      FTQ3KVDequantDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FTQ3KVDescPool, FTQ3KVDescLayout, [LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Fused TQ3 attention scores — reuses FAttnScoresDescLayout and
      // FScoresDescSet at dispatch time.
      //------------------------------------------------------------------
      FAttnScoresMHTQ3Shader := LoadShader('ATTN_SCORES_MH_TQ3');
      if FErrors.HasFatal() then Exit;
      FAttnScoresMHTQ3Bundle := FCompute.CreateComputePipelineWithPush(
        FAttnScoresMHTQ3Shader, 'main', FAttnScoresDescLayout,
        SizeOf(TVdxAttnScoresMHTQ3Push));
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Grow the streaming staging pool on TVdxCompute to fit the
      // largest Q/K/V/O projection slice this layer might ever upload.
      // Four pairs because all four projections are in flight within
      // one batch before submission — each needs its own host memory
      // to prevent the CPU from overwriting bytes the GPU hasn't yet
      // copied. Sized for F16 upper bound — Q8_0 / Q4_0 slices are
      // smaller and fit without regrowth. Future consumers (Phase 8
      // FFN) call EnsureStagingPool with their own count + bytes; the
      // pool grows to max across consumers.
      //   Q slice: HiddenDim x (NumQHeads * HeadDim)
      //   K / V   : HiddenDim x (NumKVHeads * HeadDim)
      //   O slice: (NumQHeads * HeadDim) x HiddenDim  (same bytes as Q)
      //------------------------------------------------------------------
      LQDim  := UInt64(FHiddenDim) * (UInt64(FNumQHeads) * FHeadDim);
      LKVDim := UInt64(FHiddenDim) * (UInt64(FNumKVHeads) * FHeadDim);
      if LQDim >= LKVDim then
        LMaxSliceF16 := LQDim * SizeOf(Word)
      else
        LMaxSliceF16 := LKVDim * SizeOf(Word);
      if not FCompute.EnsureStagingPool(4, LMaxSliceF16) then Exit;

      FInitialized := True;
      Result       := True;
      Status('Attention: Ready (staging=4 pairs x %d bytes)',
        [LMaxSliceF16]);
    except
      on E: Exception do
        FErrors.Add(esFatal, VDX_ERROR_ATTN_INIT_EXCEPTION,
          RSAttnInitException, [E.Message]);
    end;
  finally
    if not Result then
      Cleanup();
  end;
end;

function TVdxAttention.ResolveAttnWeights(const AReader: TVdxGGUFReader;
  const ALayerIndex: Integer;
  out AWeights: TVdxAttnLayerWeights): Boolean;

  // Resolve one Q/K/V/O tensor. Returns False with FErrors populated
  // on any failure (tensor missing, unsupported type, empty slice).
  // Outputs the mmap pointer and the computed byte size.
  function ResolveOne(const ATensorName: string;
    out APtr: PByte; out ABytes: UInt64;
    out AType: TVdxGGMLType): Boolean;
  var
    LInfo: TVdxGGUFTensorInfo;
  begin
    Result := False;
    APtr   := nil;
    ABytes := 0;
    AType  := Low(TVdxGGMLType);

    if not AReader.HasTensor(ATensorName) then
    begin
      FErrors.Add(esFatal, VDX_ERROR_ATTN_TENSOR_NOT_FOUND,
        RSAttnTensorNotFound, [ATensorName]);
      Exit;
    end;

    if not AReader.GetTensorInfo(ATensorName, LInfo) then Exit;

    AType  := LInfo.TensorType;
    ABytes := VdxGGMLTensorBytes(LInfo.TensorType,
      LInfo.Dimensions[0], LInfo.Dimensions[1]);
    if ABytes = 0 then
    begin
      FErrors.Add(esFatal, VDX_ERROR_ATTN_UNSUPPORTED_TYPE,
        RSAttnUnsupportedType,
        [ATensorName, VdxGGMLTypeName(LInfo.TensorType)]);
      Exit;
    end;

    APtr := AReader.GetTensorDataPtr(ATensorName);
    if APtr = nil then Exit;  // GGUFReader already logged the reason

    Result := True;
  end;

var
  LTypeQ, LTypeK, LTypeV, LTypeO: TVdxGGMLType;
begin
  Result   := False;
  AWeights := Default(TVdxAttnLayerWeights);

  if not FInitialized then
  begin
    FErrors.Add(esError, VDX_ERROR_ATTN_NOT_INIT, RSAttnNotInit);
    Exit;
  end;

  if AReader = nil then
  begin
    FErrors.Add(esFatal, VDX_ERROR_ATTN_RESOLVE_EXCEPTION,
      RSAttnResolveException, ['AReader is nil']);
    Exit;
  end;

  try
    if not ResolveOne(
      Format('blk.%d.attn_q.weight', [ALayerIndex]),
      AWeights.QWeightPtr, AWeights.QWeightBytes, LTypeQ) then Exit;
    if not ResolveOne(
      Format('blk.%d.attn_k.weight', [ALayerIndex]),
      AWeights.KWeightPtr, AWeights.KWeightBytes, LTypeK) then Exit;
    if not ResolveOne(
      Format('blk.%d.attn_v.weight', [ALayerIndex]),
      AWeights.VWeightPtr, AWeights.VWeightBytes, LTypeV) then Exit;
    if not ResolveOne(
      Format('blk.%d.attn_output.weight', [ALayerIndex]),
      AWeights.OWeightPtr, AWeights.OWeightBytes, LTypeO) then Exit;

    // All four projections share a single quant type in every model
    // we support. If a future model mixes quants across projections
    // within one layer, this check surfaces it loudly instead of
    // silently picking one.
    if (LTypeQ <> LTypeK) or (LTypeQ <> LTypeV) or (LTypeQ <> LTypeO) then
    begin
      FErrors.Add(esFatal, VDX_ERROR_ATTN_UNSUPPORTED_TYPE,
        RSAttnUnsupportedType,
        [Format('layer %d mixed quant types', [ALayerIndex]),
         Format('Q=%s K=%s V=%s O=%s',
           [VdxGGMLTypeName(LTypeQ), VdxGGMLTypeName(LTypeK),
            VdxGGMLTypeName(LTypeV), VdxGGMLTypeName(LTypeO)])]);
      Exit;
    end;

    AWeights.WeightType := LTypeQ;
    Result := True;
  except
    on E: Exception do
    begin
      FErrors.Add(esFatal, VDX_ERROR_ATTN_RESOLVE_EXCEPTION,
        RSAttnResolveException, [E.Message]);
      AWeights := Default(TVdxAttnLayerWeights);
    end;
  end;
end;

procedure TVdxAttention.DispatchMatVec(const AWeightBuf: TVdxGpuBuffer;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ATensorType: TVdxGGMLType);
var
  LPush:           TVdxMatVecF16Push;
  LPipeline:       VkPipeline;
  LPipelineLayout: VkPipelineLayout;
begin
  // Rebind buffers to the pre-allocated descriptor set (no pool churn)
  FCompute.UpdateDescriptorSetBuffers(FMatVecDescSet,
    [AWeightBuf, AInputBuf, AOutputBuf]);

  // Pipeline selection by quant: Q4_0 / Q8_0 use full in_dim, F16
  // uses in_dim/2 (two halves packed per uint32 in shader load).
  if ATensorType = gtQ4_0 then
  begin
    LPush.InDimHalf := AInDim;
    LPipeline       := FMatVecQ4Bundle.Pipeline;
    LPipelineLayout := FMatVecQ4Bundle.PipelineLayout;
  end
  else if ATensorType = gtQ8_0 then
  begin
    LPush.InDimHalf := AInDim;
    LPipeline       := FMatVecQ8Bundle.Pipeline;
    LPipelineLayout := FMatVecQ8Bundle.PipelineLayout;
  end
  else
  begin
    LPush.InDimHalf := AInDim div 2;
    LPipeline       := FMatVecBundle.Pipeline;
    LPipelineLayout := FMatVecBundle.PipelineLayout;
  end;

  LPush.OutDim := AOutDim;

  // One workgroup (256 threads) per output row.
  FCompute.DispatchComputeWithPush(
    LPipeline, LPipelineLayout, FMatVecDescSet,
    @LPush, SizeOf(LPush), AOutDim);
end;

procedure TVdxAttention.StreamAndDispatchMatVec(const AStagingIndex: UInt32;
  const AWeightPtr: PByte; const AWeightBytes: UInt64;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ATensorType: TVdxGGMLType);
var
  LHost:   TVdxGpuBuffer;
  LDevice: TVdxGpuBuffer;
begin
  // The indexed staging pair is the key to correct batched streaming:
  // each concurrent in-flight slice (Q at 0, K at 1, V at 2, O at 3)
  // owns its own host memory, so the CPU memcpy for a later slice
  // never overwrites bytes the GPU will later copy from an earlier
  // slice when the batch submits.
  LHost   := FCompute.GetStagingHost(AStagingIndex);
  LDevice := FCompute.GetStagingDevice(AStagingIndex);

  // Step 1 — CPU memcpy from mmap region into host-visible staging.
  // The OS page-faults in weight pages on demand via the mmap; this
  // is the disk -> RAM hop.
  FCompute.UploadToBuffer(LHost, AWeightPtr, AWeightBytes);

  // Step 2 — GPU copy host -> device-local (batch-recorded).
  FCompute.CopyBuffer(LHost, LDevice, AWeightBytes);

  // Step 3 — barrier so the matvec dispatch waits for the copy.
  FCompute.BatchBarrier();

  // Step 4 — matvec against the device-local staging buffer.
  DispatchMatVec(LDevice, AInputBuf, AOutputBuf, AInDim, AOutDim, ATensorType);
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
  LSeqLen:         UInt32;
  LQKNormPush:     TVdxQKNormPush;
  LRoPEPush:       TVdxRoPEPush;
  LKVStorePush:    TVdxKVCacheStorePush;
  LTQ3QuantPush:   TVdxTQ3KVQuantPush;
  LTQ3DequantPush: TVdxTQ3KVDequantPush;
  LScoresTQ3Push:  TVdxAttnScoresMHTQ3Push;
  LSoftmaxMHPush:  TVdxSoftmaxMHPush;
  LValueMHPush:    TVdxAttnValueMHPush;
begin
  LSeqLen := UInt32(APosition) + 1;

  //----------------------------------------------------------------
  // Step 1 — Q / K / V projections, streamed through pairs 0/1/2.
  // All three read the same AInputBuf and write separate output
  // buffers, so they can be issued without cross-barriers here —
  // the barrier after V gates the entire block for QK-norm.
  //----------------------------------------------------------------
  StreamAndDispatchMatVec(0,
    AWeights.QWeightPtr, AWeights.QWeightBytes,
    AInputBuf, FQBuf,
    FHiddenDim, FNumQHeads * FHeadDim, AWeights.WeightType);
  StreamAndDispatchMatVec(1,
    AWeights.KWeightPtr, AWeights.KWeightBytes,
    AInputBuf, FKBuf,
    FHiddenDim, FNumKVHeads * FHeadDim, AWeights.WeightType);
  StreamAndDispatchMatVec(2,
    AWeights.VWeightPtr, AWeights.VWeightBytes,
    AInputBuf, FVBuf,
    FHiddenDim, FNumKVHeads * FHeadDim, AWeights.WeightType);
  FCompute.BatchBarrier();  // Q/K/V buffers ready for QK-norm

  //----------------------------------------------------------------
  // Step 2 — QK-norm on Q (NumQHeads heads) and K (NumKVHeads heads)
  //----------------------------------------------------------------
  LQKNormPush.HeadDim := FHeadDim;
  LQKNormPush.Eps     := 1e-6;

  LQKNormPush.NumHeads := FNumQHeads;
  FCompute.UpdateDescriptorSetBuffers(FQKNormDescSet, [FQBuf, AQNormBuf]);
  FCompute.DispatchComputeWithPush(
    FQKNormBundle.Pipeline, FQKNormBundle.PipelineLayout,
    FQKNormDescSet, @LQKNormPush, SizeOf(LQKNormPush), FNumQHeads);

  LQKNormPush.NumHeads := FNumKVHeads;
  FCompute.UpdateDescriptorSetBuffers(FQKNormDescSet, [FKBuf, AKNormBuf]);
  FCompute.DispatchComputeWithPush(
    FQKNormBundle.Pipeline, FQKNormBundle.PipelineLayout,
    FQKNormDescSet, @LQKNormPush, SizeOf(LQKNormPush), FNumKVHeads);
  FCompute.BatchBarrier();  // Q/K normed, ready for RoPE

  //----------------------------------------------------------------
  // Step 3 — RoPE on Q and K
  //----------------------------------------------------------------
  LRoPEPush.HeadDim   := FHeadDim;
  LRoPEPush.Position  := UInt32(APosition);
  LRoPEPush.ThetaBase := AThetaBase;

  LRoPEPush.NumHeads := FNumQHeads;
  FCompute.UpdateDescriptorSetBuffers(FRoPEDescSet, [FQBuf]);
  FCompute.DispatchComputeWithPush(
    FRoPEBundle.Pipeline, FRoPEBundle.PipelineLayout,
    FRoPEDescSet, @LRoPEPush, SizeOf(LRoPEPush), FNumQHeads);

  LRoPEPush.NumHeads := FNumKVHeads;
  FCompute.UpdateDescriptorSetBuffers(FRoPEDescSet, [FKBuf]);
  FCompute.DispatchComputeWithPush(
    FRoPEBundle.Pipeline, FRoPEBundle.PipelineLayout,
    FRoPEDescSet, @LRoPEPush, SizeOf(LRoPEPush), FNumKVHeads);
  FCompute.BatchBarrier();  // Q/K with RoPE applied

  //----------------------------------------------------------------
  // Step 4a — Store K and V for this position into shared decode
  // buffers (one dispatch, replaces per-head CopyBufferRegion).
  //----------------------------------------------------------------
  LKVStorePush.HeadDim    := FHeadDim;
  LKVStorePush.MaxSeq     := FMaxSeqLen;
  LKVStorePush.Position   := UInt32(APosition);
  LKVStorePush.NumKVHeads := FNumKVHeads;
  FCompute.UpdateDescriptorSetBuffers(FKVStoreDescSet,
    [FKBuf, FVBuf, FKDecodeF32, FVDecodeF32]);
  FCompute.DispatchComputeWithPush(
    FKVStoreBundle.Pipeline, FKVStoreBundle.PipelineLayout,
    FKVStoreDescSet, @LKVStorePush, SizeOf(LKVStorePush), FNumKVHeads);
  FCompute.BatchBarrier();  // Decode buffers hold new position

  //----------------------------------------------------------------
  // Step 4b — Quantize current position decode -> per-layer TQ3
  // cache (K and V separately).
  //----------------------------------------------------------------
  LTQ3QuantPush.BlocksPerHead := FHeadDim div 32;
  LTQ3QuantPush.MaxSeq        := FMaxSeqLen;
  LTQ3QuantPush.Position      := UInt32(APosition);
  LTQ3QuantPush.NumHeads      := FNumKVHeads;

  FCompute.UpdateDescriptorSetBuffers(FTQ3KVQuantDescSet,
    [FKDecodeF32, FKCacheTQ3[ALayerIndex]]);
  FCompute.DispatchComputeWithPush(
    FTQ3KVQuantBundle.Pipeline, FTQ3KVQuantBundle.PipelineLayout,
    FTQ3KVQuantDescSet, @LTQ3QuantPush, SizeOf(LTQ3QuantPush),
    (FHeadDim div 32) * FNumKVHeads);

  FCompute.UpdateDescriptorSetBuffers(FTQ3KVQuantDescSet,
    [FVDecodeF32, FVCacheTQ3[ALayerIndex]]);
  FCompute.DispatchComputeWithPush(
    FTQ3KVQuantBundle.Pipeline, FTQ3KVQuantBundle.PipelineLayout,
    FTQ3KVQuantDescSet, @LTQ3QuantPush, SizeOf(LTQ3QuantPush),
    (FHeadDim div 32) * FNumKVHeads);
  FCompute.BatchBarrier();  // TQ3 caches updated

  //----------------------------------------------------------------
  // Step 4c — Dequantize V for the whole sequence so attention-value
  // can read F32. K is read directly from the TQ3 cache by the fused
  // TQ3 scores kernel — no K dequant needed.
  //----------------------------------------------------------------
  LTQ3DequantPush.BlocksPerHead := FHeadDim div 32;
  LTQ3DequantPush.MaxSeq        := FMaxSeqLen;
  LTQ3DequantPush.SeqLen        := LSeqLen;
  LTQ3DequantPush.NumHeads      := FNumKVHeads;

  FCompute.UpdateDescriptorSetBuffers(FTQ3KVDequantDescSet,
    [FVCacheTQ3[ALayerIndex], FVDecodeF32]);
  FCompute.DispatchComputeWithPush(
    FTQ3KVDequantBundle.Pipeline, FTQ3KVDequantBundle.PipelineLayout,
    FTQ3KVDequantDescSet, @LTQ3DequantPush, SizeOf(LTQ3DequantPush),
    (FHeadDim div 32) * FNumKVHeads * LSeqLen);
  FCompute.BatchBarrier();  // V decode has all positions

  //----------------------------------------------------------------
  // Step 5a — Fused TQ3 attention scores: reads Q + TQ3 K cache
  // directly, applies WHT on the fly, dots against packed centroids.
  //----------------------------------------------------------------
  LScoresTQ3Push.HeadDim       := FHeadDim;
  LScoresTQ3Push.SeqLen        := LSeqLen;
  LScoresTQ3Push.MaxSeq        := FMaxSeqLen;
  LScoresTQ3Push.Scale         := 1.0 / Sqrt(Single(FHeadDim));
  LScoresTQ3Push.NumQHeads     := FNumQHeads;
  LScoresTQ3Push.GqaRatio      := FNumQHeads div FNumKVHeads;
  LScoresTQ3Push.BlocksPerHead := FHeadDim div 32;

  FCompute.UpdateDescriptorSetBuffers(FScoresDescSet,
    [FQBuf, FKCacheTQ3[ALayerIndex], FScoresBuf]);
  FCompute.DispatchComputeWithPush(
    FAttnScoresMHTQ3Bundle.Pipeline, FAttnScoresMHTQ3Bundle.PipelineLayout,
    FScoresDescSet, @LScoresTQ3Push, SizeOf(LScoresTQ3Push),
    (LSeqLen + 255) div 256, FNumQHeads);
  FCompute.BatchBarrier();  // Scores ready for softmax

  //----------------------------------------------------------------
  // Step 5b — softmax over every head in one dispatch
  //----------------------------------------------------------------
  LSoftmaxMHPush.SeqLen    := LSeqLen;
  LSoftmaxMHPush.MaxSeq    := FMaxSeqLen;
  LSoftmaxMHPush.NumQHeads := FNumQHeads;
  FCompute.UpdateDescriptorSetBuffers(FSoftmaxDescSet, [FScoresBuf]);
  FCompute.DispatchComputeWithPush(
    FSoftmaxMHBundle.Pipeline, FSoftmaxMHBundle.PipelineLayout,
    FSoftmaxDescSet, @LSoftmaxMHPush, SizeOf(LSoftmaxMHPush), FNumQHeads);
  FCompute.BatchBarrier();  // Softmax weights ready for value

  //----------------------------------------------------------------
  // Step 5c — weighted V sum in one 2D dispatch
  //----------------------------------------------------------------
  LValueMHPush.HeadDim   := FHeadDim;
  LValueMHPush.SeqLen    := LSeqLen;
  LValueMHPush.MaxSeq    := FMaxSeqLen;
  LValueMHPush.NumQHeads := FNumQHeads;
  LValueMHPush.GqaRatio  := FNumQHeads div FNumKVHeads;

  FCompute.UpdateDescriptorSetBuffers(FValueDescSet,
    [FScoresBuf, FVDecodeF32, FAttnOutBuf]);
  FCompute.DispatchComputeWithPush(
    FAttnValueMHBundle.Pipeline, FAttnValueMHBundle.PipelineLayout,
    FValueDescSet, @LValueMHPush, SizeOf(LValueMHPush),
    (FHeadDim + 255) div 256, FNumQHeads);
  FCompute.BatchBarrier();  // AttnOut ready for O projection

  //----------------------------------------------------------------
  // Step 6 — Output projection, streamed through pair 3.
  // Weight shape: [(NumQHeads*HeadDim) x HiddenDim], so in_dim is
  // the attention-out width and out_dim is the residual width.
  //----------------------------------------------------------------
  StreamAndDispatchMatVec(3,
    AWeights.OWeightPtr, AWeights.OWeightBytes,
    FAttnOutBuf, AOutputBuf,
    FNumQHeads * FHeadDim, FHiddenDim, AWeights.WeightType);
  FCompute.BatchBarrier();  // AOutputBuf ready for caller
end;

procedure TVdxAttention.DispatchBatchMatMul(const AWeightBuf: TVdxGpuBuffer;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ANumTokens: UInt32; const ATensorType: TVdxGGMLType);
var
  LPush:           TVdxMatMulPush;
  LPipeline:       VkPipeline;
  LPipelineLayout: VkPipelineLayout;
begin
  // Reuse FMatVecDescSet — same 3-binding layout (weight, input, output).
  FCompute.UpdateDescriptorSetBuffers(FMatVecDescSet,
    [AWeightBuf, AInputBuf, AOutputBuf]);

  if ATensorType = gtQ4_0 then
  begin
    LPush.InDimParam := AInDim;
    LPipeline        := FMatMulQ4Bundle.Pipeline;
    LPipelineLayout  := FMatMulQ4Bundle.PipelineLayout;
  end
  else if ATensorType = gtQ8_0 then
  begin
    LPush.InDimParam := AInDim;
    LPipeline        := FMatMulQ8Bundle.Pipeline;
    LPipelineLayout  := FMatMulQ8Bundle.PipelineLayout;
  end
  else
  begin
    LPush.InDimParam := AInDim div 2;
    LPipeline        := FMatMulF16Bundle.Pipeline;
    LPipelineLayout  := FMatMulF16Bundle.PipelineLayout;
  end;

  LPush.OutDim    := AOutDim;
  LPush.NumTokens := ANumTokens;

  // 2D dispatch: X = output rows (OutDim), Y = token count.
  FCompute.DispatchComputeWithPush(
    LPipeline, LPipelineLayout, FMatVecDescSet,
    @LPush, SizeOf(LPush), AOutDim, ANumTokens);
end;

procedure TVdxAttention.StreamAndDispatchBatchMatMul(
  const AStagingIndex: UInt32;
  const AWeightPtr: PByte; const AWeightBytes: UInt64;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ANumTokens: UInt32; const ATensorType: TVdxGGMLType);
var
  LHost:   TVdxGpuBuffer;
  LDevice: TVdxGpuBuffer;
begin
  // Same streaming pattern as StreamAndDispatchMatVec — per-slot host
  // memory prevents CPU writes from overwriting bytes the GPU hasn't
  // yet consumed when the batch eventually submits.
  LHost   := FCompute.GetStagingHost(AStagingIndex);
  LDevice := FCompute.GetStagingDevice(AStagingIndex);

  FCompute.UploadToBuffer(LHost, AWeightPtr, AWeightBytes);
  FCompute.CopyBuffer(LHost, LDevice, AWeightBytes);
  FCompute.BatchBarrier();
  DispatchBatchMatMul(LDevice, AInputBuf, AOutputBuf,
    AInDim, AOutDim, ANumTokens, ATensorType);
end;

procedure TVdxAttention.BatchMatMul(const AWeightBuf: TVdxGpuBuffer;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ANumTokens: UInt32;
  const ATensorType: TVdxGGMLType);
begin
  DispatchBatchMatMul(AWeightBuf, AInputBuf, AOutputBuf,
    AInDim, AOutDim, ANumTokens, ATensorType);
end;

procedure TVdxAttention.TestMatVec(const AWeightBuf: TVdxGpuBuffer;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ATensorType: TVdxGGMLType);
begin
  DispatchMatVec(AWeightBuf, AInputBuf, AOutputBuf,
    AInDim, AOutDim, ATensorType);
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
  const AAttnOutMat: TVdxGpuBuffer;
  const ABidirectional: Boolean);
var
  LSeqLen:         UInt32;
  LQKNormPush:     TVdxQKNormPush;
  LRoPEPush:       TVdxRoPEBatchPush;
  LKVStoreTQ3Push: TVdxKVStoreBatchTQ3Push;
  LTQ3DequantPush: TVdxTQ3KVDequantPush;
  LScoresPush:     TVdxAttnScoresPrefillPush;
  LSoftmaxPush:    TVdxSoftmaxPrefillPush;
  LValuePush:      TVdxAttnValuePrefillPush;
  LScoresBundle:   TVdxComputePipelineBundle;
begin
  // Total filled-cache length after this batch writes — prefill
  // shaders index up to SeqLen, not just the new tokens.
  LSeqLen := AStartPos + ANumTokens;

  //----------------------------------------------------------------
  // Step 1 — Q/K/V batch matmul projections, streamed through pairs
  // 0/1/2. Output shapes: AQMat [N x QDim], AKMat [N x KVDim],
  // AVMat [N x KVDim].
  //----------------------------------------------------------------
  StreamAndDispatchBatchMatMul(0,
    AWeights.QWeightPtr, AWeights.QWeightBytes,
    AInputMat, AQMat,
    FHiddenDim, FNumQHeads * FHeadDim, ANumTokens, AWeights.WeightType);
  StreamAndDispatchBatchMatMul(1,
    AWeights.KWeightPtr, AWeights.KWeightBytes,
    AInputMat, AKMat,
    FHiddenDim, FNumKVHeads * FHeadDim, ANumTokens, AWeights.WeightType);
  StreamAndDispatchBatchMatMul(2,
    AWeights.VWeightPtr, AWeights.VWeightBytes,
    AInputMat, AVMat,
    FHiddenDim, FNumKVHeads * FHeadDim, ANumTokens, AWeights.WeightType);
  FCompute.BatchBarrier();  // Q/K/V matrices ready for norm

  //----------------------------------------------------------------
  // Step 2 — QK-norm. QMat flattens to [N*NumQHeads x HeadDim] which
  // reuses the single-token QKNorm shader with the token count baked
  // into NumHeads.
  //----------------------------------------------------------------
  LQKNormPush.HeadDim := FHeadDim;
  LQKNormPush.Eps     := 1e-6;

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
  FCompute.BatchBarrier();  // Q/K normed

  //----------------------------------------------------------------
  // Step 3 — Batched RoPE. Per-token positions run from AStartPos
  // to AStartPos + ANumTokens - 1; the shader derives each token's
  // absolute position from its Y index + StartPos.
  //----------------------------------------------------------------
  LRoPEPush.HeadDim   := FHeadDim;
  LRoPEPush.ThetaBase := AThetaBase;
  LRoPEPush.StartPos  := AStartPos;
  LRoPEPush.NumTokens := ANumTokens;

  LRoPEPush.NumHeads := FNumQHeads;
  FCompute.UpdateDescriptorSetBuffers(FPrefillRoPEDescSet, [AQMat]);
  FCompute.DispatchComputeWithPush(
    FRoPEBatchBundle.Pipeline, FRoPEBatchBundle.PipelineLayout,
    FPrefillRoPEDescSet, @LRoPEPush, SizeOf(LRoPEPush),
    FNumQHeads, ANumTokens);

  LRoPEPush.NumHeads := FNumKVHeads;
  FCompute.UpdateDescriptorSetBuffers(FPrefillRoPEDescSet, [AKMat]);
  FCompute.DispatchComputeWithPush(
    FRoPEBatchBundle.Pipeline, FRoPEBatchBundle.PipelineLayout,
    FPrefillRoPEDescSet, @LRoPEPush, SizeOf(LRoPEPush),
    FNumKVHeads, ANumTokens);
  FCompute.BatchBarrier();  // Q/K with RoPE applied

  //----------------------------------------------------------------
  // Step 4 — Fused KV store + TQ3 quantize. Writes both the F32
  // decode buffer (for subsequent prefill attention reads) and the
  // per-layer TQ3 cache (for future decode-phase reads) in one
  // dispatch per stream.
  //----------------------------------------------------------------
  LKVStoreTQ3Push.HeadDim   := FHeadDim;
  LKVStoreTQ3Push.MaxSeq    := FMaxSeqLen;
  LKVStoreTQ3Push.NumHeads  := FNumKVHeads;
  LKVStoreTQ3Push.NumTokens := ANumTokens;
  LKVStoreTQ3Push.StartPos  := AStartPos;

  FCompute.UpdateDescriptorSetBuffers(FKVStoreBatchTQ3DescSet,
    [AKMat, FKDecodeF32, FKCacheTQ3[ALayerIndex]]);
  FCompute.DispatchComputeWithPush(
    FKVStoreBatchTQ3Bundle.Pipeline,
    FKVStoreBatchTQ3Bundle.PipelineLayout,
    FKVStoreBatchTQ3DescSet, @LKVStoreTQ3Push, SizeOf(LKVStoreTQ3Push),
    (FHeadDim div 32) * FNumKVHeads, ANumTokens);

  FCompute.UpdateDescriptorSetBuffers(FKVStoreBatchTQ3DescSet,
    [AVMat, FVDecodeF32, FVCacheTQ3[ALayerIndex]]);
  FCompute.DispatchComputeWithPush(
    FKVStoreBatchTQ3Bundle.Pipeline,
    FKVStoreBatchTQ3Bundle.PipelineLayout,
    FKVStoreBatchTQ3DescSet, @LKVStoreTQ3Push, SizeOf(LKVStoreTQ3Push),
    (FHeadDim div 32) * FNumKVHeads, ANumTokens);
  FCompute.BatchBarrier();  // Decode + TQ3 caches up to date

  //----------------------------------------------------------------
  // Step 4.5 — Continuation-prefill fill-in. If AStartPos > 0 the
  // decode buffers only hold this batch's tokens; positions [0,
  // AStartPos) contain stale data from the previous call's last
  // layer. Dequantize the full [0, SeqLen) range from the TQ3
  // cache so prefill attention reads a coherent K and V.
  //
  // On fresh prefill (AStartPos = 0) the decode buffers are already
  // populated correctly by Step 4, so we skip this pass and preserve
  // bit-exact behavior on the single-call path.
  //----------------------------------------------------------------
  if AStartPos > 0 then
  begin
    LTQ3DequantPush.BlocksPerHead := FHeadDim div 32;
    LTQ3DequantPush.MaxSeq        := FMaxSeqLen;
    LTQ3DequantPush.SeqLen        := LSeqLen;
    LTQ3DequantPush.NumHeads      := FNumKVHeads;

    FCompute.UpdateDescriptorSetBuffers(FTQ3KVDequantDescSet,
      [FKCacheTQ3[ALayerIndex], FKDecodeF32]);
    FCompute.DispatchComputeWithPush(
      FTQ3KVDequantBundle.Pipeline, FTQ3KVDequantBundle.PipelineLayout,
      FTQ3KVDequantDescSet, @LTQ3DequantPush, SizeOf(LTQ3DequantPush),
      (FHeadDim div 32) * FNumKVHeads * LSeqLen);

    FCompute.UpdateDescriptorSetBuffers(FTQ3KVDequantDescSet,
      [FVCacheTQ3[ALayerIndex], FVDecodeF32]);
    FCompute.DispatchComputeWithPush(
      FTQ3KVDequantBundle.Pipeline, FTQ3KVDequantBundle.PipelineLayout,
      FTQ3KVDequantDescSet, @LTQ3DequantPush, SizeOf(LTQ3DequantPush),
      (FHeadDim div 32) * FNumKVHeads * LSeqLen);
    FCompute.BatchBarrier();
  end;

  //----------------------------------------------------------------
  // Step 5a — Prefill attention scores. 3D dispatch (keys, heads,
  // queries). ABidirectional selects between causal (default) and
  // fully-connected attention shaders — identical bindings, same
  // push struct, different mask logic inside the shader.
  //----------------------------------------------------------------
  if ABidirectional then
    LScoresBundle := FScoresPrefillBidirBundle
  else
    LScoresBundle := FScoresPrefillBundle;

  LScoresPush.HeadDim   := FHeadDim;
  LScoresPush.NumTokens := ANumTokens;
  LScoresPush.MaxSeq    := FMaxSeqLen;
  LScoresPush.Scale     := 1.0 / Sqrt(Single(FHeadDim));
  LScoresPush.NumQHeads := FNumQHeads;
  LScoresPush.GqaRatio  := FNumQHeads div FNumKVHeads;
  LScoresPush.StartPos  := AStartPos;
  LScoresPush.SeqLen    := LSeqLen;

  FCompute.UpdateDescriptorSetBuffers(FPrefillScoresDescSet,
    [AQMat, FKDecodeF32, FPrefillScoresBuf]);
  FCompute.DispatchComputeWithPush(
    LScoresBundle.Pipeline, LScoresBundle.PipelineLayout,
    FPrefillScoresDescSet, @LScoresPush, SizeOf(LScoresPush),
    (LSeqLen + 255) div 256, FNumQHeads, ANumTokens);
  FCompute.BatchBarrier();  // Scores ready for softmax

  //----------------------------------------------------------------
  // Step 5b — prefill softmax. 2D dispatch (heads, queries).
  //----------------------------------------------------------------
  LSoftmaxPush.NumTokens := ANumTokens;
  LSoftmaxPush.NumQHeads := FNumQHeads;
  LSoftmaxPush.MaxSeq    := FMaxSeqLen;
  LSoftmaxPush.SeqLen    := LSeqLen;
  LSoftmaxPush.StartPos  := AStartPos;

  FCompute.UpdateDescriptorSetBuffers(FPrefillSoftmaxDescSet,
    [FPrefillScoresBuf]);
  FCompute.DispatchComputeWithPush(
    FSoftmaxPrefillBundle.Pipeline, FSoftmaxPrefillBundle.PipelineLayout,
    FPrefillSoftmaxDescSet, @LSoftmaxPush, SizeOf(LSoftmaxPush),
    FNumQHeads, ANumTokens);
  FCompute.BatchBarrier();  // Attention weights ready

  //----------------------------------------------------------------
  // Step 5c — prefill value. 3D dispatch (dim, heads, queries).
  // Output is reused AQMat [N x (NumQHeads * HeadDim)] as the
  // attention value tensor — the Q matrix is no longer needed after
  // the scores dispatch, so reusing saves an allocation.
  //----------------------------------------------------------------
  LValuePush.HeadDim   := FHeadDim;
  LValuePush.NumTokens := ANumTokens;
  LValuePush.MaxSeq    := FMaxSeqLen;
  LValuePush.NumQHeads := FNumQHeads;
  LValuePush.GqaRatio  := FNumQHeads div FNumKVHeads;
  LValuePush.StartPos  := AStartPos;
  LValuePush.SeqLen    := LSeqLen;

  FCompute.UpdateDescriptorSetBuffers(FPrefillValueDescSet,
    [FPrefillScoresBuf, FVDecodeF32, AQMat]);
  FCompute.DispatchComputeWithPush(
    FValuePrefillBundle.Pipeline, FValuePrefillBundle.PipelineLayout,
    FPrefillValueDescSet, @LValuePush, SizeOf(LValuePush),
    (FHeadDim + 255) div 256, FNumQHeads, ANumTokens);
  FCompute.BatchBarrier();  // AQMat holds [N x QDim] value output

  //----------------------------------------------------------------
  // Step 6 — Output projection, streamed through pair 3.
  //----------------------------------------------------------------
  StreamAndDispatchBatchMatMul(3,
    AWeights.OWeightPtr, AWeights.OWeightBytes,
    AQMat, AAttnOutMat,
    FNumQHeads * FHeadDim, FHiddenDim, ANumTokens, AWeights.WeightType);
  FCompute.BatchBarrier();  // AAttnOutMat ready for caller
end;

end.
