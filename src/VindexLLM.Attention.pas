{===============================================================================
  VindexLLM - Graph-Walk LLM Inference Engine

  Copyright (c) 2026-present tinyBigGAMES LLC
  All Rights Reserved.

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Attention;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  VindexLLM.Utils,
  VindexLLM.GGUFReader,
  VindexLLM.VulkanCompute;

// ============================================================================
//  Push Constant Records (must match shader layouts exactly)
// ============================================================================

type
  TVdxMatVecF16Push = record
    InDimHalf: UInt32;
    OutDim: UInt32;
  end;
  TVdxQKNormPush = record
    HeadDim: UInt32;
    NumHeads: UInt32;
    Eps: Single;
  end;

  TVdxRoPEPush = record
    HeadDim: UInt32;
    NumHeads: UInt32;
    Position: UInt32;
    ThetaBase: Single;
  end;

  // Multi-head push constants (all heads in one dispatch)
  TVdxAttnScoresMHPush = record
    HeadDim: UInt32;
    SeqLen: UInt32;
    MaxSeq: UInt32;
    Scale: Single;
    NumQHeads: UInt32;
    GqaRatio: UInt32;
  end;

  TVdxSoftmaxMHPush = record
    SeqLen: UInt32;
    MaxSeq: UInt32;
    NumQHeads: UInt32;
  end;

  TVdxAttnValueMHPush = record
    HeadDim: UInt32;
    SeqLen: UInt32;
    MaxSeq: UInt32;
    NumQHeads: UInt32;
    GqaRatio: UInt32;
  end;
// ============================================================================
//  Per-Layer Attention Weight GPU Buffers
// ============================================================================

  TVdxAttnLayerWeights = record
    QWeightGpu: TVdxGpuBuffer;   // F16 or Q4_0 [2560 x 2048] = Q projection
    KWeightGpu: TVdxGpuBuffer;   // F16 or Q4_0 [2560 x 1024] = K projection
    VWeightGpu: TVdxGpuBuffer;   // F16 or Q4_0 [2560 x 1024] = V projection
    OWeightGpu: TVdxGpuBuffer;   // F16 or Q4_0 [2048 x 2560] = output projection
    WeightType: TVdxGGMLType;    // tensor format (gtF16 or gtQ4_0)
  end;

// ============================================================================
//  TVdxAttention — Full attention layer (QKV + QK-norm + RoPE + GQA + O)
// ============================================================================

  { TVdxAttention }
  TVdxAttention = class(TVdxBaseObject)
  private
    FCompute: TVdxVulkanCompute;

    // Shader modules
    FMatVecShader: VkShaderModule;
    FMatVecQ8Shader: VkShaderModule;
    FQKNormShader: VkShaderModule;
    FRoPEShader: VkShaderModule;
    FAttnScoresMHShader: VkShaderModule;
    FSoftmaxMHShader: VkShaderModule;
    FAttnValueMHShader: VkShaderModule;
    // Pipeline bundles
    FMatVecBundle: TVdxComputePipelineBundle;
    FMatVecQ8Bundle: TVdxComputePipelineBundle;
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

    // Scratch buffers (reused every Forward call)
    FQBuf: TVdxGpuBuffer;        // [NumQHeads * HeadDim] F32
    FKBuf: TVdxGpuBuffer;        // [NumKVHeads * HeadDim] F32
    FVBuf: TVdxGpuBuffer;        // [NumKVHeads * HeadDim] F32
    FScoresBuf: TVdxGpuBuffer;   // [MaxSeqLen] F32
    FAttnOutBuf: TVdxGpuBuffer;  // [NumQHeads * HeadDim] F32

    // KV cache — one pair per layer
    FKCache: array of TVdxGpuBuffer;  // each [NumKVHeads * MaxSeq * HeadDim] F32
    FVCache: array of TVdxGpuBuffer;  // same layout
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

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Initialize all shaders, pipelines, scratch buffers, and KV cache
    procedure Init(const ACompute: TVdxVulkanCompute;
      const AHiddenDim: UInt32; const ANumQHeads: UInt32;
      const ANumKVHeads: UInt32; const AHeadDim: UInt32;
      const ANumLayers: UInt32; const AMaxSeqLen: UInt32);

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

    // Diagnostic read-only access to internal buffers (for debugging tests)
    property ScoresBuf: TVdxGpuBuffer read FScoresBuf;
    property QBuf: TVdxGpuBuffer read FQBuf;
    property KBuf: TVdxGpuBuffer read FKBuf;
    property VBuf: TVdxGpuBuffer read FVBuf;
    property AttnOutBufInternal: TVdxGpuBuffer read FAttnOutBuf;
    function GetKCache(const ALayerIndex: Integer): TVdxGpuBuffer;
    function GetVCache(const ALayerIndex: Integer): TVdxGpuBuffer;
  end;

implementation

uses
  System.IOUtils;
// ============================================================================
//  TVdxAttention — Construction / Destruction
// ============================================================================

constructor TVdxAttention.Create();
begin
  inherited;

  FCompute := nil;
  FMatVecShader := VK_NULL_HANDLE;
  FMatVecQ8Shader := VK_NULL_HANDLE;
  FQKNormShader := VK_NULL_HANDLE;
  FRoPEShader := VK_NULL_HANDLE;
  FAttnScoresMHShader := VK_NULL_HANDLE;
  FSoftmaxMHShader := VK_NULL_HANDLE;
  FAttnValueMHShader := VK_NULL_HANDLE;

  FMatVecBundle.Pipeline := VK_NULL_HANDLE;
  FMatVecQ8Bundle.Pipeline := VK_NULL_HANDLE;
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
end;
destructor TVdxAttention.Destroy();
begin
  if FCompute <> nil then
    Cleanup();

  inherited;
end;

// ============================================================================
//  TVdxAttention — LoadShader helper
// ============================================================================

function TVdxAttention.LoadShader(const AFileName: string): VkShaderModule;
var
  LSpvPath: string;
  LSpvData: TBytes;
begin
  LSpvPath := TPath.Combine(
    TPath.GetDirectoryName(ParamStr(0)),
    '..\shaders\' + AFileName
  );
  LSpvPath := TPath.GetFullPath(LSpvPath);
  TVdxUtils.FailIf(not TFile.Exists(LSpvPath),
    'Attention: shader not found: %s', [LSpvPath]);
  LSpvData := TFile.ReadAllBytes(LSpvPath);
  Result := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
end;
// ============================================================================
//  TVdxAttention — Init: Load shaders, create pipelines, allocate buffers
// ============================================================================

procedure TVdxAttention.Init(const ACompute: TVdxVulkanCompute;
  const AHiddenDim: UInt32; const ANumQHeads: UInt32;
  const ANumKVHeads: UInt32; const AHeadDim: UInt32;
  const ANumLayers: UInt32; const AMaxSeqLen: UInt32);
var
  LI: Integer;
  LCacheSize: UInt64;
begin
  FCompute := ACompute;
  FHiddenDim := AHiddenDim;
  FNumQHeads := ANumQHeads;
  FNumKVHeads := ANumKVHeads;
  FHeadDim := AHeadDim;
  FNumLayers := ANumLayers;
  FMaxSeqLen := AMaxSeqLen;

  // Load all shaders
  FMatVecShader := LoadShader('matvec_f16.spv');
  FMatVecQ8Shader := LoadShader('matvec_q8_0.spv');
  FQKNormShader := LoadShader('qk_norm.spv');
  FRoPEShader := LoadShader('rope.spv');
  FAttnScoresMHShader := LoadShader('attn_scores_mh.spv');
  FSoftmaxMHShader := LoadShader('softmax_mh.spv');
  FAttnValueMHShader := LoadShader('attn_value_mh.spv');
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
  FMatVecQ8Bundle := FCompute.CreateComputePipelineWithPush(
    FMatVecQ8Shader, 'main', FMatVecDescLayout, SizeOf(TVdxMatVecF16Push));
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
  // Allocate KV cache per layer
  // Layout: [NumKVHeads * MaxSeqLen * HeadDim] F32 per layer
  LCacheSize := UInt64(FNumKVHeads) * FMaxSeqLen * FHeadDim * SizeOf(Single);
  SetLength(FKCache, FNumLayers);
  SetLength(FVCache, FNumLayers);

  for LI := 0 to FNumLayers - 1 do
  begin
    FKCache[LI] := FCompute.CreateGpuBuffer(
      LCacheSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT
        or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    FVCache[LI] := FCompute.CreateGpuBuffer(
      LCacheSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT
        or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  end;
end;
// ============================================================================
//  TVdxAttention — Cleanup
// ============================================================================

procedure TVdxAttention.Cleanup();
var
  LI: Integer;
begin
  if FCompute = nil then
    Exit;

  // Free KV cache
  for LI := 0 to Length(FKCache) - 1 do
  begin
    if FKCache[LI].Buffer <> VK_NULL_HANDLE then
      FCompute.DestroyGpuBuffer(FKCache[LI]);
    if FVCache[LI].Buffer <> VK_NULL_HANDLE then
      FCompute.DestroyGpuBuffer(FVCache[LI]);
  end;
  SetLength(FKCache, 0);
  SetLength(FVCache, 0);

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
  FCompute.DestroyComputePipelineBundle(FQKNormBundle);
  FCompute.DestroyComputePipelineBundle(FRoPEBundle);
  FCompute.DestroyComputePipelineBundle(FAttnScoresMHBundle);
  FCompute.DestroyComputePipelineBundle(FSoftmaxMHBundle);
  FCompute.DestroyComputePipelineBundle(FAttnValueMHBundle);

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
  FCompute.DestroyShaderModuleHandle(FQKNormShader);
  FCompute.DestroyShaderModuleHandle(FRoPEShader);
  FCompute.DestroyShaderModuleHandle(FAttnScoresMHShader);
  FCompute.DestroyShaderModuleHandle(FSoftmaxMHShader);
  FCompute.DestroyShaderModuleHandle(FAttnValueMHShader);

  FCompute := nil;
end;
// ============================================================================
//  TVdxAttention — DispatchMatVec: F16 weight matrix × F32 input → F32 output
// ============================================================================

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

  // Select pipeline: Q8_0 uses full in_dim, F16 uses in_dim/2
  if ATensorType = gtQ8_0 then
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
// ============================================================================
//  TVdxAttention — TestMatVec: Public wrapper for testing
// ============================================================================

procedure TVdxAttention.TestMatVec(const AWeightBuf: TVdxGpuBuffer;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ATensorType: TVdxGGMLType);
begin
  DispatchMatVec(AWeightBuf, AInputBuf, AOutputBuf, AInDim, AOutDim, ATensorType);
end;
// ============================================================================
//  TVdxAttention — Upload Attention Weights from GGUF
// ============================================================================

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

// ============================================================================
//  TVdxAttention — Free Attention Weight GPU Buffers
// ============================================================================

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
// ============================================================================
//  TVdxAttention — Diagnostic accessors for KV cache
// ============================================================================

function TVdxAttention.GetKCache(const ALayerIndex: Integer): TVdxGpuBuffer;
begin
  Result := FKCache[ALayerIndex];
end;

function TVdxAttention.GetVCache(const ALayerIndex: Integer): TVdxGpuBuffer;
begin
  Result := FVCache[ALayerIndex];
end;

// ============================================================================
//  TVdxAttention — Forward: Full attention for one layer at one position
// ============================================================================

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
  LKVHead: UInt32;
  LHeadBytes: UInt64;
  LKVHeadBytes: UInt64;
  LQKNormPush: TVdxQKNormPush;
  LRoPEPush: TVdxRoPEPush;
  LScoresMHPush: TVdxAttnScoresMHPush;
  LSoftmaxMHPush: TVdxSoftmaxMHPush;
  LValueMHPush: TVdxAttnValueMHPush;
begin
  LSeqLen := UInt32(APosition) + 1;
  LHeadBytes := UInt64(FHeadDim) * SizeOf(Single);
  LKVHeadBytes := UInt64(FMaxSeqLen) * FHeadDim * SizeOf(Single);
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
  // ---- Step 4: Store K and V in KV cache at current position ----
  // Cache layout: [NumKVHeads × MaxSeqLen × HeadDim]
  // For each KV head h, copy HeadDim floats from K/VBuf to cache
  for LKVHead := 0 to FNumKVHeads - 1 do
  begin
    // Source: FKBuf at offset h * HeadDim * sizeof(float)
    // Dest: KCache at offset (h * MaxSeqLen + position) * HeadDim * sizeof(float)
    FCompute.CopyBufferRegion(
      FKBuf,
      UInt64(LKVHead) * LHeadBytes,
      FKCache[ALayerIndex],
      (UInt64(LKVHead) * FMaxSeqLen + UInt64(APosition)) * FHeadDim * SizeOf(Single),
      LHeadBytes);

    FCompute.CopyBufferRegion(
      FVBuf,
      UInt64(LKVHead) * LHeadBytes,
      FVCache[ALayerIndex],
      (UInt64(LKVHead) * FMaxSeqLen + UInt64(APosition)) * FHeadDim * SizeOf(Single),
      LHeadBytes);
  end;
  FCompute.BatchBarrier(); // KV cache updated, ready for attention scores
  // ---- Step 5: Multi-head attention (all heads in 3 dispatches) ----

  // 5a. All heads' scores in one 2D dispatch
  LScoresMHPush.HeadDim := FHeadDim;
  LScoresMHPush.SeqLen := LSeqLen;
  LScoresMHPush.MaxSeq := FMaxSeqLen;
  LScoresMHPush.Scale := 1.0 / Sqrt(Single(FHeadDim));
  LScoresMHPush.NumQHeads := FNumQHeads;
  LScoresMHPush.GqaRatio := FNumQHeads div FNumKVHeads;

  FCompute.UpdateDescriptorSetBuffers(FScoresDescSet,
    [FQBuf, FKCache[ALayerIndex], FScoresBuf]);
  FCompute.DispatchComputeWithPush(
    FAttnScoresMHBundle.Pipeline, FAttnScoresMHBundle.PipelineLayout,
    FScoresDescSet, @LScoresMHPush, SizeOf(LScoresMHPush),
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
    [FScoresBuf, FVCache[ALayerIndex], FAttnOutBuf]);
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