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

  TVdxAttnScoresPush = record
    HeadDim: UInt32;
    SeqLen: UInt32;
    Scale: Single;
    QOffset: UInt32;
    KOffset: UInt32;
  end;

  TVdxSoftmaxPush = record
    SeqLen: UInt32;
  end;

  TVdxAttnValuePush = record
    HeadDim: UInt32;
    SeqLen: UInt32;
    VOffset: UInt32;
    OutOffset: UInt32;
  end;
// ============================================================================
//  Per-Layer Attention Weight GPU Buffers
// ============================================================================

  TVdxAttnLayerWeights = record
    QWeightGpu: TVdxGpuBuffer;   // F16 [2560 x 2048] = Q projection
    KWeightGpu: TVdxGpuBuffer;   // F16 [2560 x 1024] = K projection
    VWeightGpu: TVdxGpuBuffer;   // F16 [2560 x 1024] = V projection
    OWeightGpu: TVdxGpuBuffer;   // F16 [2048 x 2560] = output projection
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
    FQKNormShader: VkShaderModule;
    FRoPEShader: VkShaderModule;
    FAttnScoresShader: VkShaderModule;
    FSoftmaxShader: VkShaderModule;
    FAttnValueShader: VkShaderModule;
    // Pipeline bundles
    FMatVecBundle: TVdxComputePipelineBundle;
    FQKNormBundle: TVdxComputePipelineBundle;
    FRoPEBundle: TVdxComputePipelineBundle;
    FAttnScoresBundle: TVdxComputePipelineBundle;
    FSoftmaxBundle: TVdxComputePipelineBundle;
    FAttnValueBundle: TVdxComputePipelineBundle;

    // Descriptor set layouts
    FMatVecDescLayout: VkDescriptorSetLayout;      // 3 bindings
    FQKNormDescLayout: VkDescriptorSetLayout;      // 2 bindings
    FRoPEDescLayout: VkDescriptorSetLayout;        // 1 binding
    FAttnScoresDescLayout: VkDescriptorSetLayout;  // 3 bindings
    FSoftmaxDescLayout: VkDescriptorSetLayout;     // 1 binding
    FAttnValueDescLayout: VkDescriptorSetLayout;   // 3 bindings

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
      const AInDim: UInt32; const AOutDim: UInt32);

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
      const AOutputBuf: TVdxGpuBuffer);

    // Expose for testing: run a single matvec F16 dispatch
    procedure TestMatVec(const AWeightBuf: TVdxGpuBuffer;
      const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
      const AInDim: UInt32; const AOutDim: UInt32);
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
  FQKNormShader := VK_NULL_HANDLE;
  FRoPEShader := VK_NULL_HANDLE;
  FAttnScoresShader := VK_NULL_HANDLE;
  FSoftmaxShader := VK_NULL_HANDLE;
  FAttnValueShader := VK_NULL_HANDLE;

  FMatVecBundle.Pipeline := VK_NULL_HANDLE;
  FQKNormBundle.Pipeline := VK_NULL_HANDLE;
  FRoPEBundle.Pipeline := VK_NULL_HANDLE;
  FAttnScoresBundle.Pipeline := VK_NULL_HANDLE;
  FSoftmaxBundle.Pipeline := VK_NULL_HANDLE;
  FAttnValueBundle.Pipeline := VK_NULL_HANDLE;

  FMatVecDescLayout := VK_NULL_HANDLE;
  FQKNormDescLayout := VK_NULL_HANDLE;
  FRoPEDescLayout := VK_NULL_HANDLE;
  FAttnScoresDescLayout := VK_NULL_HANDLE;
  FSoftmaxDescLayout := VK_NULL_HANDLE;
  FAttnValueDescLayout := VK_NULL_HANDLE;
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

  // Load all 6 shaders
  FMatVecShader := LoadShader('matvec_f16.spv');
  FQKNormShader := LoadShader('qk_norm.spv');
  FRoPEShader := LoadShader('rope.spv');
  FAttnScoresShader := LoadShader('attn_scores.spv');
  FSoftmaxShader := LoadShader('softmax.spv');
  FAttnValueShader := LoadShader('attn_value.spv');
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
  FQKNormBundle := FCompute.CreateComputePipelineWithPush(
    FQKNormShader, 'main', FQKNormDescLayout, SizeOf(TVdxQKNormPush));
  FRoPEBundle := FCompute.CreateComputePipelineWithPush(
    FRoPEShader, 'main', FRoPEDescLayout, SizeOf(TVdxRoPEPush));
  FAttnScoresBundle := FCompute.CreateComputePipelineWithPush(
    FAttnScoresShader, 'main', FAttnScoresDescLayout,
    SizeOf(TVdxAttnScoresPush));
  FSoftmaxBundle := FCompute.CreateComputePipelineWithPush(
    FSoftmaxShader, 'main', FSoftmaxDescLayout, SizeOf(TVdxSoftmaxPush));
  FAttnValueBundle := FCompute.CreateComputePipelineWithPush(
    FAttnValueShader, 'main', FAttnValueDescLayout,
    SizeOf(TVdxAttnValuePush));
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
    UInt64(FMaxSeqLen) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    FVCache[LI] := FCompute.CreateGpuBuffer(
      LCacheSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
  FCompute.DestroyComputePipelineBundle(FQKNormBundle);
  FCompute.DestroyComputePipelineBundle(FRoPEBundle);
  FCompute.DestroyComputePipelineBundle(FAttnScoresBundle);
  FCompute.DestroyComputePipelineBundle(FSoftmaxBundle);
  FCompute.DestroyComputePipelineBundle(FAttnValueBundle);

  // Destroy descriptor set layouts
  FCompute.DestroyDescriptorSetLayoutHandle(FMatVecDescLayout);
  FCompute.DestroyDescriptorSetLayoutHandle(FQKNormDescLayout);
  FCompute.DestroyDescriptorSetLayoutHandle(FRoPEDescLayout);
  FCompute.DestroyDescriptorSetLayoutHandle(FAttnScoresDescLayout);
  FCompute.DestroyDescriptorSetLayoutHandle(FSoftmaxDescLayout);
  FCompute.DestroyDescriptorSetLayoutHandle(FAttnValueDescLayout);

  // Destroy shader modules
  FCompute.DestroyShaderModuleHandle(FMatVecShader);
  FCompute.DestroyShaderModuleHandle(FQKNormShader);
  FCompute.DestroyShaderModuleHandle(FRoPEShader);
  FCompute.DestroyShaderModuleHandle(FAttnScoresShader);
  FCompute.DestroyShaderModuleHandle(FSoftmaxShader);
  FCompute.DestroyShaderModuleHandle(FAttnValueShader);

  FCompute := nil;
end;
// ============================================================================
//  TVdxAttention — DispatchMatVec: F16 weight matrix × F32 input → F32 output
// ============================================================================

procedure TVdxAttention.DispatchMatVec(const AWeightBuf: TVdxGpuBuffer;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32);
var
  LPush: TVdxMatVecF16Push;
  LDescPool: VkDescriptorPool;
  LDescSet: VkDescriptorSet;
  LGroups: UInt32;
begin
  LDescPool := FCompute.CreateDescriptorPoolForStorage(1, 3);
  try
    LDescSet := FCompute.AllocateDescriptorSetForBuffers(
      LDescPool, FMatVecDescLayout,
      [AWeightBuf, AInputBuf, AOutputBuf]);

    LPush.InDimHalf := AInDim div 2;
    LPush.OutDim := AOutDim;

    // One invocation per output row, 256 threads per workgroup
    LGroups := (AOutDim + 255) div 256;

    FCompute.DispatchComputeWithPush(
      FMatVecBundle.Pipeline,
      FMatVecBundle.PipelineLayout,
      LDescSet,
      @LPush,
      SizeOf(LPush),
      LGroups);
  finally
    FCompute.DestroyDescriptorPoolHandle(LDescPool);
  end;
end;
// ============================================================================
//  TVdxAttention — TestMatVec: Public wrapper for testing
// ============================================================================

procedure TVdxAttention.TestMatVec(const AWeightBuf: TVdxGpuBuffer;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32);
begin
  DispatchMatVec(AWeightBuf, AInputBuf, AOutputBuf, AInDim, AOutDim);
end;
// ============================================================================
//  TVdxAttention — Upload Attention Weights from GGUF
// ============================================================================

procedure TVdxAttention.UploadAttnWeights(const AReader: TVdxGGUFReader;
  const ALayerIndex: Integer; out AWeights: TVdxAttnLayerWeights);

  // Upload one F16 tensor from GGUF to device-local GPU buffer
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

    // Compute byte size: dims[0] * dims[1] * bytes_per_element
    // F16 = 2 bytes per element
    LSize := UInt64(LInfo.Dimensions[0]) * UInt64(LInfo.Dimensions[1]) * 2;

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
begin
  AWeights := Default(TVdxAttnLayerWeights);

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
//  TVdxAttention — Forward: Full attention for one layer at one position
// ============================================================================

procedure TVdxAttention.Forward(const AInputBuf: TVdxGpuBuffer;
  const AWeights: TVdxAttnLayerWeights;
  const AQNormBuf: TVdxGpuBuffer;
  const AKNormBuf: TVdxGpuBuffer;
  const ALayerIndex: Integer;
  const APosition: Integer;
  const AOutputBuf: TVdxGpuBuffer);
var
  LSeqLen: UInt32;
  LQHead: UInt32;
  LKVHead: UInt32;
  LQOffset: UInt32;
  LKVCacheOffset: UInt32;
  LHeadBytes: UInt64;
  LKVHeadBytes: UInt64;
  LDescPool: VkDescriptorPool;
  LDescSet: VkDescriptorSet;
  LQKNormPush: TVdxQKNormPush;
  LRoPEPush: TVdxRoPEPush;
  LScoresPush: TVdxAttnScoresPush;
  LSoftmaxPush: TVdxSoftmaxPush;
  LValuePush: TVdxAttnValuePush;
begin
  LSeqLen := UInt32(APosition) + 1;
  LHeadBytes := UInt64(FHeadDim) * SizeOf(Single);
  LKVHeadBytes := UInt64(FMaxSeqLen) * FHeadDim * SizeOf(Single);
  // ---- Step 1: Q/K/V projections (F16 matvec) ----
  DispatchMatVec(AWeights.QWeightGpu, AInputBuf, FQBuf,
    FHiddenDim, FNumQHeads * FHeadDim);
  DispatchMatVec(AWeights.KWeightGpu, AInputBuf, FKBuf,
    FHiddenDim, FNumKVHeads * FHeadDim);
  DispatchMatVec(AWeights.VWeightGpu, AInputBuf, FVBuf,
    FHiddenDim, FNumKVHeads * FHeadDim);

  // ---- Step 2: QK-norm on Q (8 heads) and K (4 heads) ----
  LQKNormPush.HeadDim := FHeadDim;
  LQKNormPush.Eps := 1e-6;

  // QK-norm on Q
  LQKNormPush.NumHeads := FNumQHeads;
  LDescPool := FCompute.CreateDescriptorPoolForStorage(1, 2);
  try
    LDescSet := FCompute.AllocateDescriptorSetForBuffers(
      LDescPool, FQKNormDescLayout, [FQBuf, AQNormBuf]);
    FCompute.DispatchComputeWithPush(
      FQKNormBundle.Pipeline, FQKNormBundle.PipelineLayout,
      LDescSet, @LQKNormPush, SizeOf(LQKNormPush), FNumQHeads);
  finally
    FCompute.DestroyDescriptorPoolHandle(LDescPool);
  end;

  // QK-norm on K
  LQKNormPush.NumHeads := FNumKVHeads;
  LDescPool := FCompute.CreateDescriptorPoolForStorage(1, 2);
  try
    LDescSet := FCompute.AllocateDescriptorSetForBuffers(
      LDescPool, FQKNormDescLayout, [FKBuf, AKNormBuf]);
    FCompute.DispatchComputeWithPush(
      FQKNormBundle.Pipeline, FQKNormBundle.PipelineLayout,
      LDescSet, @LQKNormPush, SizeOf(LQKNormPush), FNumKVHeads);
  finally
    FCompute.DestroyDescriptorPoolHandle(LDescPool);
  end;
  // ---- Step 3: RoPE on Q and K ----
  LRoPEPush.HeadDim := FHeadDim;
  LRoPEPush.Position := UInt32(APosition);
  LRoPEPush.ThetaBase := 10000.0;

  // RoPE on Q (8 heads)
  LRoPEPush.NumHeads := FNumQHeads;
  LDescPool := FCompute.CreateDescriptorPoolForStorage(1, 1);
  try
    LDescSet := FCompute.AllocateDescriptorSetForBuffers(
      LDescPool, FRoPEDescLayout, [FQBuf]);
    FCompute.DispatchComputeWithPush(
      FRoPEBundle.Pipeline, FRoPEBundle.PipelineLayout,
      LDescSet, @LRoPEPush, SizeOf(LRoPEPush), FNumQHeads);
  finally
    FCompute.DestroyDescriptorPoolHandle(LDescPool);
  end;

  // RoPE on K (4 heads)
  LRoPEPush.NumHeads := FNumKVHeads;
  LDescPool := FCompute.CreateDescriptorPoolForStorage(1, 1);
  try
    LDescSet := FCompute.AllocateDescriptorSetForBuffers(
      LDescPool, FRoPEDescLayout, [FKBuf]);
    FCompute.DispatchComputeWithPush(
      FRoPEBundle.Pipeline, FRoPEBundle.PipelineLayout,
      LDescSet, @LRoPEPush, SizeOf(LRoPEPush), FNumKVHeads);
  finally
    FCompute.DestroyDescriptorPoolHandle(LDescPool);
  end;
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
  // ---- Step 5: Per-head attention (scores → softmax → value) ----
  // GQA: each KV head serves NumQHeads/NumKVHeads Q heads
  for LQHead := 0 to FNumQHeads - 1 do
  begin
    LKVHead := LQHead div (FNumQHeads div FNumKVHeads);
    LQOffset := LQHead * FHeadDim;
    // KV cache offset for this KV head: h * MaxSeqLen * HeadDim (in floats)
    LKVCacheOffset := LKVHead * FMaxSeqLen * FHeadDim;

    // 5a. Attention scores: q_head dot all K cache entries
    LScoresPush.HeadDim := FHeadDim;
    LScoresPush.SeqLen := LSeqLen;
    LScoresPush.Scale := 1.0 / Sqrt(Single(FHeadDim));
    LScoresPush.QOffset := LQOffset;
    LScoresPush.KOffset := LKVCacheOffset;

    LDescPool := FCompute.CreateDescriptorPoolForStorage(1, 3);
    try
      LDescSet := FCompute.AllocateDescriptorSetForBuffers(
        LDescPool, FAttnScoresDescLayout,
        [FQBuf, FKCache[ALayerIndex], FScoresBuf]);
      FCompute.DispatchComputeWithPush(
        FAttnScoresBundle.Pipeline, FAttnScoresBundle.PipelineLayout,
        LDescSet, @LScoresPush, SizeOf(LScoresPush),
        (LSeqLen + 255) div 256);
    finally
      FCompute.DestroyDescriptorPoolHandle(LDescPool);
    end;
    // 5b. Softmax on scores
    LSoftmaxPush.SeqLen := LSeqLen;

    LDescPool := FCompute.CreateDescriptorPoolForStorage(1, 1);
    try
      LDescSet := FCompute.AllocateDescriptorSetForBuffers(
        LDescPool, FSoftmaxDescLayout, [FScoresBuf]);
      FCompute.DispatchComputeWithPush(
        FSoftmaxBundle.Pipeline, FSoftmaxBundle.PipelineLayout,
        LDescSet, @LSoftmaxPush, SizeOf(LSoftmaxPush),
        1);  // single workgroup
    finally
      FCompute.DestroyDescriptorPoolHandle(LDescPool);
    end;

    // 5c. Weighted V sum → write to AttnOutBuf at this head's offset
    LValuePush.HeadDim := FHeadDim;
    LValuePush.SeqLen := LSeqLen;
    LValuePush.VOffset := LKVCacheOffset;
    LValuePush.OutOffset := LQOffset;

    LDescPool := FCompute.CreateDescriptorPoolForStorage(1, 3);
    try
      LDescSet := FCompute.AllocateDescriptorSetForBuffers(
        LDescPool, FAttnValueDescLayout,
        [FScoresBuf, FVCache[ALayerIndex], FAttnOutBuf]);
      FCompute.DispatchComputeWithPush(
        FAttnValueBundle.Pipeline, FAttnValueBundle.PipelineLayout,
        LDescSet, @LValuePush, SizeOf(LValuePush),
        (FHeadDim + 255) div 256);
    finally
      FCompute.DestroyDescriptorPoolHandle(LDescPool);
    end;
  end;
  // ---- Step 6: Output projection ----
  // O: W[2048 × 2560] F16 × attn_out[2048] F32 → output[2560] F32
  DispatchMatVec(AWeights.OWeightGpu, FAttnOutBuf, AOutputBuf,
    FNumQHeads * FHeadDim, FHiddenDim);
end;

end.