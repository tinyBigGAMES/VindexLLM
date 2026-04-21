{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.LayerNorm;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  VindexLLM.Utils,
  VindexLLM.GGUFReader,
  VindexLLM.Vulkan,
  VindexLLM.Compute;

type

  { TVdxRMSNormPush }
  TVdxRMSNormPush = record
    HiddenDim: UInt32;
    Eps: Single;
  end;

  { TVdxRMSNormBatchPush }
  TVdxRMSNormBatchPush = record
    HiddenDim: UInt32;
    Eps: Single;
    NumTokens: UInt32;
  end;

  { TVdxNormLayerWeights }
  TVdxNormLayerWeights = record
    AttnNormGpu: TVdxGpuBuffer;       // F32 x HiddenDim, pre-attention norm
    PostAttnNormGpu: TVdxGpuBuffer;   // F32 x HiddenDim, post-attention norm
    FFNNormGpu: TVdxGpuBuffer;        // F32 x HiddenDim, pre-FFN norm
    PostFFNNormGpu: TVdxGpuBuffer;    // F32 x HiddenDim, post-FFN norm
    QNormGpu: TVdxGpuBuffer;          // F32 x HeadDim (256), QK-norm on queries
    KNormGpu: TVdxGpuBuffer;          // F32 x HeadDim (256), QK-norm on keys
  end;

  { TVdxLayerNorm }
  TVdxLayerNorm = class(TVdxBaseObject)
  private
    FCompute: TVdxCompute;
    FShaderModule: VkShaderModule;
    FBundle: TVdxComputePipelineBundle;
    FDescLayout: VkDescriptorSetLayout;
    FDescPool: VkDescriptorPool;
    FDescSet: VkDescriptorSet;

    // Fused copy+norm pipeline (reads source, norms, writes dest)
    FShaderModuleCopy: VkShaderModule;
    FBundleCopy: TVdxComputePipelineBundle;
    FDescLayoutCopy: VkDescriptorSetLayout;  // 3 bindings: source, weight, dest
    FDescPoolCopy: VkDescriptorPool;
    FDescSetCopy: VkDescriptorSet;

    // Fused norm+add pipeline (reads input, norms, adds to accumulator)
    FShaderModuleAdd: VkShaderModule;
    FBundleAdd: TVdxComputePipelineBundle;
    FDescPoolAdd: VkDescriptorPool;
    FDescSetAdd: VkDescriptorSet;

    // Batched fused norm+add pipeline
    FShaderModuleAddBatch: VkShaderModule;
    FBundleAddBatch: TVdxComputePipelineBundle;
    FDescPoolAddBatch: VkDescriptorPool;
    FDescSetAddBatch: VkDescriptorSet;

    // Batched norm pipelines (Phase 6D — prefill batching)
    // In-place batch: dispatch (num_tokens,1,1), reuses FDescLayout (2 bindings)
    FShaderModuleBatch: VkShaderModule;
    FBundleBatch: TVdxComputePipelineBundle;
    FDescPoolBatch: VkDescriptorPool;
    FDescSetBatch: VkDescriptorSet;

    // Fused copy+norm batch: reuses FDescLayoutCopy (3 bindings)
    FShaderModuleCopyBatch: VkShaderModule;
    FBundleCopyBatch: TVdxComputePipelineBundle;
    FDescPoolCopyBatch: VkDescriptorPool;
    FDescSetCopyBatch: VkDescriptorSet;

    FEpsilon: Single;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Initialize shader pipeline (call after setting status callback)
    procedure Init(const ACompute: TVdxCompute;
      const AEpsilon: Single = 1e-6);

    // Release GPU resources
    procedure Cleanup();

    // Apply RMSNorm in-place on residual using weight buffer
    procedure Apply(const AResidualBuf: TVdxGpuBuffer;
      const AWeightBuf: TVdxGpuBuffer; const AHiddenDim: UInt32);

    // Fused copy+norm: read source, normalize, write to dest (source untouched)
    procedure ApplyCopy(const ASourceBuf: TVdxGpuBuffer;
      const AWeightBuf: TVdxGpuBuffer; const ADestBuf: TVdxGpuBuffer;
      const AHiddenDim: UInt32);

    // Batched in-place RMSNorm on matrix [NumTokens x HiddenDim]
    procedure ApplyBatch(const AMatrixBuf: TVdxGpuBuffer;
      const AWeightBuf: TVdxGpuBuffer; const AHiddenDim: UInt32;
      const ANumTokens: UInt32);

    // Batched fused copy+norm on matrices [NumTokens x HiddenDim]
    procedure ApplyCopyBatch(const ASourceBuf: TVdxGpuBuffer;
      const AWeightBuf: TVdxGpuBuffer; const ADestBuf: TVdxGpuBuffer;
      const AHiddenDim: UInt32; const ANumTokens: UInt32);

    // Fused norm+add: read input, normalize, add to accumulator (input untouched)
    procedure ApplyAdd(const AInputBuf: TVdxGpuBuffer;
      const AWeightBuf: TVdxGpuBuffer; const AAccumBuf: TVdxGpuBuffer;
      const AHiddenDim: UInt32);

    // Batched fused norm+add on matrices [NumTokens x HiddenDim]
    procedure ApplyAddBatch(const AInputBuf: TVdxGpuBuffer;
      const AWeightBuf: TVdxGpuBuffer; const AAccumBuf: TVdxGpuBuffer;
      const AHiddenDim: UInt32; const ANumTokens: UInt32);

    // Upload attn_norm + ffn_norm weights from GGUF to GPU for one layer
    procedure UploadNormWeights(const AReader: TVdxGGUFReader;
      const ALayerIndex: Integer; out AWeights: TVdxNormLayerWeights);

    // Free GPU buffers for one layer's norm weights
    procedure FreeNormWeights(var AWeights: TVdxNormLayerWeights);
  end;

implementation

uses
  System.IOUtils,
  VindexLLM.Shaders;

{ TVdxLayerNorm }
constructor TVdxLayerNorm.Create();
begin
  inherited;

  FCompute := nil;
  FShaderModule := VK_NULL_HANDLE;
  FBundle.Pipeline := VK_NULL_HANDLE;
  FBundle.PipelineLayout := VK_NULL_HANDLE;
  FDescLayout := VK_NULL_HANDLE;
  FDescPool := VK_NULL_HANDLE;
  FDescSet := VK_NULL_HANDLE;
  FShaderModuleCopy := VK_NULL_HANDLE;
  FBundleCopy.Pipeline := VK_NULL_HANDLE;
  FBundleCopy.PipelineLayout := VK_NULL_HANDLE;
  FDescLayoutCopy := VK_NULL_HANDLE;
  FDescPoolCopy := VK_NULL_HANDLE;
  FDescSetCopy := VK_NULL_HANDLE;
  FShaderModuleAdd := VK_NULL_HANDLE;
  FBundleAdd.Pipeline := VK_NULL_HANDLE;
  FBundleAdd.PipelineLayout := VK_NULL_HANDLE;
  FDescPoolAdd := VK_NULL_HANDLE;
  FDescSetAdd := VK_NULL_HANDLE;
  FShaderModuleAddBatch := VK_NULL_HANDLE;
  FBundleAddBatch.Pipeline := VK_NULL_HANDLE;
  FBundleAddBatch.PipelineLayout := VK_NULL_HANDLE;
  FDescPoolAddBatch := VK_NULL_HANDLE;
  FDescSetAddBatch := VK_NULL_HANDLE;
  FShaderModuleBatch := VK_NULL_HANDLE;
  FBundleBatch.Pipeline := VK_NULL_HANDLE;
  FBundleBatch.PipelineLayout := VK_NULL_HANDLE;
  FDescPoolBatch := VK_NULL_HANDLE;
  FDescSetBatch := VK_NULL_HANDLE;
  FShaderModuleCopyBatch := VK_NULL_HANDLE;
  FBundleCopyBatch.Pipeline := VK_NULL_HANDLE;
  FBundleCopyBatch.PipelineLayout := VK_NULL_HANDLE;
  FDescPoolCopyBatch := VK_NULL_HANDLE;
  FDescSetCopyBatch := VK_NULL_HANDLE;
  FEpsilon := 1e-6;
end;

destructor TVdxLayerNorm.Destroy();
begin
  if FCompute <> nil then
    Cleanup();

  inherited;
end;

procedure TVdxLayerNorm.Init(const ACompute: TVdxCompute;
  const AEpsilon: Single);
var
  LSpvData: TBytes;
  LDummyBuf: TVdxGpuBuffer;
begin
  FCompute := ACompute;
  FEpsilon := AEpsilon;

  Status('LayerNorm: Init (eps=%e)', [Double(FEpsilon)]);

  // Load rmsnorm shader from embedded resource
  LSpvData := VdxLoadShader('RMSNORM');
  FShaderModule := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  // Descriptor layout: binding 0=residual, 1=weight
  FDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);

  // Pipeline with push constants (hidden_dim + eps)
  FBundle := FCompute.CreateComputePipelineWithPush(
    FShaderModule, 'main', FDescLayout, SizeOf(TVdxRMSNormPush));

  // Pre-allocate reusable descriptor pool + set (eliminates per-Apply churn)
  // 1 set, 2 storage buffer bindings (residual + weight)
  LDummyBuf := Default(TVdxGpuBuffer);
  FDescPool := FCompute.CreateDescriptorPoolForStorage(1, 2);
  FDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FDescPool, FDescLayout, [LDummyBuf, LDummyBuf]);

  // Load fused copy+norm shader from embedded resource
  LSpvData := VdxLoadShader('RMSNORM_COPY');
  FShaderModuleCopy := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  // 3-binding layout: source (readonly), weight (readonly), dest (writeonly)
  FDescLayoutCopy := FCompute.CreateStorageDescriptorSetLayout(3);

  // Pipeline with same push constants (hidden_dim + eps)
  FBundleCopy := FCompute.CreateComputePipelineWithPush(
    FShaderModuleCopy, 'main', FDescLayoutCopy, SizeOf(TVdxRMSNormPush));

  // Pre-allocate reusable descriptor pool + set for fused shader
  FDescPoolCopy := FCompute.CreateDescriptorPoolForStorage(1, 3);
  FDescSetCopy := FCompute.AllocateDescriptorSetForBuffers(
    FDescPoolCopy, FDescLayoutCopy, [LDummyBuf, LDummyBuf, LDummyBuf]);

  // --- Batched in-place norm (Phase 6D) ---
  LSpvData := VdxLoadShader('RMSNORM_BATCH');
  FShaderModuleBatch := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  // Reuse FDescLayout (2 bindings: matrix, weight)
  FBundleBatch := FCompute.CreateComputePipelineWithPush(
    FShaderModuleBatch, 'main', FDescLayout, SizeOf(TVdxRMSNormBatchPush));
  FDescPoolBatch := FCompute.CreateDescriptorPoolForStorage(1, 2);
  FDescSetBatch := FCompute.AllocateDescriptorSetForBuffers(
    FDescPoolBatch, FDescLayout, [LDummyBuf, LDummyBuf]);

  // --- Batched fused copy+norm (Phase 6D) ---
  LSpvData := VdxLoadShader('RMSNORM_COPY_BATCH');
  FShaderModuleCopyBatch := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  // Reuse FDescLayoutCopy (3 bindings: source, weight, dest)
  FBundleCopyBatch := FCompute.CreateComputePipelineWithPush(
    FShaderModuleCopyBatch, 'main', FDescLayoutCopy,
    SizeOf(TVdxRMSNormBatchPush));
  FDescPoolCopyBatch := FCompute.CreateDescriptorPoolForStorage(1, 3);
  FDescSetCopyBatch := FCompute.AllocateDescriptorSetForBuffers(
    FDescPoolCopyBatch, FDescLayoutCopy, [LDummyBuf, LDummyBuf, LDummyBuf]);

  // --- Fused norm+add (single token) ---
  LSpvData := VdxLoadShader('RMSNORM_ADD');
  FShaderModuleAdd := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  // Reuse FDescLayoutCopy (3 bindings: input, weight, accumulator)
  FBundleAdd := FCompute.CreateComputePipelineWithPush(
    FShaderModuleAdd, 'main', FDescLayoutCopy, SizeOf(TVdxRMSNormPush));
  FDescPoolAdd := FCompute.CreateDescriptorPoolForStorage(1, 3);
  FDescSetAdd := FCompute.AllocateDescriptorSetForBuffers(
    FDescPoolAdd, FDescLayoutCopy, [LDummyBuf, LDummyBuf, LDummyBuf]);

  // --- Batched fused norm+add ---
  LSpvData := VdxLoadShader('RMSNORM_ADD_BATCH');
  FShaderModuleAddBatch := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  // Reuse FDescLayoutCopy (3 bindings: input, weight, accumulator)
  FBundleAddBatch := FCompute.CreateComputePipelineWithPush(
    FShaderModuleAddBatch, 'main', FDescLayoutCopy,
    SizeOf(TVdxRMSNormBatchPush));
  FDescPoolAddBatch := FCompute.CreateDescriptorPoolForStorage(1, 3);
  FDescSetAddBatch := FCompute.AllocateDescriptorSetForBuffers(
    FDescPoolAddBatch, FDescLayoutCopy, [LDummyBuf, LDummyBuf, LDummyBuf]);

  Status('LayerNorm: Ready');
end;

procedure TVdxLayerNorm.Cleanup();
begin
  if FCompute = nil then
    Exit;

  FCompute.DestroyComputePipelineBundle(FBundle);

  // Destroy fused copy+norm resources
  FCompute.DestroyComputePipelineBundle(FBundleCopy);

  // Destroy batched norm resources (Phase 6D)
  FCompute.DestroyComputePipelineBundle(FBundleBatch);
  if FDescPoolBatch <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FDescPoolBatch);
    FDescPoolBatch := VK_NULL_HANDLE;
  end;
  FCompute.DestroyShaderModuleHandle(FShaderModuleBatch);
  FShaderModuleBatch := VK_NULL_HANDLE;

  FCompute.DestroyComputePipelineBundle(FBundleCopyBatch);
  if FDescPoolCopyBatch <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FDescPoolCopyBatch);
    FDescPoolCopyBatch := VK_NULL_HANDLE;
  end;
  FCompute.DestroyShaderModuleHandle(FShaderModuleCopyBatch);
  FShaderModuleCopyBatch := VK_NULL_HANDLE;

  // Destroy fused norm+add resources
  FCompute.DestroyComputePipelineBundle(FBundleAdd);
  if FDescPoolAdd <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FDescPoolAdd);
    FDescPoolAdd := VK_NULL_HANDLE;
  end;
  FCompute.DestroyShaderModuleHandle(FShaderModuleAdd);
  FShaderModuleAdd := VK_NULL_HANDLE;

  FCompute.DestroyComputePipelineBundle(FBundleAddBatch);
  if FDescPoolAddBatch <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FDescPoolAddBatch);
    FDescPoolAddBatch := VK_NULL_HANDLE;
  end;
  FCompute.DestroyShaderModuleHandle(FShaderModuleAddBatch);
  FShaderModuleAddBatch := VK_NULL_HANDLE;

  if FDescPoolCopy <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FDescPoolCopy);
    FDescPoolCopy := VK_NULL_HANDLE;
  end;

  FCompute.DestroyDescriptorSetLayoutHandle(FDescLayoutCopy);
  FDescLayoutCopy := VK_NULL_HANDLE;

  FCompute.DestroyShaderModuleHandle(FShaderModuleCopy);
  FShaderModuleCopy := VK_NULL_HANDLE;

  // Destroy in-place norm resources
  if FDescPool <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FDescPool);
    FDescPool := VK_NULL_HANDLE;
  end;

  FCompute.DestroyDescriptorSetLayoutHandle(FDescLayout);
  FDescLayout := VK_NULL_HANDLE;

  FCompute.DestroyShaderModuleHandle(FShaderModule);
  FShaderModule := VK_NULL_HANDLE;

  FCompute := nil;
end;

procedure TVdxLayerNorm.Apply(const AResidualBuf: TVdxGpuBuffer;
  const AWeightBuf: TVdxGpuBuffer; const AHiddenDim: UInt32);
var
  LPush: TVdxRMSNormPush;
begin
  // Rebind buffers to pre-allocated descriptor set (no pool create/destroy)
  FCompute.UpdateDescriptorSetBuffers(FDescSet, [AResidualBuf, AWeightBuf]);

  LPush.HiddenDim := AHiddenDim;
  LPush.Eps := FEpsilon;

  // Single workgroup — all 2560 elements handled by 256 threads
  FCompute.DispatchComputeWithPush(
    FBundle.Pipeline,
    FBundle.PipelineLayout,
    FDescSet,
    @LPush,
    SizeOf(LPush),
    1  // 1 workgroup
  );
end;

procedure TVdxLayerNorm.ApplyCopy(const ASourceBuf: TVdxGpuBuffer;
  const AWeightBuf: TVdxGpuBuffer; const ADestBuf: TVdxGpuBuffer;
  const AHiddenDim: UInt32);
var
  LPush: TVdxRMSNormPush;
begin
  // Rebind buffers to pre-allocated fused descriptor set
  FCompute.UpdateDescriptorSetBuffers(FDescSetCopy,
    [ASourceBuf, AWeightBuf, ADestBuf]);

  LPush.HiddenDim := AHiddenDim;
  LPush.Eps := FEpsilon;

  // Single workgroup — reads source, norms, writes dest
  FCompute.DispatchComputeWithPush(
    FBundleCopy.Pipeline,
    FBundleCopy.PipelineLayout,
    FDescSetCopy,
    @LPush,
    SizeOf(LPush),
    1  // 1 workgroup
  );
end;

procedure TVdxLayerNorm.ApplyBatch(const AMatrixBuf: TVdxGpuBuffer;
  const AWeightBuf: TVdxGpuBuffer; const AHiddenDim: UInt32;
  const ANumTokens: UInt32);
var
  LPush: TVdxRMSNormBatchPush;
begin
  FCompute.UpdateDescriptorSetBuffers(FDescSetBatch,
    [AMatrixBuf, AWeightBuf]);

  LPush.HiddenDim := AHiddenDim;
  LPush.Eps := FEpsilon;
  LPush.NumTokens := ANumTokens;

  FCompute.DispatchComputeWithPush(
    FBundleBatch.Pipeline,
    FBundleBatch.PipelineLayout,
    FDescSetBatch,
    @LPush,
    SizeOf(LPush),
    ANumTokens  // one workgroup per token row
  );
end;

procedure TVdxLayerNorm.ApplyCopyBatch(const ASourceBuf: TVdxGpuBuffer;
  const AWeightBuf: TVdxGpuBuffer; const ADestBuf: TVdxGpuBuffer;
  const AHiddenDim: UInt32; const ANumTokens: UInt32);
var
  LPush: TVdxRMSNormBatchPush;
begin
  FCompute.UpdateDescriptorSetBuffers(FDescSetCopyBatch,
    [ASourceBuf, AWeightBuf, ADestBuf]);

  LPush.HiddenDim := AHiddenDim;
  LPush.Eps := FEpsilon;
  LPush.NumTokens := ANumTokens;

  FCompute.DispatchComputeWithPush(
    FBundleCopyBatch.Pipeline,
    FBundleCopyBatch.PipelineLayout,
    FDescSetCopyBatch,
    @LPush,
    SizeOf(LPush),
    ANumTokens  // one workgroup per token row
  );
end;

procedure TVdxLayerNorm.ApplyAdd(const AInputBuf: TVdxGpuBuffer;
  const AWeightBuf: TVdxGpuBuffer; const AAccumBuf: TVdxGpuBuffer;
  const AHiddenDim: UInt32);
var
  LPush: TVdxRMSNormPush;
begin
  // Rebind buffers: input (readonly), weight (readonly), accumulator (readwrite)
  FCompute.UpdateDescriptorSetBuffers(FDescSetAdd,
    [AInputBuf, AWeightBuf, AAccumBuf]);

  LPush.HiddenDim := AHiddenDim;
  LPush.Eps := FEpsilon;

  // Single workgroup — reads input, norms, adds to accumulator
  FCompute.DispatchComputeWithPush(
    FBundleAdd.Pipeline,
    FBundleAdd.PipelineLayout,
    FDescSetAdd,
    @LPush,
    SizeOf(LPush),
    1  // 1 workgroup
  );
end;

procedure TVdxLayerNorm.ApplyAddBatch(const AInputBuf: TVdxGpuBuffer;
  const AWeightBuf: TVdxGpuBuffer; const AAccumBuf: TVdxGpuBuffer;
  const AHiddenDim: UInt32; const ANumTokens: UInt32);
var
  LPush: TVdxRMSNormBatchPush;
begin
  FCompute.UpdateDescriptorSetBuffers(FDescSetAddBatch,
    [AInputBuf, AWeightBuf, AAccumBuf]);

  LPush.HiddenDim := AHiddenDim;
  LPush.Eps := FEpsilon;
  LPush.NumTokens := ANumTokens;

  FCompute.DispatchComputeWithPush(
    FBundleAddBatch.Pipeline,
    FBundleAddBatch.PipelineLayout,
    FDescSetAddBatch,
    @LPush,
    SizeOf(LPush),
    ANumTokens  // one workgroup per token row
  );
end;

procedure TVdxLayerNorm.UploadNormWeights(const AReader: TVdxGGUFReader;
  const ALayerIndex: Integer; out AWeights: TVdxNormLayerWeights);

  // Upload a single F32 norm tensor from GGUF to device-local GPU buffer
  function UploadOneTensor(const ATensorName: string): TVdxGpuBuffer;
  var
    LInfo: TVdxGGUFTensorInfo;
    LPtr: Pointer;
    LSize: UInt64;
    LStaging: TVdxGpuBuffer;
  begin
    Result := Default(TVdxGpuBuffer);

    if not AReader.GetTensorInfo(ATensorName, LInfo) then
    begin
      FErrors.Add(esError, 'LN01',
        'LayerNorm: tensor not found: %s', [ATensorName]);
      Exit;
    end;

    if LInfo.TensorType <> gtF32 then
    begin
      FErrors.Add(esError, 'LN02',
        'LayerNorm: %s expected F32 but got %s',
        [ATensorName, VdxGGMLTypeName(LInfo.TensorType)]);
      Exit;
    end;

    LPtr := AReader.GetTensorDataPtr(ATensorName);
    LSize := LInfo.Dimensions[0] * SizeOf(Single);

    LStaging := FCompute.CreateGpuBuffer(
      LSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    try
      FCompute.UploadToBuffer(LStaging, LPtr, LSize);

      Result := FCompute.CreateGpuBuffer(
        LSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
      );

      FCompute.CopyBuffer(LStaging, Result, LSize);
    finally
      FCompute.DestroyGpuBuffer(LStaging);
    end;
  end;

begin
  AWeights := Default(TVdxNormLayerWeights);

  AWeights.AttnNormGpu := UploadOneTensor(
    Format('blk.%d.attn_norm.weight', [ALayerIndex]));
  if FErrors.HasErrors() then Exit;
  AWeights.PostAttnNormGpu := UploadOneTensor(
    Format('blk.%d.post_attention_norm.weight', [ALayerIndex]));
  if FErrors.HasErrors() then Exit;
  AWeights.FFNNormGpu := UploadOneTensor(
    Format('blk.%d.ffn_norm.weight', [ALayerIndex]));
  if FErrors.HasErrors() then Exit;
  AWeights.PostFFNNormGpu := UploadOneTensor(
    Format('blk.%d.post_ffw_norm.weight', [ALayerIndex]));
  if FErrors.HasErrors() then Exit;
  AWeights.QNormGpu := UploadOneTensor(
    Format('blk.%d.attn_q_norm.weight', [ALayerIndex]));
  if FErrors.HasErrors() then Exit;
  AWeights.KNormGpu := UploadOneTensor(
    Format('blk.%d.attn_k_norm.weight', [ALayerIndex]));

  Status('LayerNorm: Uploaded 6 norm weights for layer %d', [ALayerIndex]);
end;

procedure TVdxLayerNorm.FreeNormWeights(var AWeights: TVdxNormLayerWeights);
begin
  if AWeights.AttnNormGpu.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(AWeights.AttnNormGpu);

  if AWeights.PostAttnNormGpu.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(AWeights.PostAttnNormGpu);

  if AWeights.FFNNormGpu.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(AWeights.FFNNormGpu);

  if AWeights.PostFFNNormGpu.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(AWeights.PostFFNNormGpu);

  if AWeights.QNormGpu.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(AWeights.QNormGpu);

  if AWeights.KNormGpu.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(AWeights.KNormGpu);
end;

end.
