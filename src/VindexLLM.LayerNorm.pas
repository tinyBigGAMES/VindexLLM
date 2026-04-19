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
  VindexLLM.Resources,
  VindexLLM.GGUFReader,
  VindexLLM.Vulkan,
  VindexLLM.Compute;

const

  //--------------------------------------------------------------------------
  // Error Codes
  //--------------------------------------------------------------------------
  VDX_ERROR_LN_NOT_INIT           = 'LN01';
  VDX_ERROR_LN_ALREADY_INIT       = 'LN02';
  VDX_ERROR_LN_COMPUTE_NIL        = 'LN03';
  VDX_ERROR_LN_INIT_EXCEPTION     = 'LN04';
  VDX_ERROR_LN_TENSOR_NOT_FOUND   = 'LN05';
  VDX_ERROR_LN_TENSOR_WRONG_TYPE  = 'LN06';
  VDX_ERROR_LN_UPLOAD_EXCEPTION   = 'LN07';

type

  { TVdxRMSNormPush }
  // Push-constant struct matching rmsnorm.comp / rmsnorm_copy.comp.
  TVdxRMSNormPush = record
    HiddenDim: UInt32;
    Eps: Single;
  end;

  { TVdxRMSNormBatchPush }
  // Push-constant struct matching rmsnorm_batch.comp /
  // rmsnorm_copy_batch.comp. NumTokens selects which row this
  // workgroup processes.
  TVdxRMSNormBatchPush = record
    HiddenDim: UInt32;
    Eps: Single;
    NumTokens: UInt32;
  end;

  { TVdxNormLayerWeights }
  // All six per-layer norm tensors for a Gemma-3 block. These are F32
  // 1-D vectors, stay resident in DEVICE_LOCAL VRAM for the lifetime
  // of the model (principle: streaming only targets attention + FFN).
  TVdxNormLayerWeights = record
    AttnNormGpu:     TVdxGpuBuffer;  // [HiddenDim] pre-attention norm
    PostAttnNormGpu: TVdxGpuBuffer;  // [HiddenDim] post-attention norm
    FFNNormGpu:      TVdxGpuBuffer;  // [HiddenDim] pre-FFN norm
    PostFFNNormGpu:  TVdxGpuBuffer;  // [HiddenDim] post-FFN norm
    QNormGpu:        TVdxGpuBuffer;  // [HeadDim]   QK-norm on queries
    KNormGpu:        TVdxGpuBuffer;  // [HeadDim]   QK-norm on keys
  end;

  { TVdxLayerNorm }
  // RMSNorm compute primitive + norm-weight uploader. Owns four
  // pipelines (single in-place, single fused copy+norm, batched
  // in-place, batched fused copy+norm) and their descriptor pools.
  //
  // TVdxCompute is passed in at Init — this class does NOT own it.
  // The test (or TVdxModel once it lands in Phase 12) is expected to
  // call SetErrors(FCompute.GetErrors()) beforehand so failures from
  // underlying compute calls land in a single shared FErrors buffer.
  TVdxLayerNorm = class(TVdxBaseObject)
  private
    FCompute: TVdxCompute;
    FEpsilon: Single;
    FInitialized: Boolean;

    // Pipeline A: RMSNORM (single token, in-place, 2 bindings)
    FShaderModule: VkShaderModule;
    FBundle:       TVdxComputePipelineBundle;
    FDescLayout:   VkDescriptorSetLayout;
    FDescPool:     VkDescriptorPool;
    FDescSet:      VkDescriptorSet;

    // Pipeline B: RMSNORM_COPY (single token, fused src->dst, 3 bindings)
    FShaderModuleCopy: VkShaderModule;
    FBundleCopy:       TVdxComputePipelineBundle;
    FDescLayoutCopy:   VkDescriptorSetLayout;
    FDescPoolCopy:     VkDescriptorPool;
    FDescSetCopy:      VkDescriptorSet;

    // Pipeline C: RMSNORM_BATCH (N tokens, in-place, reuses FDescLayout)
    FShaderModuleBatch: VkShaderModule;
    FBundleBatch:       TVdxComputePipelineBundle;
    FDescPoolBatch:     VkDescriptorPool;
    FDescSetBatch:      VkDescriptorSet;

    // Pipeline D: RMSNORM_COPY_BATCH (N tokens, fused, reuses FDescLayoutCopy)
    FShaderModuleCopyBatch: VkShaderModule;
    FBundleCopyBatch:       TVdxComputePipelineBundle;
    FDescPoolCopyBatch:     VkDescriptorPool;
    FDescSetCopyBatch:      VkDescriptorSet;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // One-shot pipeline build. Returns False + populates FErrors on
    // failure, calling Cleanup() before returning so no partial GPU
    // state leaks. AEpsilon is retained and used by every Apply* call.
    function Init(const ACompute: TVdxCompute;
      const AEpsilon: Single = 1e-6): Boolean;

    // Idempotent teardown. Safe to call multiple times and safe after
    // partial Init failure — every destroy call is guarded by a
    // VK_NULL_HANDLE check.
    procedure Cleanup();

    // --- Dispatch primitives ---
    // Each returns True on a successfully recorded dispatch, False on
    // any underlying compute-layer failure (with FErrors populated via
    // the shared error buffer). Callers should bail on the first False.

    // In-place RMSNorm on a single HiddenDim-length vector. AResidualBuf
    // is read, normalized, and written back.
    function Apply(const AResidualBuf: TVdxGpuBuffer;
      const AWeightBuf: TVdxGpuBuffer;
      const AHiddenDim: UInt32): Boolean;

    // Fused copy+norm: reads ASourceBuf, writes normalized result to
    // ADestBuf. Source is untouched.
    function ApplyCopy(const ASourceBuf: TVdxGpuBuffer;
      const AWeightBuf: TVdxGpuBuffer;
      const ADestBuf: TVdxGpuBuffer;
      const AHiddenDim: UInt32): Boolean;

    // Batched in-place RMSNorm on a [NumTokens x HiddenDim] matrix.
    // Each row is normalized independently.
    function ApplyBatch(const AMatrixBuf: TVdxGpuBuffer;
      const AWeightBuf: TVdxGpuBuffer;
      const AHiddenDim: UInt32;
      const ANumTokens: UInt32): Boolean;

    // Batched fused copy+norm on two [NumTokens x HiddenDim] matrices.
    function ApplyCopyBatch(const ASourceBuf: TVdxGpuBuffer;
      const AWeightBuf: TVdxGpuBuffer;
      const ADestBuf: TVdxGpuBuffer;
      const AHiddenDim: UInt32;
      const ANumTokens: UInt32): Boolean;

    // --- GGUF weight upload ---

    // Uploads all six Gemma-3 norm tensors for one block index from
    // GGUF metadata to DEVICE_LOCAL VRAM. On any failure, rolls back
    // by calling FreeNormWeights on the out record and returns False.
    function UploadNormWeights(const AReader: TVdxGGUFReader;
      const ALayerIndex: Integer;
      out AWeights: TVdxNormLayerWeights): Boolean;

    // Destroys every non-null GPU buffer handle inside AWeights and
    // zeros the record. Nil-safe re: unset handles, safe to call on
    // partially-populated records.
    procedure FreeNormWeights(var AWeights: TVdxNormLayerWeights);

    property Epsilon: Single read FEpsilon;
    property Initialized: Boolean read FInitialized;
  end;

implementation

uses
  VindexLLM.Shaders;

{ TVdxLayerNorm }

constructor TVdxLayerNorm.Create();
begin
  inherited Create();

  FCompute := nil;
  FEpsilon := 1e-6;
  FInitialized := False;

  FShaderModule := VK_NULL_HANDLE;
  FBundle       := Default(TVdxComputePipelineBundle);
  FDescLayout   := VK_NULL_HANDLE;
  FDescPool     := VK_NULL_HANDLE;
  FDescSet      := VK_NULL_HANDLE;

  FShaderModuleCopy := VK_NULL_HANDLE;
  FBundleCopy       := Default(TVdxComputePipelineBundle);
  FDescLayoutCopy   := VK_NULL_HANDLE;
  FDescPoolCopy     := VK_NULL_HANDLE;
  FDescSetCopy      := VK_NULL_HANDLE;

  FShaderModuleBatch := VK_NULL_HANDLE;
  FBundleBatch       := Default(TVdxComputePipelineBundle);
  FDescPoolBatch     := VK_NULL_HANDLE;
  FDescSetBatch      := VK_NULL_HANDLE;

  FShaderModuleCopyBatch := VK_NULL_HANDLE;
  FBundleCopyBatch       := Default(TVdxComputePipelineBundle);
  FDescPoolCopyBatch     := VK_NULL_HANDLE;
  FDescSetCopyBatch      := VK_NULL_HANDLE;
end;

destructor TVdxLayerNorm.Destroy();
begin
  Cleanup();
  inherited;
end;

function TVdxLayerNorm.Init(const ACompute: TVdxCompute;
  const AEpsilon: Single): Boolean;
var
  LSpvData: TBytes;
  LDummyBuf: TVdxGpuBuffer;
begin
  Result := False;

  if FInitialized then
  begin
    FErrors.Add(esError, VDX_ERROR_LN_ALREADY_INIT, RSLNAlreadyInit);
    Exit;
  end;

  if ACompute = nil then
  begin
    FErrors.Add(esFatal, VDX_ERROR_LN_COMPUTE_NIL, RSLNComputeNil);
    Exit;
  end;

  FCompute := ACompute;
  FEpsilon := AEpsilon;
  LDummyBuf := Default(TVdxGpuBuffer);

  Status('LayerNorm: Init (eps=%e)', [Double(FEpsilon)]);

  // Outer try/finally guarantees Cleanup runs on any early-exit.
  // Inner try/except converts raises from VdxLoadShader or any other
  // RTL call into esFatal errors without propagating past this boundary.
  try
    try
      //
      // Pipeline A: RMSNORM (single-token, in-place)
      //   bindings: 0=residual (rw), 1=weight (ro)
      //
      LSpvData := VdxLoadShader('RMSNORM');
      FShaderModule := FCompute.CreateShaderModule(
        @LSpvData[0], NativeUInt(Length(LSpvData)));
      if FErrors.HasFatal() then Exit;

      FDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
      if FErrors.HasFatal() then Exit;

      FBundle := FCompute.CreateComputePipelineWithPush(
        FShaderModule, 'main', FDescLayout, SizeOf(TVdxRMSNormPush));
      if FErrors.HasFatal() then Exit;

      FDescPool := FCompute.CreateDescriptorPoolForStorage(1, 2);
      if FErrors.HasFatal() then Exit;

      FDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FDescPool, FDescLayout, [LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;

      //
      // Pipeline B: RMSNORM_COPY (single-token, fused src->dst)
      //   bindings: 0=source (ro), 1=weight (ro), 2=dest (wo)
      //
      LSpvData := VdxLoadShader('RMSNORM_COPY');
      FShaderModuleCopy := FCompute.CreateShaderModule(
        @LSpvData[0], NativeUInt(Length(LSpvData)));
      if FErrors.HasFatal() then Exit;

      FDescLayoutCopy := FCompute.CreateStorageDescriptorSetLayout(3);
      if FErrors.HasFatal() then Exit;

      FBundleCopy := FCompute.CreateComputePipelineWithPush(
        FShaderModuleCopy, 'main', FDescLayoutCopy,
        SizeOf(TVdxRMSNormPush));
      if FErrors.HasFatal() then Exit;

      FDescPoolCopy := FCompute.CreateDescriptorPoolForStorage(1, 3);
      if FErrors.HasFatal() then Exit;

      FDescSetCopy := FCompute.AllocateDescriptorSetForBuffers(
        FDescPoolCopy, FDescLayoutCopy,
        [LDummyBuf, LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;

      //
      // Pipeline C: RMSNORM_BATCH (N-token, in-place)
      //   reuses FDescLayout (2 bindings: matrix, weight)
      //
      LSpvData := VdxLoadShader('RMSNORM_BATCH');
      FShaderModuleBatch := FCompute.CreateShaderModule(
        @LSpvData[0], NativeUInt(Length(LSpvData)));
      if FErrors.HasFatal() then Exit;

      FBundleBatch := FCompute.CreateComputePipelineWithPush(
        FShaderModuleBatch, 'main', FDescLayout,
        SizeOf(TVdxRMSNormBatchPush));
      if FErrors.HasFatal() then Exit;

      FDescPoolBatch := FCompute.CreateDescriptorPoolForStorage(1, 2);
      if FErrors.HasFatal() then Exit;

      FDescSetBatch := FCompute.AllocateDescriptorSetForBuffers(
        FDescPoolBatch, FDescLayout, [LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;

      //
      // Pipeline D: RMSNORM_COPY_BATCH (N-token, fused)
      //   reuses FDescLayoutCopy (3 bindings: source, weight, dest)
      //
      LSpvData := VdxLoadShader('RMSNORM_COPY_BATCH');
      FShaderModuleCopyBatch := FCompute.CreateShaderModule(
        @LSpvData[0], NativeUInt(Length(LSpvData)));
      if FErrors.HasFatal() then Exit;

      FBundleCopyBatch := FCompute.CreateComputePipelineWithPush(
        FShaderModuleCopyBatch, 'main', FDescLayoutCopy,
        SizeOf(TVdxRMSNormBatchPush));
      if FErrors.HasFatal() then Exit;

      FDescPoolCopyBatch := FCompute.CreateDescriptorPoolForStorage(1, 3);
      if FErrors.HasFatal() then Exit;

      FDescSetCopyBatch := FCompute.AllocateDescriptorSetForBuffers(
        FDescPoolCopyBatch, FDescLayoutCopy,
        [LDummyBuf, LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;

      FInitialized := True;
      Result := True;
      Status('LayerNorm: Ready');
    except
      on E: Exception do
        FErrors.Add(esFatal, VDX_ERROR_LN_INIT_EXCEPTION,
          RSLNInitException, [E.Message]);
    end;
  finally
    if not Result then
      Cleanup();
  end;
end;

procedure TVdxLayerNorm.Cleanup();
begin
  // Early-out if Cleanup runs before Compute was set. Every handle
  // below is checked against VK_NULL_HANDLE to keep this safe after
  // partial Init failure.
  if FCompute = nil then
  begin
    FInitialized := False;
    Exit;
  end;

  // --- Pipeline D: RMSNORM_COPY_BATCH ---
  if FBundleCopyBatch.Pipeline <> VK_NULL_HANDLE then
    FCompute.DestroyComputePipelineBundle(FBundleCopyBatch);
  if FDescPoolCopyBatch <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FDescPoolCopyBatch);
    FDescPoolCopyBatch := VK_NULL_HANDLE;
  end;
  if FShaderModuleCopyBatch <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FShaderModuleCopyBatch);
    FShaderModuleCopyBatch := VK_NULL_HANDLE;
  end;
  FDescSetCopyBatch := VK_NULL_HANDLE;

  // --- Pipeline C: RMSNORM_BATCH ---
  if FBundleBatch.Pipeline <> VK_NULL_HANDLE then
    FCompute.DestroyComputePipelineBundle(FBundleBatch);
  if FDescPoolBatch <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FDescPoolBatch);
    FDescPoolBatch := VK_NULL_HANDLE;
  end;
  if FShaderModuleBatch <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FShaderModuleBatch);
    FShaderModuleBatch := VK_NULL_HANDLE;
  end;
  FDescSetBatch := VK_NULL_HANDLE;

  // --- Pipeline B: RMSNORM_COPY ---
  if FBundleCopy.Pipeline <> VK_NULL_HANDLE then
    FCompute.DestroyComputePipelineBundle(FBundleCopy);
  if FDescPoolCopy <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FDescPoolCopy);
    FDescPoolCopy := VK_NULL_HANDLE;
  end;
  if FDescLayoutCopy <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorSetLayoutHandle(FDescLayoutCopy);
    FDescLayoutCopy := VK_NULL_HANDLE;
  end;
  if FShaderModuleCopy <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FShaderModuleCopy);
    FShaderModuleCopy := VK_NULL_HANDLE;
  end;
  FDescSetCopy := VK_NULL_HANDLE;

  // --- Pipeline A: RMSNORM ---
  if FBundle.Pipeline <> VK_NULL_HANDLE then
    FCompute.DestroyComputePipelineBundle(FBundle);
  if FDescPool <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FDescPool);
    FDescPool := VK_NULL_HANDLE;
  end;
  if FDescLayout <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorSetLayoutHandle(FDescLayout);
    FDescLayout := VK_NULL_HANDLE;
  end;
  if FShaderModule <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FShaderModule);
    FShaderModule := VK_NULL_HANDLE;
  end;
  FDescSet := VK_NULL_HANDLE;

  FCompute := nil;
  FInitialized := False;
end;

function TVdxLayerNorm.Apply(const AResidualBuf: TVdxGpuBuffer;
  const AWeightBuf: TVdxGpuBuffer;
  const AHiddenDim: UInt32): Boolean;
var
  LPush: TVdxRMSNormPush;
begin
  Result := False;

  if not FInitialized then
  begin
    FErrors.Add(esError, VDX_ERROR_LN_NOT_INIT, RSLNNotInit);
    Exit;
  end;

  FCompute.UpdateDescriptorSetBuffers(FDescSet,
    [AResidualBuf, AWeightBuf]);
  if FErrors.HasFatal() then Exit;

  LPush.HiddenDim := AHiddenDim;
  LPush.Eps       := FEpsilon;

  // Single workgroup — shader uses 256 threads to cover all elements.
  if not FCompute.DispatchComputeWithPush(
    FBundle.Pipeline, FBundle.PipelineLayout, FDescSet,
    @LPush, SizeOf(LPush), 1) then Exit;

  Result := True;
end;

function TVdxLayerNorm.ApplyCopy(const ASourceBuf: TVdxGpuBuffer;
  const AWeightBuf: TVdxGpuBuffer;
  const ADestBuf: TVdxGpuBuffer;
  const AHiddenDim: UInt32): Boolean;
var
  LPush: TVdxRMSNormPush;
begin
  Result := False;

  if not FInitialized then
  begin
    FErrors.Add(esError, VDX_ERROR_LN_NOT_INIT, RSLNNotInit);
    Exit;
  end;

  FCompute.UpdateDescriptorSetBuffers(FDescSetCopy,
    [ASourceBuf, AWeightBuf, ADestBuf]);
  if FErrors.HasFatal() then Exit;

  LPush.HiddenDim := AHiddenDim;
  LPush.Eps       := FEpsilon;

  if not FCompute.DispatchComputeWithPush(
    FBundleCopy.Pipeline, FBundleCopy.PipelineLayout, FDescSetCopy,
    @LPush, SizeOf(LPush), 1) then Exit;

  Result := True;
end;

function TVdxLayerNorm.ApplyBatch(const AMatrixBuf: TVdxGpuBuffer;
  const AWeightBuf: TVdxGpuBuffer;
  const AHiddenDim: UInt32;
  const ANumTokens: UInt32): Boolean;
var
  LPush: TVdxRMSNormBatchPush;
begin
  Result := False;

  if not FInitialized then
  begin
    FErrors.Add(esError, VDX_ERROR_LN_NOT_INIT, RSLNNotInit);
    Exit;
  end;

  if ANumTokens = 0 then
  begin
    Result := True;  // no-op is a valid outcome
    Exit;
  end;

  FCompute.UpdateDescriptorSetBuffers(FDescSetBatch,
    [AMatrixBuf, AWeightBuf]);
  if FErrors.HasFatal() then Exit;

  LPush.HiddenDim := AHiddenDim;
  LPush.Eps       := FEpsilon;
  LPush.NumTokens := ANumTokens;

  // One workgroup per token row.
  if not FCompute.DispatchComputeWithPush(
    FBundleBatch.Pipeline, FBundleBatch.PipelineLayout, FDescSetBatch,
    @LPush, SizeOf(LPush), ANumTokens) then Exit;

  Result := True;
end;

function TVdxLayerNorm.ApplyCopyBatch(const ASourceBuf: TVdxGpuBuffer;
  const AWeightBuf: TVdxGpuBuffer;
  const ADestBuf: TVdxGpuBuffer;
  const AHiddenDim: UInt32;
  const ANumTokens: UInt32): Boolean;
var
  LPush: TVdxRMSNormBatchPush;
begin
  Result := False;

  if not FInitialized then
  begin
    FErrors.Add(esError, VDX_ERROR_LN_NOT_INIT, RSLNNotInit);
    Exit;
  end;

  if ANumTokens = 0 then
  begin
    Result := True;
    Exit;
  end;

  FCompute.UpdateDescriptorSetBuffers(FDescSetCopyBatch,
    [ASourceBuf, AWeightBuf, ADestBuf]);
  if FErrors.HasFatal() then Exit;

  LPush.HiddenDim := AHiddenDim;
  LPush.Eps       := FEpsilon;
  LPush.NumTokens := ANumTokens;

  if not FCompute.DispatchComputeWithPush(
    FBundleCopyBatch.Pipeline, FBundleCopyBatch.PipelineLayout,
    FDescSetCopyBatch,
    @LPush, SizeOf(LPush), ANumTokens) then Exit;

  Result := True;
end;

function TVdxLayerNorm.UploadNormWeights(const AReader: TVdxGGUFReader;
  const ALayerIndex: Integer;
  out AWeights: TVdxNormLayerWeights): Boolean;

  // Uploads one F32 1-D tensor to a fresh DEVICE_LOCAL buffer via a
  // HOST_VISIBLE staging buffer. Returns False + populates FErrors on
  // any failure; on success ABuf holds a valid GPU buffer tagged
  // vcWeight for VRAM accounting.
  function UploadOneTensor(const ATensorName: string;
    out ABuf: TVdxGpuBuffer): Boolean;
  var
    LInfo: TVdxGGUFTensorInfo;
    LPtr: PByte;
    LSize: UInt64;
    LStaging: TVdxGpuBuffer;
  begin
    Result   := False;
    ABuf     := Default(TVdxGpuBuffer);
    LStaging := Default(TVdxGpuBuffer);

    if not AReader.HasTensor(ATensorName) then
    begin
      FErrors.Add(esFatal, VDX_ERROR_LN_TENSOR_NOT_FOUND,
        RSLNTensorNotFound, [ATensorName]);
      Exit;
    end;

    if not AReader.GetTensorInfo(ATensorName, LInfo) then
      Exit;  // reader already logged

    if LInfo.TensorType <> gtF32 then
    begin
      FErrors.Add(esFatal, VDX_ERROR_LN_TENSOR_WRONG_TYPE,
        RSLNTensorWrongType,
        [ATensorName, VdxGGMLTypeName(LInfo.TensorType)]);
      Exit;
    end;

    LPtr := AReader.GetTensorDataPtr(ATensorName);
    if LPtr = nil then
      Exit;  // reader already logged

    LSize := LInfo.Dimensions[0] * SizeOf(Single);

    // Staging: HOST_VISIBLE for CPU write, TRANSFER_SRC for GPU copy.
    LStaging := FCompute.CreateGpuBuffer(
      LSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      vcBuffer);
    if FErrors.HasFatal() or (LStaging.Buffer = VK_NULL_HANDLE) then Exit;

    try
      if not FCompute.UploadToBuffer(LStaging, LPtr, LSize) then Exit;

      // Destination: DEVICE_LOCAL for fast GPU access, tagged vcWeight
      // so VRAM accounting attributes this to the weights bucket.
      ABuf := FCompute.CreateGpuBuffer(
        LSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vcWeight);
      if FErrors.HasFatal() or (ABuf.Buffer = VK_NULL_HANDLE) then Exit;

      if not FCompute.CopyBuffer(LStaging, ABuf, LSize) then Exit;

      Result := True;
    finally
      FCompute.DestroyGpuBuffer(LStaging);
    end;
  end;

begin
  Result := False;
  AWeights := Default(TVdxNormLayerWeights);

  if not FInitialized then
  begin
    FErrors.Add(esError, VDX_ERROR_LN_NOT_INIT, RSLNNotInit);
    Exit;
  end;

  if AReader = nil then
  begin
    FErrors.Add(esFatal, VDX_ERROR_LN_COMPUTE_NIL,
      'LayerNorm.UploadNormWeights: AReader is nil');
    Exit;
  end;

  // Protect against any unexpected raise from the reader or Vulkan
  // layer — we already try to route everything through Boolean
  // returns, but this is defense in depth.
  try
    if not UploadOneTensor(
      Format('blk.%d.attn_norm.weight', [ALayerIndex]),
      AWeights.AttnNormGpu) then
    begin
      FreeNormWeights(AWeights);
      Exit;
    end;

    if not UploadOneTensor(
      Format('blk.%d.post_attention_norm.weight', [ALayerIndex]),
      AWeights.PostAttnNormGpu) then
    begin
      FreeNormWeights(AWeights);
      Exit;
    end;

    if not UploadOneTensor(
      Format('blk.%d.ffn_norm.weight', [ALayerIndex]),
      AWeights.FFNNormGpu) then
    begin
      FreeNormWeights(AWeights);
      Exit;
    end;

    if not UploadOneTensor(
      Format('blk.%d.post_ffw_norm.weight', [ALayerIndex]),
      AWeights.PostFFNNormGpu) then
    begin
      FreeNormWeights(AWeights);
      Exit;
    end;

    if not UploadOneTensor(
      Format('blk.%d.attn_q_norm.weight', [ALayerIndex]),
      AWeights.QNormGpu) then
    begin
      FreeNormWeights(AWeights);
      Exit;
    end;

    if not UploadOneTensor(
      Format('blk.%d.attn_k_norm.weight', [ALayerIndex]),
      AWeights.KNormGpu) then
    begin
      FreeNormWeights(AWeights);
      Exit;
    end;

    Status('LayerNorm: Uploaded 6 norm weights for layer %d',
      [ALayerIndex]);
    Result := True;
  except
    on E: Exception do
    begin
      FErrors.Add(esFatal, VDX_ERROR_LN_UPLOAD_EXCEPTION,
        RSLNUploadException, [E.Message]);
      FreeNormWeights(AWeights);
    end;
  end;
end;

procedure TVdxLayerNorm.FreeNormWeights(var AWeights: TVdxNormLayerWeights);
begin
  if FCompute = nil then
  begin
    AWeights := Default(TVdxNormLayerWeights);
    Exit;
  end;

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

  AWeights := Default(TVdxNormLayerWeights);
end;

end.
