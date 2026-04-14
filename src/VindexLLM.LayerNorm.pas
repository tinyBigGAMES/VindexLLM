{===============================================================================
  VindexLLM™ - Graph-Walk LLM Inference Engine

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
  VindexLLM.VulkanCompute;

type

  // Push constant layout matching rmsnorm.comp
  TVdxRMSNormPush = record
    HiddenDim: UInt32;
    Eps: Single;
  end;

  // Per-layer norm weight GPU buffers (Gemma 3 sandwich norm pattern)
  TVdxNormLayerWeights = record
    AttnNormGpu: TVdxGpuBuffer;       // F32 x HiddenDim, pre-attention norm
    PostAttnNormGpu: TVdxGpuBuffer;   // F32 x HiddenDim, post-attention norm
    FFNNormGpu: TVdxGpuBuffer;        // F32 x HiddenDim, pre-FFN norm
    PostFFNNormGpu: TVdxGpuBuffer;    // F32 x HiddenDim, post-FFN norm
    QNormGpu: TVdxGpuBuffer;          // F32 x HeadDim (256), QK-norm on queries
    KNormGpu: TVdxGpuBuffer;          // F32 x HeadDim (256), QK-norm on keys
  end;

  { TVdxLayerNorm }
  TVdxLayerNorm = class(TVdxStatusObject)
  private
    FCompute: TVdxVulkanCompute;
    FShaderModule: VkShaderModule;
    FBundle: TVdxComputePipelineBundle;
    FDescLayout: VkDescriptorSetLayout;
    FEpsilon: Single;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Initialize shader pipeline (call after setting status callback)
    procedure Init(const ACompute: TVdxVulkanCompute;
      const AEpsilon: Single = 1e-6);

    // Release GPU resources
    procedure Cleanup();

    // Apply RMSNorm in-place on residual using weight buffer
    procedure Apply(const AResidualBuf: TVdxGpuBuffer;
      const AWeightBuf: TVdxGpuBuffer; const AHiddenDim: UInt32);

    // Upload attn_norm + ffn_norm weights from GGUF to GPU for one layer
    procedure UploadNormWeights(const AReader: TVdxGGUFReader;
      const ALayerIndex: Integer; out AWeights: TVdxNormLayerWeights);

    // Free GPU buffers for one layer's norm weights
    procedure FreeNormWeights(var AWeights: TVdxNormLayerWeights);
  end;

implementation

uses
  System.IOUtils;

// ============================================================================
//  TVdxLayerNorm — Construction / Destruction
// ============================================================================

constructor TVdxLayerNorm.Create();
begin
  inherited;

  FCompute := nil;
  FShaderModule := VK_NULL_HANDLE;
  FBundle.Pipeline := VK_NULL_HANDLE;
  FBundle.PipelineLayout := VK_NULL_HANDLE;
  FDescLayout := VK_NULL_HANDLE;
  FEpsilon := 1e-6;
end;

destructor TVdxLayerNorm.Destroy();
begin
  if FCompute <> nil then
    Cleanup();

  inherited;
end;

// ============================================================================
//  TVdxLayerNorm — Init: Load shader, create pipeline
// ============================================================================

procedure TVdxLayerNorm.Init(const ACompute: TVdxVulkanCompute;
  const AEpsilon: Single);
var
  LSpvPath: string;
  LSpvData: TBytes;
begin
  FCompute := ACompute;
  FEpsilon := AEpsilon;

  Status('LayerNorm: Init (eps=%e)', [Double(FEpsilon)]);

  // Load rmsnorm shader from .spv file
  LSpvPath := TPath.Combine(
    TPath.GetDirectoryName(ParamStr(0)),
    '..\shaders\rmsnorm.spv'
  );
  LSpvPath := TPath.GetFullPath(LSpvPath);
  TVdxUtils.FailIf(not TFile.Exists(LSpvPath),
    'rmsnorm.spv not found: %s', [LSpvPath]);
  LSpvData := TFile.ReadAllBytes(LSpvPath);
  FShaderModule := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  // Descriptor layout: binding 0=residual, 1=weight
  FDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);

  // Pipeline with push constants (hidden_dim + eps)
  FBundle := FCompute.CreateComputePipelineWithPush(
    FShaderModule, 'main', FDescLayout, SizeOf(TVdxRMSNormPush));

  Status('LayerNorm: Ready');
end;

// ============================================================================
//  TVdxLayerNorm — Cleanup
// ============================================================================

procedure TVdxLayerNorm.Cleanup();
begin
  if FCompute = nil then
    Exit;

  FCompute.DestroyComputePipelineBundle(FBundle);
  FCompute.DestroyDescriptorSetLayoutHandle(FDescLayout);
  FDescLayout := VK_NULL_HANDLE;

  FCompute.DestroyShaderModuleHandle(FShaderModule);
  FShaderModule := VK_NULL_HANDLE;

  FCompute := nil;
end;

// ============================================================================
//  TVdxLayerNorm — Apply: RMSNorm in-place on residual
// ============================================================================

procedure TVdxLayerNorm.Apply(const AResidualBuf: TVdxGpuBuffer;
  const AWeightBuf: TVdxGpuBuffer; const AHiddenDim: UInt32);
var
  LPush: TVdxRMSNormPush;
  LDescPool: VkDescriptorPool;
  LDescSet: VkDescriptorSet;
begin
  // Create temporary descriptor pool + set for this dispatch
  LDescPool := FCompute.CreateDescriptorPoolForStorage(1, 2);
  try
    LDescSet := FCompute.AllocateDescriptorSetForBuffers(
      LDescPool, FDescLayout,
      [AResidualBuf, AWeightBuf]
    );

    LPush.HiddenDim := AHiddenDim;
    LPush.Eps := FEpsilon;

    // Single workgroup — all 2560 elements handled by 256 threads
    FCompute.DispatchComputeWithPush(
      FBundle.Pipeline,
      FBundle.PipelineLayout,
      LDescSet,
      @LPush,
      SizeOf(LPush),
      1  // 1 workgroup
    );
  finally
    FCompute.DestroyDescriptorPoolHandle(LDescPool);
  end;
end;

// ============================================================================
//  TVdxLayerNorm — Upload Norm Weights from GGUF
// ============================================================================

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
    TVdxUtils.FailIf(not AReader.HasTensor(ATensorName),
      'LayerNorm: tensor not found: %s', [ATensorName]);

    LInfo := AReader.GetTensorInfo(ATensorName);
    TVdxUtils.FailIf(LInfo.TensorType <> gtF32,
      'LayerNorm: %s expected F32 but got %s',
      [ATensorName, VdxGGMLTypeName(LInfo.TensorType)]);

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
  AWeights.PostAttnNormGpu := UploadOneTensor(
    Format('blk.%d.post_attention_norm.weight', [ALayerIndex]));
  AWeights.FFNNormGpu := UploadOneTensor(
    Format('blk.%d.ffn_norm.weight', [ALayerIndex]));
  AWeights.PostFFNNormGpu := UploadOneTensor(
    Format('blk.%d.post_ffw_norm.weight', [ALayerIndex]));
  AWeights.QNormGpu := UploadOneTensor(
    Format('blk.%d.attn_q_norm.weight', [ALayerIndex]));
  AWeights.KNormGpu := UploadOneTensor(
    Format('blk.%d.attn_k_norm.weight', [ALayerIndex]));

  Status('LayerNorm: Uploaded 6 norm weights for layer %d', [ALayerIndex]);
end;

// ============================================================================
//  TVdxLayerNorm — Free Norm Weight GPU Buffers
// ============================================================================

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
