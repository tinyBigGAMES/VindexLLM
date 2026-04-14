{===============================================================================
  VindexLLM™ - Graph-Walk LLM Inference Engine

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Vindex;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Math,
  System.Generics.Collections,
  VindexLLM.Utils,
  VindexLLM.GGUFReader,
  VindexLLM.VulkanCompute;

type

  PUInt16 = ^UInt16;

  { TVdxFFNLayerView }
  TVdxFFNLayerView = record
    LayerIndex: Integer;
    GatePtr: Pointer;         // Zero-copy pointer into mmap'd GGUF
    UpPtr: Pointer;           // Zero-copy pointer into mmap'd GGUF (up projection)
    DownPtr: Pointer;         // Zero-copy pointer into mmap'd GGUF
    GateType: TVdxGGMLType;
    DownType: TVdxGGMLType;
    FeatureCount: UInt64;     // FFN width (10240 for Gemma 3 4B)
    HiddenDim: UInt64;        // Residual dim (2560 for Gemma 3 4B)
    GateSizeBytes: UInt64;    // Computed: FeatureCount * HiddenDim * ElementSize
    DownSizeBytes: UInt64;
    GateGpuBuffer: TVdxGpuBuffer;   // Populated after GPU upload
    DownGpuBuffer: TVdxGpuBuffer;
  end;

  { TVdxVindex }
  TVdxVindex = class(TVdxBaseObject)
  private
    FLayers: TArray<TVdxFFNLayerView>;
    FLayerCount: Integer;
    FHiddenDim: UInt64;
    FFFNWidth: UInt64;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Scan GGUF reader for FFN gate/down tensors, build layer-indexed view
    function BuildFromGGUF(const AReader: TVdxGGUFReader): Boolean;

    // Upload one layer's gate+down to device-local GPU memory via staging
    procedure UploadLayer(const ALayerIndex: Integer; const ACompute: TVdxVulkanCompute);

    // Upload all layers to GPU
    procedure UploadAll(const ACompute: TVdxVulkanCompute);

    // Free GPU buffers for one layer
    procedure FreeLayerGpu(const ALayerIndex: Integer; const ACompute: TVdxVulkanCompute);

    // Free all GPU buffers
    procedure FreeAllGpu(const ACompute: TVdxVulkanCompute);

    // Accessors
    function GetLayerCount(): Integer;
    function GetLayer(const AIndex: Integer): TVdxFFNLayerView;
    function GetHiddenDim(): UInt64;
    function GetFFNWidth(): UInt64;
  end;

implementation

// ============================================================================
//  TVdxVindex — Construction / Destruction
// ============================================================================

constructor TVdxVindex.Create();
begin
  inherited;

  FLayers := nil;
  FLayerCount := 0;
  FHiddenDim := 0;
  FFFNWidth := 0;
end;

destructor TVdxVindex.Destroy();
begin
  FLayers := nil;
  inherited;
end;

// ============================================================================
//  BuildFromGGUF — Scan tensors, build layer-indexed view
// ============================================================================

function TVdxVindex.BuildFromGGUF(const AReader: TVdxGGUFReader): Boolean;
var
  LTensorList: TList<TVdxGGUFTensorInfo>;
  LTensor: TVdxGGUFTensorInfo;
  LI: Integer;
  LLayerIdx: Integer;
  LMaxLayer: Integer;
  LGateName: string;
  LUpName: string;
  LDownName: string;
  LGateInfo: TVdxGGUFTensorInfo;
  LDownInfo: TVdxGGUFTensorInfo;
  LElementSize: UInt64;
  LLayer: TVdxFFNLayerView;
begin
  Result := False;

  // Find the highest layer index by scanning for gate tensors
  LTensorList := AReader.GetTensorList();
  LMaxLayer := -1;

  for LI := 0 to LTensorList.Count - 1 do
  begin
    LTensor := LTensorList[LI];
    // Match pattern: blk.N.ffn_gate.weight
    if LTensor.TensorName.StartsWith('blk.') and
       LTensor.TensorName.EndsWith('.ffn_gate.weight') then
    begin
      // Extract layer index from "blk.N.ffn_gate.weight"
      LLayerIdx := StrToIntDef(
        Copy(LTensor.TensorName, 5,
          Pos('.ffn_gate.weight', LTensor.TensorName) - 5),
        -1
      );
      if LLayerIdx > LMaxLayer then
        LMaxLayer := LLayerIdx;
    end;
  end;

  if LMaxLayer < 0 then
    Exit;

  // Allocate layer array
  FLayerCount := LMaxLayer + 1;
  SetLength(FLayers, FLayerCount);

  // Build each layer view
  for LI := 0 to FLayerCount - 1 do
  begin
    LGateName := Format('blk.%d.ffn_gate.weight', [LI]);
    LUpName := Format('blk.%d.ffn_up.weight', [LI]);
    LDownName := Format('blk.%d.ffn_down.weight', [LI]);

    if (not AReader.HasTensor(LGateName)) or
       (not AReader.HasTensor(LUpName)) or
       (not AReader.HasTensor(LDownName)) then
      Exit;

    LGateInfo := AReader.GetTensorInfo(LGateName);
    LDownInfo := AReader.GetTensorInfo(LDownName);

    LLayer := Default(TVdxFFNLayerView);
    LLayer.LayerIndex := LI;
    LLayer.GatePtr := AReader.GetTensorDataPtr(LGateName);
    LLayer.UpPtr := AReader.GetTensorDataPtr(LUpName);
    LLayer.DownPtr := AReader.GetTensorDataPtr(LDownName);
    LLayer.GateType := LGateInfo.TensorType;
    LLayer.DownType := LDownInfo.TensorType;

    // Gate tensor: [HiddenDim x FeatureCount]
    // Dimensions[0] = HiddenDim (2560), Dimensions[1] = FeatureCount (10240)
    LLayer.HiddenDim := LGateInfo.Dimensions[0];
    LLayer.FeatureCount := LGateInfo.Dimensions[1];

    // Compute byte sizes
    LElementSize := VdxGGMLTypeSize(LGateInfo.TensorType);
    LLayer.GateSizeBytes := LLayer.HiddenDim * LLayer.FeatureCount * LElementSize;

    LElementSize := VdxGGMLTypeSize(LDownInfo.TensorType);
    LLayer.DownSizeBytes := LDownInfo.Dimensions[0] * LDownInfo.Dimensions[1] * LElementSize;

    // GPU buffers start zeroed (not yet uploaded)
    FLayers[LI] := LLayer;
  end;

  // Store model dimensions from layer 0
  FHiddenDim := FLayers[0].HiddenDim;
  FFFNWidth := FLayers[0].FeatureCount;

  Result := True;
end;

// ============================================================================
//  GPU Upload — Staging buffer pattern
// ============================================================================

procedure TVdxVindex.UploadLayer(const ALayerIndex: Integer; const ACompute: TVdxVulkanCompute);
var
  LStaging: TVdxGpuBuffer;
begin
  TVdxUtils.FailIf((ALayerIndex < 0) or (ALayerIndex >= FLayerCount),
    'UploadLayer: layer index %d out of range [0..%d]',
    [ALayerIndex, FLayerCount - 1]);

  // Upload gate tensor via staging buffer
  LStaging := ACompute.CreateGpuBuffer(
    FLayers[ALayerIndex].GateSizeBytes,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  );
  try
    ACompute.UploadToBuffer(LStaging, FLayers[ALayerIndex].GatePtr,
      FLayers[ALayerIndex].GateSizeBytes);

    // Create device-local destination buffer (TRANSFER_SRC for readback/debug)
    FLayers[ALayerIndex].GateGpuBuffer := ACompute.CreateGpuBuffer(
      FLayers[ALayerIndex].GateSizeBytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    // Copy staging → device-local
    ACompute.CopyBuffer(LStaging, FLayers[ALayerIndex].GateGpuBuffer,
      FLayers[ALayerIndex].GateSizeBytes);
  finally
    ACompute.DestroyGpuBuffer(LStaging);
  end;

  // Upload down tensor via staging buffer
  LStaging := ACompute.CreateGpuBuffer(
    FLayers[ALayerIndex].DownSizeBytes,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  );
  try
    ACompute.UploadToBuffer(LStaging, FLayers[ALayerIndex].DownPtr,
      FLayers[ALayerIndex].DownSizeBytes);

    FLayers[ALayerIndex].DownGpuBuffer := ACompute.CreateGpuBuffer(
      FLayers[ALayerIndex].DownSizeBytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    ACompute.CopyBuffer(LStaging, FLayers[ALayerIndex].DownGpuBuffer,
      FLayers[ALayerIndex].DownSizeBytes);
  finally
    ACompute.DestroyGpuBuffer(LStaging);
  end;
end;

procedure TVdxVindex.UploadAll(const ACompute: TVdxVulkanCompute);
var
  LI: Integer;
begin
  for LI := 0 to FLayerCount - 1 do
    UploadLayer(LI, ACompute);
end;

// ============================================================================
//  GPU Cleanup
// ============================================================================

procedure TVdxVindex.FreeLayerGpu(const ALayerIndex: Integer; const ACompute: TVdxVulkanCompute);
begin
  TVdxUtils.FailIf((ALayerIndex < 0) or (ALayerIndex >= FLayerCount),
    'FreeLayerGpu: layer index %d out of range [0..%d]',
    [ALayerIndex, FLayerCount - 1]);

  if FLayers[ALayerIndex].GateGpuBuffer.Buffer <> VK_NULL_HANDLE then
    ACompute.DestroyGpuBuffer(FLayers[ALayerIndex].GateGpuBuffer);

  if FLayers[ALayerIndex].DownGpuBuffer.Buffer <> VK_NULL_HANDLE then
    ACompute.DestroyGpuBuffer(FLayers[ALayerIndex].DownGpuBuffer);
end;

procedure TVdxVindex.FreeAllGpu(const ACompute: TVdxVulkanCompute);
var
  LI: Integer;
begin
  for LI := 0 to FLayerCount - 1 do
    FreeLayerGpu(LI, ACompute);
end;

// ============================================================================
//  Accessors
// ============================================================================

function TVdxVindex.GetLayerCount(): Integer;
begin
  Result := FLayerCount;
end;

function TVdxVindex.GetLayer(const AIndex: Integer): TVdxFFNLayerView;
begin
  TVdxUtils.FailIf((AIndex < 0) or (AIndex >= FLayerCount),
    'GetLayer: index %d out of range [0..%d]',
    [AIndex, FLayerCount - 1]);
  Result := FLayers[AIndex];
end;

function TVdxVindex.GetHiddenDim(): UInt64;
begin
  Result := FHiddenDim;
end;

function TVdxVindex.GetFFNWidth(): UInt64;
begin
  Result := FFFNWidth;
end;

end.
