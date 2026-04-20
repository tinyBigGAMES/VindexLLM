{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Model;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Math,
  System.Generics.Collections,
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.GGUFReader,
  VindexLLM.Compute,
  VindexLLM.LayerNorm,
  VindexLLM.Attention,
  VindexLLM.FFN,
  VindexLLM.Tokenizer,
  VindexLLM.VirtualBuffer,
  VindexLLM.Shaders;

type

  { TVdxGeluMulPush }
  TVdxGeluMulPush = record
    Count: UInt32;
  end;

  { TVdxVecAddPush }
  TVdxVecAddPush = record
    Count: UInt32;
  end;

  { TVdxEmbedLookupPush }
  TVdxEmbedLookupPush = record
    TokenId: UInt32;
    DimParam: UInt32;
    EmbedScale: Single;
  end;

  { TVdxEmbedBatchPush }
  TVdxEmbedBatchPush = record
    DimParam: UInt32;
    EmbedScale: Single;
    NumTokens: UInt32;
  end;

  TVdxModel = class;
  TVdxModelClass = class of TVdxModel;

  { TVdxModel }
  TVdxModel = class(TVdxBaseObject)
  protected
    // Owned subsystems
    FReader: TVdxGGUFReader;
    FCompute: TVdxCompute;
    FNorm: TVdxLayerNorm;
    FAttn: TVdxAttention;
    FFFN: TVdxFFN;
    FTokenizer: TVdxTokenizer;

    // Model config (populated by descendant LoadModelConfig)
    FArchitecture: string;
    FGGUFPath: string;
    FMaxContext: Integer;
    FNumLayers: UInt32;
    FHiddenDim: UInt32;
    FFFNWidth: UInt32;
    FNumQHeads: UInt32;
    FNumKVHeads: UInt32;
    FHeadDim: UInt32;
    FVocabSize: Integer;
    FMaxSeqLen: UInt32;
    FWeightType: TVdxGGMLType;
    FEmbedType: TVdxGGMLType;

    // Per-layer weights (populated by descendant LoadWeights)
    FAttnWeights: array of TVdxAttnLayerWeights;
    FNormWeights: array of TVdxNormLayerWeights;
    FUpWeights: array of TVdxGpuBuffer;
    FOutputNormGpu: TVdxGpuBuffer;
    FEmbedPtr: PByte;
    FEmbedScale: Single;

    // Decode-path scratch buffers
    FResidualGpu: TVdxGpuBuffer;
    FWorkBufA: TVdxGpuBuffer;
    FAttnOutBuf: TVdxGpuBuffer;
    FFFNOutBuf: TVdxGpuBuffer;
    FGateBuf: TVdxGpuBuffer;
    FUpBuf: TVdxGpuBuffer;

    // GELU-mul pipeline
    FGeluMulShader: VkShaderModule;
    FGeluMulBundle: TVdxComputePipelineBundle;
    FGeluMulDescLayout: VkDescriptorSetLayout;
    FGeluMulDescPool: VkDescriptorPool;
    FGeluMulDescSet: VkDescriptorSet;

    // Vec-add pipeline
    FVecAddShader: VkShaderModule;
    FVecAddBundle: TVdxComputePipelineBundle;
    FVecAddDescLayout: VkDescriptorSetLayout;
    FVecAddDescPool: VkDescriptorPool;
    FVecAddAttnDescSet: VkDescriptorSet;
    FVecAddFFNDescSet: VkDescriptorSet;

    // Single-token embed lookup pipelines
    FEmbedF16Shader: VkShaderModule;
    FEmbedQ8Shader: VkShaderModule;
    FEmbedQ4Shader: VkShaderModule;
    FEmbedF16Bundle: TVdxComputePipelineBundle;
    FEmbedQ8Bundle: TVdxComputePipelineBundle;
    FEmbedQ4Bundle: TVdxComputePipelineBundle;
    FEmbedDescLayout: VkDescriptorSetLayout;
    FEmbedDescPool: VkDescriptorPool;
    FEmbedDescSet: VkDescriptorSet;

    // Batched embed lookup pipelines
    FEmbedBatchF16Shader: VkShaderModule;
    FEmbedBatchQ8Shader: VkShaderModule;
    FEmbedBatchQ4Shader: VkShaderModule;
    FEmbedBatchF16Bundle: TVdxComputePipelineBundle;
    FEmbedBatchQ8Bundle: TVdxComputePipelineBundle;
    FEmbedBatchQ4Bundle: TVdxComputePipelineBundle;
    FEmbedBatchDescLayout: VkDescriptorSetLayout;
    FEmbedBatchDescPool: VkDescriptorPool;
    FEmbedBatchDescSet: VkDescriptorSet;
    FTokenIdsGpu: TVdxGpuBuffer;

    // Batch-path matrix buffers
    FResidualMat: TVdxGpuBuffer;
    FWorkMat: TVdxGpuBuffer;
    FQMat: TVdxGpuBuffer;
    FKMat: TVdxGpuBuffer;
    FVMat: TVdxGpuBuffer;
    FAttnOutMatBuf: TVdxGpuBuffer;
    FGateMat: TVdxGpuBuffer;
    FUpMatBuf: TVdxGpuBuffer;
    FFFNOutMat: TVdxGpuBuffer;

    // Batch elementwise descriptor sets
    FBatchEWDescPool: VkDescriptorPool;
    FBatchVecAddAttnDescSet: VkDescriptorSet;
    FBatchVecAddFFNDescSet: VkDescriptorSet;
    FBatchGeluMulDescSet: VkDescriptorSet;

    // Unembed resources
    FEmbedGpu: TVdxGpuBuffer;
    FLogitsBuf: TVdxGpuBuffer;
    FLogitsVBuf: TVdxVirtualBuffer<Single>;

    // CPU embedding scratch (for F32 embed fallback)
    FResidual: array of Single;

    // Resource lifecycle helpers
    function BuildDecodeResources(): Boolean;
    procedure FreeDecodeResources();
    function BuildBatchResources(): Boolean;
    procedure FreeBatchResources();

    // Weight upload helpers
    function UploadNormWeight(const ATensorName: string;
      const ACount: UInt32): TVdxGpuBuffer;
    function UploadWeightTensor(const ATensorName: string): TVdxGpuBuffer;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    procedure SetErrors(const AErrors: TVdxErrors); override;
    procedure SetStatusCallback(const ACallback: TVdxStatusCallback;
      const AUserData: Pointer = nil); override;

    // Architecture identity — descendants override
    class function SupportedArchitectures(): TArray<string>; virtual;

    // Lifecycle hooks (called by LoadModel factory in sequence)
    function LoadModelConfig(const AReader: TVdxGGUFReader;
      const AMaxContext: Integer): Boolean; virtual;
    function InitSubsystems(): Boolean; virtual;
    function LoadWeights(): Boolean; virtual;
    procedure FreeWeights(); virtual;

    // Forward pass — descendants MUST override
    procedure RunLayerForward(const ALayer: Integer;
      const APosition: Integer); virtual;
    procedure RunLayerForwardBatch(const ALayer: Integer;
      const ANumTokens: UInt32; const AStartPos: UInt32;
      const ABidirectional: Boolean = False); virtual;

    // Per-layer RoPE theta (default 10000.0)
    function GetRoPETheta(const ALayer: Integer): Single; virtual;

    // Template surface
    function FormatPrompt(const APrompt: string): string; virtual;
    function FormatEmbedding(const AText: string;
      const AIsQuery: Boolean): string; virtual;
    function GetStopTokenStrings(): TArray<string>; virtual;
    function SupportsEmbedding(): Boolean; virtual;

    // Output norm helpers
    procedure ApplyOutputNormBatch(const ANumTokens: UInt32);

    // Embedding helpers
    procedure EmbedToken(const ATokenId: Integer);
    procedure EmbedTokensBatch(const ATokenIds: TArray<Integer>;
      const ANumTokens: Integer; const AOutputBuf: TVdxGpuBuffer);
    procedure SeedResidualFromBatchLast(const ANumTokens: UInt32);
    procedure UnembedToLogits(const AOutLogits: TVdxGpuBuffer);

    // Factory — reads GGUF arch, resolves via registry, creates instance
    class function LoadModel(const AGGUFPath: string;
      const AMaxContext: Integer;
      const AStatusCallback: TVdxStatusCallback = nil;
      const AStatusUserData: Pointer = nil): TVdxModel;

    // Accessors
    property Architecture: string read FArchitecture;
    property GGUFPath: string read FGGUFPath;
    property MaxContext: Integer read FMaxContext;
    property NumLayers: UInt32 read FNumLayers;
    property HiddenDim: UInt32 read FHiddenDim;
    property FFNWidth: UInt32 read FFFNWidth;
    property NumQHeads: UInt32 read FNumQHeads;
    property NumKVHeads: UInt32 read FNumKVHeads;
    property HeadDim: UInt32 read FHeadDim;
    property VocabSize: Integer read FVocabSize;
    property MaxSeqLen: UInt32 read FMaxSeqLen;
    property WeightType: TVdxGGMLType read FWeightType;
    property EmbedType: TVdxGGMLType read FEmbedType;
    property Compute: TVdxCompute read FCompute;
    property Norm: TVdxLayerNorm read FNorm;
    property Attn: TVdxAttention read FAttn;
    property FFN: TVdxFFN read FFFN;
    property Tokenizer: TVdxTokenizer read FTokenizer;
    property Reader: TVdxGGUFReader read FReader;
    property ResidualGpuBuffer: TVdxGpuBuffer read FResidualGpu;
    property ResidualMatBuffer: TVdxGpuBuffer read FResidualMat;
    property EmbedGpuBuffer: TVdxGpuBuffer read FEmbedGpu;
    property LogitsBuffer: TVdxGpuBuffer read FLogitsBuf;
    property LogitsVBuf: TVdxVirtualBuffer<Single> read FLogitsVBuf;
  end;

implementation

uses
  VindexLLM.Model.Registry;

{ TVdxModel }

constructor TVdxModel.Create();
begin
  inherited;
  FReader := nil;
  FArchitecture := '';
  FGGUFPath := '';
  FMaxContext := 0;
  FNumLayers := 0;
  FHiddenDim := 0;
  FFFNWidth := 0;
  FNumQHeads := 0;
  FNumKVHeads := 0;
  FHeadDim := 0;
  FVocabSize := 0;
  FMaxSeqLen := 0;
  FWeightType := TVdxGGMLType(0);
  FEmbedType := TVdxGGMLType(0);
  FEmbedPtr := nil;
  FEmbedScale := 0.0;
  FLogitsVBuf := nil;

  // Create subsystems — cheap construction, real work in Init*
  FCompute := TVdxCompute.Create();
  FCompute.SetErrors(FErrors);
  FNorm := TVdxLayerNorm.Create();
  FNorm.SetErrors(FErrors);
  FAttn := TVdxAttention.Create();
  FAttn.SetErrors(FErrors);
  FFFN := TVdxFFN.Create();
  FFFN.SetErrors(FErrors);
  FTokenizer := TVdxTokenizer.Create();
  FTokenizer.SetErrors(FErrors);
end;

destructor TVdxModel.Destroy();
begin
  FreeWeights();
  FreeBatchResources();
  FreeDecodeResources();

  // Subsystems — reverse construction order
  FTokenizer.Free();
  FFFN.Free();
  FAttn.Free();
  FNorm.Free();
  FCompute.Free();

  FreeAndNil(FReader);
  FreeAndNil(FLogitsVBuf);

  inherited;
end;

procedure TVdxModel.SetErrors(const AErrors: TVdxErrors);
begin
  if AErrors = FErrors then Exit;

  if FCompute <> nil then FCompute.SetErrors(AErrors);
  if FNorm <> nil then FNorm.SetErrors(AErrors);
  if FAttn <> nil then FAttn.SetErrors(AErrors);
  if FFFN <> nil then FFFN.SetErrors(AErrors);
  if FTokenizer <> nil then FTokenizer.SetErrors(AErrors);

  inherited SetErrors(AErrors);
end;

procedure TVdxModel.SetStatusCallback(const ACallback: TVdxStatusCallback;
  const AUserData: Pointer);
begin
  inherited SetStatusCallback(ACallback, AUserData);

  if FCompute <> nil then
    FCompute.SetStatusCallback(ACallback, AUserData);
  if FNorm <> nil then
    FNorm.SetStatusCallback(ACallback, AUserData);
end;

// --- Default virtual implementations ---

class function TVdxModel.SupportedArchitectures(): TArray<string>;
begin
  Result := nil;
end;

function TVdxModel.LoadModelConfig(const AReader: TVdxGGUFReader;
  const AMaxContext: Integer): Boolean;
begin
  FReader := AReader;
  FMaxContext := AMaxContext;
  Result := True;
end;

function TVdxModel.InitSubsystems(): Boolean;
begin
  Result := False;

  // Vulkan init
  FCompute.Init();
  if FErrors.HasFatal() then Exit;

  // Subsystem init
  FNorm.Init(FCompute);
  if FErrors.HasFatal() then Exit;

  FAttn.Init(FCompute, FHiddenDim, FNumQHeads, FNumKVHeads,
    FHeadDim, FNumLayers, FMaxSeqLen, FFFNWidth);
  if FErrors.HasFatal() then Exit;

  // Tokenizer
  if not FTokenizer.LoadFromGGUF(FReader) then
  begin
    FErrors.Add(esFatal, 'INIT', 'Failed to load tokenizer from GGUF');
    Exit;
  end;
  FVocabSize := FTokenizer.GetVocabSize();
  Status('Tokenizer loaded: %d tokens, BOS=%d, EOS=%d',
    [FVocabSize, FTokenizer.GetBosId(), FTokenizer.GetEosId()]);

  // Decode-path resources
  if not BuildDecodeResources() then Exit;

  Result := True;
end;

function TVdxModel.LoadWeights(): Boolean;
begin
  Result := True;
end;

procedure TVdxModel.FreeWeights();
var
  LLayer: Integer;
begin
  // Per-layer attn weights
  if FAttn <> nil then
    for LLayer := 0 to Length(FAttnWeights) - 1 do
      FAttn.FreeAttnWeights(FAttnWeights[LLayer]);
  FAttnWeights := nil;

  // Per-layer up weights
  if FCompute <> nil then
    for LLayer := 0 to Length(FUpWeights) - 1 do
    begin
      if FUpWeights[LLayer].Buffer <> VK_NULL_HANDLE then
        FCompute.DestroyGpuBuffer(FUpWeights[LLayer]);
    end;
  FUpWeights := nil;

  // Per-layer norm weights
  if FCompute <> nil then
    for LLayer := 0 to Length(FNormWeights) - 1 do
    begin
      if FNormWeights[LLayer].AttnNormGpu.Buffer <> VK_NULL_HANDLE then
        FCompute.DestroyGpuBuffer(FNormWeights[LLayer].AttnNormGpu);
      if FNormWeights[LLayer].PostAttnNormGpu.Buffer <> VK_NULL_HANDLE then
        FCompute.DestroyGpuBuffer(FNormWeights[LLayer].PostAttnNormGpu);
      if FNormWeights[LLayer].FFNNormGpu.Buffer <> VK_NULL_HANDLE then
        FCompute.DestroyGpuBuffer(FNormWeights[LLayer].FFNNormGpu);
      if FNormWeights[LLayer].PostFFNNormGpu.Buffer <> VK_NULL_HANDLE then
        FCompute.DestroyGpuBuffer(FNormWeights[LLayer].PostFFNNormGpu);
      if FNormWeights[LLayer].QNormGpu.Buffer <> VK_NULL_HANDLE then
        FCompute.DestroyGpuBuffer(FNormWeights[LLayer].QNormGpu);
      if FNormWeights[LLayer].KNormGpu.Buffer <> VK_NULL_HANDLE then
        FCompute.DestroyGpuBuffer(FNormWeights[LLayer].KNormGpu);
    end;
  FNormWeights := nil;

  // Global output norm
  if (FCompute <> nil) and (FOutputNormGpu.Buffer <> VK_NULL_HANDLE) then
  begin
    FCompute.DestroyGpuBuffer(FOutputNormGpu);
    FOutputNormGpu := Default(TVdxGpuBuffer);
  end;

  // Gate/down via FFN
  if (FFFN <> nil) and (FCompute <> nil) then
    FFFN.FreeAllGpu(FCompute);

  FEmbedPtr := nil;
  FEmbedScale := 0.0;
  FResidual := nil;
end;

procedure TVdxModel.RunLayerForward(const ALayer: Integer;
  const APosition: Integer);
begin
  raise ENotImplemented.CreateFmt(
    '%s.RunLayerForward must be overridden', [ClassName]);
end;

procedure TVdxModel.RunLayerForwardBatch(const ALayer: Integer;
  const ANumTokens: UInt32; const AStartPos: UInt32;
  const ABidirectional: Boolean);
begin
  raise ENotImplemented.CreateFmt(
    '%s.RunLayerForwardBatch must be overridden', [ClassName]);
end;

function TVdxModel.GetRoPETheta(const ALayer: Integer): Single;
begin
  Result := 10000.0;
end;

function TVdxModel.FormatPrompt(const APrompt: string): string;
begin
  Result := APrompt;
end;

function TVdxModel.FormatEmbedding(const AText: string;
  const AIsQuery: Boolean): string;
begin
  Result := AText;
end;

function TVdxModel.GetStopTokenStrings(): TArray<string>;
begin
  Result := nil;
end;

function TVdxModel.SupportsEmbedding(): Boolean;
begin
  Result := False;
end;

// --- Weight upload helpers (from src_ref Inference) ---

function TVdxModel.UploadNormWeight(const ATensorName: string;
  const ACount: UInt32): TVdxGpuBuffer;
var
  LPtr: Pointer;
  LData: array of Single;
begin
  FillChar(Result, SizeOf(Result), 0);

  LPtr := FReader.GetTensorDataPtr(ATensorName);
  if LPtr = nil then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Norm tensor not found: %s', [ATensorName]);
    Exit;
  end;

  try
    SetLength(LData, ACount);
    Move(LPtr^, LData[0], ACount * SizeOf(Single));

    Result := FCompute.CreateGpuBuffer(
      UInt64(ACount) * SizeOf(Single),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    FCompute.UploadToBuffer(Result, @LData[0], UInt64(ACount) * SizeOf(Single));
  except
    on E: Exception do
    begin
      FErrors.Add(esFatal, 'LOAD',
        'Exception uploading norm "%s": %s', [ATensorName, E.Message]);
    end;
  end;
end;

function TVdxModel.UploadWeightTensor(const ATensorName: string): TVdxGpuBuffer;
var
  LInfo: TVdxGGUFTensorInfo;
  LPtr: Pointer;
  LSize: UInt64;
  LStaging: TVdxGpuBuffer;
begin
  if not FReader.GetTensorInfo(ATensorName, LInfo) then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Tensor not found: %s', [ATensorName]);
    FillChar(Result, SizeOf(Result), 0);
    Exit;
  end;

  LPtr := FReader.GetTensorDataPtr(ATensorName);
  if LPtr = nil then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Failed to resolve mmap pointer for "%s"', [ATensorName]);
    FillChar(Result, SizeOf(Result), 0);
    Exit;
  end;

  LSize := VdxGGMLTensorBytes(LInfo.TensorType,
    LInfo.Dimensions[0], LInfo.Dimensions[1]);
  if LSize = 0 then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Unsupported tensor type for %s: %s',
      [ATensorName, VdxGGMLTypeName(LInfo.TensorType)]);
    FillChar(Result, SizeOf(Result), 0);
    Exit;
  end;

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

// --- BuildDecodeResources ---

function TVdxModel.BuildDecodeResources(): Boolean;
var
  LBufSize: UInt64;
  LSpvData: TBytes;
begin
  Result := False;
  LBufSize := UInt64(FHiddenDim) * SizeOf(Single);

  // GPU residual — host-visible for initial embed upload
  FResidualGpu := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Work buffer — normed-input scratch
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

  // Dense FFN scratch buffers (device-local)
  FGateBuf := FCompute.CreateGpuBuffer(
    UInt64(FFFNWidth) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  FUpBuf := FCompute.CreateGpuBuffer(
    UInt64(FFFNWidth) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if FErrors.HasFatal() then Exit;

  // --- GELU-mul pipeline ---
  LSpvData := VdxLoadShader('GELU_MUL');
  FGeluMulShader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  FGeluMulDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  FGeluMulBundle := FCompute.CreateComputePipelineWithPush(
    FGeluMulShader, 'main', FGeluMulDescLayout, SizeOf(TVdxGeluMulPush));
  FGeluMulDescPool := FCompute.CreateDescriptorPoolForStorage(1, 2);
  FGeluMulDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FGeluMulDescPool, FGeluMulDescLayout, [FGateBuf, FUpBuf]);

  if FErrors.HasFatal() then Exit;

  // --- Vec-add pipeline ---
  LSpvData := VdxLoadShader('VEC_ADD');
  FVecAddShader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  FVecAddDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  FVecAddBundle := FCompute.CreateComputePipelineWithPush(
    FVecAddShader, 'main', FVecAddDescLayout, SizeOf(TVdxVecAddPush));
  FVecAddDescPool := FCompute.CreateDescriptorPoolForStorage(2, 4);
  FVecAddAttnDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FVecAddDescPool, FVecAddDescLayout, [FResidualGpu, FAttnOutBuf]);
  FVecAddFFNDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FVecAddDescPool, FVecAddDescLayout, [FResidualGpu, FFFNOutBuf]);

  if FErrors.HasFatal() then Exit;

  // --- Single-token embed lookup pipelines ---
  LSpvData := VdxLoadShader('EMBED_LOOKUP_F16');
  FEmbedF16Shader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  LSpvData := VdxLoadShader('EMBED_LOOKUP_Q8');
  FEmbedQ8Shader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  LSpvData := VdxLoadShader('EMBED_LOOKUP_Q4_0');
  FEmbedQ4Shader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  FEmbedDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  FEmbedF16Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedF16Shader, 'main', FEmbedDescLayout, SizeOf(TVdxEmbedLookupPush));
  FEmbedQ8Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedQ8Shader, 'main', FEmbedDescLayout, SizeOf(TVdxEmbedLookupPush));
  FEmbedQ4Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedQ4Shader, 'main', FEmbedDescLayout, SizeOf(TVdxEmbedLookupPush));

  if FErrors.HasFatal() then Exit;

  // Desc set allocated later in BuildBatchResources once FEmbedGpu is valid

  // CPU embedding scratch
  SetLength(FResidual, FHiddenDim);

  if FErrors.HasFatal() then Exit;
  Result := True;
end;

procedure TVdxModel.FreeDecodeResources();
begin
  if FCompute = nil then Exit;

  // GELU-mul pipeline
  if FGeluMulDescPool <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FGeluMulDescPool);
    FGeluMulDescPool := VK_NULL_HANDLE;
    FGeluMulDescSet := VK_NULL_HANDLE;
  end;
  if FGeluMulBundle.Pipeline <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyComputePipelineBundle(FGeluMulBundle);
    FGeluMulBundle := Default(TVdxComputePipelineBundle);
  end;
  if FGeluMulDescLayout <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorSetLayoutHandle(FGeluMulDescLayout);
    FGeluMulDescLayout := VK_NULL_HANDLE;
  end;
  if FGeluMulShader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FGeluMulShader);
    FGeluMulShader := VK_NULL_HANDLE;
  end;

  // Vec-add pipeline
  if FVecAddDescPool <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FVecAddDescPool);
    FVecAddDescPool := VK_NULL_HANDLE;
    FVecAddAttnDescSet := VK_NULL_HANDLE;
    FVecAddFFNDescSet := VK_NULL_HANDLE;
  end;
  if FVecAddBundle.Pipeline <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyComputePipelineBundle(FVecAddBundle);
    FVecAddBundle := Default(TVdxComputePipelineBundle);
  end;
  if FVecAddDescLayout <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorSetLayoutHandle(FVecAddDescLayout);
    FVecAddDescLayout := VK_NULL_HANDLE;
  end;
  if FVecAddShader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FVecAddShader);
    FVecAddShader := VK_NULL_HANDLE;
  end;

  // Single-token embed pipelines (pool is in FreeBatchResources)
  if FEmbedF16Bundle.Pipeline <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyComputePipelineBundle(FEmbedF16Bundle);
    FEmbedF16Bundle := Default(TVdxComputePipelineBundle);
  end;
  if FEmbedQ8Bundle.Pipeline <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyComputePipelineBundle(FEmbedQ8Bundle);
    FEmbedQ8Bundle := Default(TVdxComputePipelineBundle);
  end;
  if FEmbedQ4Bundle.Pipeline <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyComputePipelineBundle(FEmbedQ4Bundle);
    FEmbedQ4Bundle := Default(TVdxComputePipelineBundle);
  end;
  if FEmbedDescLayout <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorSetLayoutHandle(FEmbedDescLayout);
    FEmbedDescLayout := VK_NULL_HANDLE;
  end;
  if FEmbedF16Shader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FEmbedF16Shader);
    FEmbedF16Shader := VK_NULL_HANDLE;
  end;
  if FEmbedQ8Shader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FEmbedQ8Shader);
    FEmbedQ8Shader := VK_NULL_HANDLE;
  end;
  if FEmbedQ4Shader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FEmbedQ4Shader);
    FEmbedQ4Shader := VK_NULL_HANDLE;
  end;

  // Work buffers
  if FResidualGpu.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FResidualGpu);
    FResidualGpu := Default(TVdxGpuBuffer);
  end;
  if FWorkBufA.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FWorkBufA);
    FWorkBufA := Default(TVdxGpuBuffer);
  end;
  if FAttnOutBuf.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FAttnOutBuf);
    FAttnOutBuf := Default(TVdxGpuBuffer);
  end;
  if FFFNOutBuf.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FFFNOutBuf);
    FFFNOutBuf := Default(TVdxGpuBuffer);
  end;
  if FGateBuf.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FGateBuf);
    FGateBuf := Default(TVdxGpuBuffer);
  end;
  if FUpBuf.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FUpBuf);
    FUpBuf := Default(TVdxGpuBuffer);
  end;

  FResidual := nil;
end;

// --- BuildBatchResources ---
// Called from descendant LoadWeights after FEmbedGpu and FVocabSize are set.

function TVdxModel.BuildBatchResources(): Boolean;
var
  LSpvData: TBytes;
begin
  Result := False;

  // Single-token embed desc set (needs FEmbedGpu)
  FEmbedDescPool := FCompute.CreateDescriptorPoolForStorage(1, 2);
  FEmbedDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FEmbedDescPool, FEmbedDescLayout, [FEmbedGpu, FResidualGpu]);

  if FErrors.HasFatal() then Exit;

  // --- Batched embed lookup pipelines ---
  LSpvData := VdxLoadShader('EMBED_LOOKUP_BATCH_F16');
  FEmbedBatchF16Shader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  LSpvData := VdxLoadShader('EMBED_LOOKUP_BATCH_Q8');
  FEmbedBatchQ8Shader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  LSpvData := VdxLoadShader('EMBED_LOOKUP_BATCH_Q4_0');
  FEmbedBatchQ4Shader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  FEmbedBatchDescLayout := FCompute.CreateStorageDescriptorSetLayout(3);
  FEmbedBatchF16Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedBatchF16Shader, 'main', FEmbedBatchDescLayout,
    SizeOf(TVdxEmbedBatchPush));
  FEmbedBatchQ8Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedBatchQ8Shader, 'main', FEmbedBatchDescLayout,
    SizeOf(TVdxEmbedBatchPush));
  FEmbedBatchQ4Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedBatchQ4Shader, 'main', FEmbedBatchDescLayout,
    SizeOf(TVdxEmbedBatchPush));

  // Token IDs buffer: host-visible, sized for max sequence length
  FTokenIdsGpu := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * SizeOf(UInt32),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Batch embed desc set (rebound per call via UpdateDescriptorSetBuffers)
  FEmbedBatchDescPool := FCompute.CreateDescriptorPoolForStorage(1, 3);
  FEmbedBatchDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FEmbedBatchDescPool, FEmbedBatchDescLayout,
    [FEmbedGpu, FResidualGpu, FTokenIdsGpu]);

  if FErrors.HasFatal() then Exit;

  // --- Batch prefill matrix buffers ---
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

  if FErrors.HasFatal() then Exit;

  // Batch elementwise descriptor sets (vec-add + gelu-mul on matrices)
  FBatchEWDescPool := FCompute.CreateDescriptorPoolForStorage(3, 6);
  FBatchVecAddAttnDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FBatchEWDescPool, FVecAddDescLayout, [FResidualMat, FAttnOutMatBuf]);
  FBatchVecAddFFNDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FBatchEWDescPool, FVecAddDescLayout, [FResidualMat, FFFNOutMat]);
  FBatchGeluMulDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FBatchEWDescPool, FGeluMulDescLayout, [FGateMat, FUpMatBuf]);

  if FErrors.HasFatal() then Exit;

  // Logits buffer: F32 x VocabSize, host-visible for CPU sampling
  FLogitsBuf := FCompute.CreateGpuBuffer(
    UInt64(FVocabSize) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Pre-allocated CPU logits buffer (reused every token)
  FLogitsVBuf := TVdxVirtualBuffer<Single>.Create();
  FLogitsVBuf.Allocate(UInt64(FVocabSize));

  if FErrors.HasFatal() then Exit;
  Result := True;
end;

procedure TVdxModel.FreeBatchResources();
begin
  if FCompute = nil then Exit;

  // Single-token embed desc pool (created in BuildBatchResources)
  if FEmbedDescPool <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FEmbedDescPool);
    FEmbedDescPool := VK_NULL_HANDLE;
    FEmbedDescSet := VK_NULL_HANDLE;
  end;

  // Batched embed pipeline
  if FEmbedBatchDescPool <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FEmbedBatchDescPool);
    FEmbedBatchDescPool := VK_NULL_HANDLE;
    FEmbedBatchDescSet := VK_NULL_HANDLE;
  end;
  if FEmbedBatchF16Bundle.Pipeline <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyComputePipelineBundle(FEmbedBatchF16Bundle);
    FEmbedBatchF16Bundle := Default(TVdxComputePipelineBundle);
  end;
  if FEmbedBatchQ8Bundle.Pipeline <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyComputePipelineBundle(FEmbedBatchQ8Bundle);
    FEmbedBatchQ8Bundle := Default(TVdxComputePipelineBundle);
  end;
  if FEmbedBatchQ4Bundle.Pipeline <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyComputePipelineBundle(FEmbedBatchQ4Bundle);
    FEmbedBatchQ4Bundle := Default(TVdxComputePipelineBundle);
  end;
  if FEmbedBatchDescLayout <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorSetLayoutHandle(FEmbedBatchDescLayout);
    FEmbedBatchDescLayout := VK_NULL_HANDLE;
  end;
  if FEmbedBatchF16Shader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FEmbedBatchF16Shader);
    FEmbedBatchF16Shader := VK_NULL_HANDLE;
  end;
  if FEmbedBatchQ8Shader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FEmbedBatchQ8Shader);
    FEmbedBatchQ8Shader := VK_NULL_HANDLE;
  end;
  if FEmbedBatchQ4Shader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FEmbedBatchQ4Shader);
    FEmbedBatchQ4Shader := VK_NULL_HANDLE;
  end;
  if FTokenIdsGpu.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FTokenIdsGpu);
    FTokenIdsGpu := Default(TVdxGpuBuffer);
  end;

  // Batch elementwise desc pool
  if FBatchEWDescPool <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FBatchEWDescPool);
    FBatchEWDescPool := VK_NULL_HANDLE;
    FBatchVecAddAttnDescSet := VK_NULL_HANDLE;
    FBatchVecAddFFNDescSet := VK_NULL_HANDLE;
    FBatchGeluMulDescSet := VK_NULL_HANDLE;
  end;

  // Batch matrices
  if FResidualMat.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FResidualMat);
    FResidualMat := Default(TVdxGpuBuffer);
  end;
  if FWorkMat.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FWorkMat);
    FWorkMat := Default(TVdxGpuBuffer);
  end;
  if FQMat.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FQMat);
    FQMat := Default(TVdxGpuBuffer);
  end;
  if FKMat.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FKMat);
    FKMat := Default(TVdxGpuBuffer);
  end;
  if FVMat.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FVMat);
    FVMat := Default(TVdxGpuBuffer);
  end;
  if FAttnOutMatBuf.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FAttnOutMatBuf);
    FAttnOutMatBuf := Default(TVdxGpuBuffer);
  end;
  if FGateMat.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FGateMat);
    FGateMat := Default(TVdxGpuBuffer);
  end;
  if FUpMatBuf.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FUpMatBuf);
    FUpMatBuf := Default(TVdxGpuBuffer);
  end;
  if FFFNOutMat.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FFFNOutMat);
    FFFNOutMat := Default(TVdxGpuBuffer);
  end;

  // Embedding GPU mirror
  if FEmbedGpu.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FEmbedGpu);
    FEmbedGpu := Default(TVdxGpuBuffer);
  end;

  // Logits
  if FLogitsBuf.Buffer <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyGpuBuffer(FLogitsBuf);
    FLogitsBuf := Default(TVdxGpuBuffer);
  end;
  FreeAndNil(FLogitsVBuf);
end;

// --- Embedding helpers (from src_ref Inference) ---

procedure TVdxModel.EmbedToken(const ATokenId: Integer);
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
    // GPU dispatch: F16, Q8_0, or Q4_0 embedding lookup
    LPush.TokenId := UInt32(ATokenId);
    LPush.EmbedScale := FEmbedScale;

    if FEmbedType = gtQ4_0 then
    begin
      LPush.DimParam := FHiddenDim;
      FCompute.DispatchComputeWithPush(
        FEmbedQ4Bundle.Pipeline, FEmbedQ4Bundle.PipelineLayout,
        FEmbedDescSet, @LPush, SizeOf(LPush), 1);
    end
    else if FEmbedType = gtQ8_0 then
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
    FCompute.BatchBarrier();
  end;
end;

procedure TVdxModel.EmbedTokensBatch(const ATokenIds: TArray<Integer>;
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

  if FEmbedType = gtQ4_0 then
  begin
    LPush.DimParam := FHiddenDim;
    FCompute.DispatchComputeWithPush(
      FEmbedBatchQ4Bundle.Pipeline, FEmbedBatchQ4Bundle.PipelineLayout,
      FEmbedBatchDescSet, @LPush, SizeOf(LPush),
      UInt32(ANumTokens));
  end
  else if FEmbedType = gtQ8_0 then
  begin
    LPush.DimParam := FHiddenDim;
    FCompute.DispatchComputeWithPush(
      FEmbedBatchQ8Bundle.Pipeline, FEmbedBatchQ8Bundle.PipelineLayout,
      FEmbedBatchDescSet, @LPush, SizeOf(LPush),
      UInt32(ANumTokens));
  end
  else
  begin
    LPush.DimParam := FHiddenDim div 2;
    FCompute.DispatchComputeWithPush(
      FEmbedBatchF16Bundle.Pipeline, FEmbedBatchF16Bundle.PipelineLayout,
      FEmbedBatchDescSet, @LPush, SizeOf(LPush),
      (FHiddenDim div 2 + 255) div 256, UInt32(ANumTokens));
  end;
  FCompute.BatchBarrier();
end;

procedure TVdxModel.SeedResidualFromBatchLast(const ANumTokens: UInt32);
begin
  FCompute.CopyBufferRegion(
    FResidualMat,
    UInt64(ANumTokens - 1) * UInt64(FHiddenDim) * SizeOf(Single),
    FResidualGpu, 0,
    UInt64(FHiddenDim) * SizeOf(Single));
end;

procedure TVdxModel.ApplyOutputNormBatch(const ANumTokens: UInt32);
begin
  FCompute.BeginBatch();
  FNorm.ApplyBatch(FResidualMat, FOutputNormGpu, FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();
  FCompute.EndBatch();
end;

procedure TVdxModel.UnembedToLogits(const AOutLogits: TVdxGpuBuffer);
begin
  FCompute.BeginBatch();

  // Fused copy+norm: residual → OutputNorm → WorkBufA
  FNorm.ApplyCopy(FResidualGpu, FOutputNormGpu, FWorkBufA, FHiddenDim);
  FCompute.BatchBarrier();

  // Tied embedding matvec: WorkBufA × EmbedGpu^T → logits
  FAttn.TestMatVec(FEmbedGpu, FWorkBufA, AOutLogits,
    FHiddenDim, UInt32(FVocabSize), FEmbedType);

  FCompute.EndBatch();
end;

// --- Factory ---

class function TVdxModel.LoadModel(const AGGUFPath: string;
  const AMaxContext: Integer;
  const AStatusCallback: TVdxStatusCallback;
  const AStatusUserData: Pointer): TVdxModel;
var
  LReader: TVdxGGUFReader;
  LArch: string;
  LModelClass: TVdxModelClass;
  LRegistered: TArray<string>;
begin
  Result := nil;

  // Open GGUF and detect architecture
  LReader := TVdxGGUFReader.Create();
  if Assigned(AStatusCallback) then
    LReader.SetStatusCallback(AStatusCallback, AStatusUserData);

  if not LReader.Open(AGGUFPath) then
  begin
    LReader.Free();
    Exit;
  end;

  // Detect architecture from GGUF metadata
  if LReader.HasMetadata('general.architecture') then
    LArch := LowerCase(LReader.GetMetadataString('general.architecture'))
  else
    LArch := 'unknown';

  // Resolve concrete class from registry
  LModelClass := TVdxModelRegistry.ResolveClass(LArch);
  if LModelClass = nil then
  begin
    LRegistered := TVdxModelRegistry.ListArchitectures();
    LReader.Close();
    LReader.Free();
    Exit;
  end;

  // Create concrete instance
  Result := LModelClass.Create();
  Result.FArchitecture := LArch;
  Result.FGGUFPath := AGGUFPath;

  if Assigned(AStatusCallback) then
    Result.SetStatusCallback(AStatusCallback, AStatusUserData);

  // Lifecycle: LoadModelConfig → InitSubsystems → LoadWeights
  if not Result.LoadModelConfig(LReader, AMaxContext) then
  begin
    // If inherited ran, FReader is set and destructor handles cleanup.
    // If it didn't, reader is leaked — but inherited always succeeds.
    FreeAndNil(Result);
    Exit;
  end;

  if not Result.InitSubsystems() then
  begin
    FreeAndNil(Result);
    Exit;
  end;

  if not Result.LoadWeights() then
  begin
    FreeAndNil(Result);
    Exit;
  end;

  Result.Status('Model loaded successfully (%s)', [LArch]);
end;

end.
