{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.FFNWeights;

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
  VDX_ERROR_FFN_COMPUTE_NIL       = 'FN01';
  VDX_ERROR_FFN_ALREADY_INIT      = 'FN02';
  VDX_ERROR_FFN_NOT_INIT          = 'FN03';
  VDX_ERROR_FFN_INIT_EXCEPTION    = 'FN04';
  VDX_ERROR_FFN_RESOLVE_EXCEPTION = 'FN05';
  VDX_ERROR_FFN_TENSOR_NOT_FOUND  = 'FN06';
  VDX_ERROR_FFN_UNSUPPORTED_TYPE  = 'FN07';

type

  //==========================================================================
  // Push-constant records — match the layout of the corresponding SPIR-V
  // shaders byte-for-byte. Do not reorder fields or change widths; any
  // divergence breaks the shader binding.
  //==========================================================================

  { TVdxFFNMatVecPush }
  // Used by both MATVEC_F16 (InDimParam = in_dim / 2) and MATVEC_Q8_0 /
  // MATVEC_Q4_0 (InDimParam = in_dim). Identical layout to the matvec
  // push in Attention — kept local so FFN does not reach across units.
  TVdxFFNMatVecPush = record
    InDimParam: UInt32;
    OutDim:     UInt32;
  end;

  { TVdxFFNMatMulPush }
  // Batch matmul push — X dim is rows of OutDim, Y dim is token count.
  TVdxFFNMatMulPush = record
    InDimParam: UInt32;  // in_dim/2 for F16, in_dim for Q8_0/Q4_0
    OutDim:     UInt32;
    NumTokens:  UInt32;
  end;

  { TVdxGeluMulPush }
  // Element-wise GELU(gate) * up, in-place into gate buffer. Count is the
  // total number of elements across the whole flattened buffer (single-token
  // = FFNWidth; batch = NumTokens * FFNWidth).
  TVdxGeluMulPush = record
    Count: UInt32;
  end;

  { TVdxFFNLayerWeights }
  // Streaming weight reference for one FFN layer. Mirrors
  // TVdxAttnLayerWeights: holds mmap pointers into the GGUF tensor region,
  // NOT permanent GPU buffers. Populated by TVdxFFN.ResolveFFNWeights;
  // consumed by TVdxFFN.Forward / ForwardBatch, which copy the slice into
  // the shared staging pool owned by TVdxCompute before each dispatch.
  // Valid only while the source TVdxGGUFReader is open.
  TVdxFFNLayerWeights = record
    GateWeightPtr:   PByte;
    UpWeightPtr:     PByte;
    DownWeightPtr:   PByte;
    GateWeightBytes: UInt64;
    UpWeightBytes:   UInt64;
    DownWeightBytes: UInt64;
    WeightType:      TVdxGGMLType;
  end;

  { TVdxFFN }
  // Streaming FFN compute — gate / up / down projection weights are NOT
  // permanently uploaded to per-layer VRAM. Each forward call copies the
  // slice it needs from the mmap'd GGUF into the shared staging buffer
  // pairs owned by TVdxCompute, then dispatches. See
  // .claude/tasks/TASK-REFACTOR.md Phase 8 for the full design.
  //
  // Owns MATVEC_F16 / MATVEC_Q8_0 / MATVEC_Q4_0 shaders for single-token
  // decode, MATMUL_F16 / MATMUL_Q8_0 / MATMUL_Q4_0 for prefill batch, plus
  // GELU_MUL for the element-wise activation step between up-proj and
  // down-proj. Keeps its own private scratch buffers FGateBuf / FUpBuf
  // [FFNWidth] F32 for the single-token path; the batch path uses caller-
  // provided workspace matrices so they can be reused across layers.
  //
  // Batching contract — Forward / ForwardBatch MUST be called inside an
  // active batch (FCompute.BeginBatch / EndBatch). They record CopyBuffer
  // and dispatch commands into staging pool pairs 0 / 1 / 2 (gate / up /
  // down). Callers must ensure no other streaming consumer holds those
  // pair indices within the same batch submission — all CPU memcpys
  // happen at record time before EndBatch, and reusing a pair's host
  // buffer mid-batch would overwrite bytes the GPU has not yet copied.
  // Under the expected model-level orchestration, Attention and FFN run
  // in separate batches, so the same 3 pair indices are reused safely
  // across both.
  TVdxFFN = class(TVdxBaseObject)
  private
    FCompute:     TVdxCompute;
    FInitialized: Boolean;

    // Model dimensions — stashed by Init, consumed by everything else.
    FHiddenDim: UInt32;
    FFFNWidth:  UInt32;

    // Shader modules — single-token decode path
    FMatVecShader:   VkShaderModule;
    FMatVecQ8Shader: VkShaderModule;
    FMatVecQ4Shader: VkShaderModule;

    // Shader modules — batch matmul (prefill)
    FMatMulF16Shader: VkShaderModule;
    FMatMulQ8Shader:  VkShaderModule;
    FMatMulQ4Shader:  VkShaderModule;

    // Shader module — GELU-tanh(gate) * up in-place
    FGeluMulShader: VkShaderModule;

    // Pipeline bundles — single-token decode path
    FMatVecBundle:   TVdxComputePipelineBundle;
    FMatVecQ8Bundle: TVdxComputePipelineBundle;
    FMatVecQ4Bundle: TVdxComputePipelineBundle;

    // Pipeline bundles — batch matmul
    FMatMulF16Bundle: TVdxComputePipelineBundle;
    FMatMulQ8Bundle:  TVdxComputePipelineBundle;
    FMatMulQ4Bundle:  TVdxComputePipelineBundle;

    // Pipeline bundle — GELU-mul
    FGeluMulBundle: TVdxComputePipelineBundle;

    // Descriptor set layouts
    FMatVecDescLayout:  VkDescriptorSetLayout;   // 3 bindings (weight, in, out)
    FGeluMulDescLayout: VkDescriptorSetLayout;   // 2 bindings (gate, up)

    // Descriptor pool + pre-allocated sets (rebound per dispatch)
    FDescPool:       VkDescriptorPool;
    FMatVecDescSet:  VkDescriptorSet;
    FGeluMulDescSet: VkDescriptorSet;

    // Scratch buffers — reused every single-token Forward call. Batch
    // path uses caller-provided AGateMat / AUpMat instead.
    FGateBuf: TVdxGpuBuffer;   // [FFNWidth] F32
    FUpBuf:   TVdxGpuBuffer;   // [FFNWidth] F32

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

    // Initialize all shaders, pipelines, scratch buffers, and grow the
    // shared streaming-staging pool on TVdxCompute to the largest slice
    // this unit might ever upload (sized to the F16 upper bound — smaller
    // quant types fit without regrowth). Three pairs requested because
    // gate / up / down are concurrent in-flight within one FFN batch.
    // Returns False with FErrors populated on failure; partially-
    // constructed state is rolled back via Cleanup.
    function Init(const ACompute: TVdxCompute;
      const AHiddenDim: UInt32;
      const AFFNWidth: UInt32): Boolean;

    // Release all GPU resources. Safe to call on an uninitialized,
    // partially-initialized, or already-cleaned-up instance.
    procedure Cleanup();

    // Resolve mmap pointers + byte sizes for one layer's gate / up / down
    // projection tensors. Zero GPU allocation — the returned record just
    // references the GGUF mapping. Pointers are valid until AReader is
    // closed.
    function ResolveFFNWeights(const AReader: TVdxGGUFReader;
      const ALayerIndex: Integer;
      out AWeights: TVdxFFNLayerWeights): Boolean;

    // Run full FFN for one layer at one position. Must be called inside
    // an active batch (FCompute.BeginBatch / EndBatch). Streams gate
    // through staging pair 0, up through pair 1, down through pair 2.
    //
    // AInputBuf  : pre-normed residual [HiddenDim] F32
    // AWeights   : mmap pointers + byte sizes from ResolveFFNWeights
    // AOutputBuf : FFN output [HiddenDim] F32 — caller adds this to the
    //              residual stream (post-FFN-norm happens outside)
    procedure Forward(const AInputBuf: TVdxGpuBuffer;
      const AWeights: TVdxFFNLayerWeights;
      const AOutputBuf: TVdxGpuBuffer);

    // Batched prefill path for N tokens through one layer. Same batch-mode
    // contract as Forward — caller wraps in BeginBatch / EndBatch. Streams
    // gate / up / down weight slices through staging pairs 0 / 1 / 2.
    //
    // AInputMat  : pre-normed residual [NumTokens x HiddenDim] F32
    // AGateMat   : workspace [NumTokens x FFNWidth] F32, caller-owned
    // AUpMat     : workspace [NumTokens x FFNWidth] F32, caller-owned
    // AOutputMat : output [NumTokens x HiddenDim] F32 — caller adds to
    //              residual
    procedure ForwardBatch(const AInputMat: TVdxGpuBuffer;
      const AWeights: TVdxFFNLayerWeights;
      const ANumTokens: UInt32;
      const AGateMat: TVdxGpuBuffer;
      const AUpMat: TVdxGpuBuffer;
      const AOutputMat: TVdxGpuBuffer);

    property Initialized: Boolean read FInitialized;
  end;

implementation

{ TVdxFFN }

constructor TVdxFFN.Create();
begin
  inherited;

  FCompute     := nil;
  FInitialized := False;

  FHiddenDim := 0;
  FFFNWidth  := 0;

  // Shader modules
  FMatVecShader    := VK_NULL_HANDLE;
  FMatVecQ8Shader  := VK_NULL_HANDLE;
  FMatVecQ4Shader  := VK_NULL_HANDLE;
  FMatMulF16Shader := VK_NULL_HANDLE;
  FMatMulQ8Shader  := VK_NULL_HANDLE;
  FMatMulQ4Shader  := VK_NULL_HANDLE;
  FGeluMulShader   := VK_NULL_HANDLE;

  // Pipeline bundles
  FMatVecBundle    := Default(TVdxComputePipelineBundle);
  FMatVecQ8Bundle  := Default(TVdxComputePipelineBundle);
  FMatVecQ4Bundle  := Default(TVdxComputePipelineBundle);
  FMatMulF16Bundle := Default(TVdxComputePipelineBundle);
  FMatMulQ8Bundle  := Default(TVdxComputePipelineBundle);
  FMatMulQ4Bundle  := Default(TVdxComputePipelineBundle);
  FGeluMulBundle   := Default(TVdxComputePipelineBundle);

  // Descriptor layouts + pool + sets
  FMatVecDescLayout  := VK_NULL_HANDLE;
  FGeluMulDescLayout := VK_NULL_HANDLE;
  FDescPool          := VK_NULL_HANDLE;
  FMatVecDescSet     := VK_NULL_HANDLE;
  FGeluMulDescSet    := VK_NULL_HANDLE;

  // Scratch buffers
  FGateBuf := Default(TVdxGpuBuffer);
  FUpBuf   := Default(TVdxGpuBuffer);
end;

destructor TVdxFFN.Destroy();
begin
  Cleanup();
  inherited;
end;

function TVdxFFN.LoadShader(const AName: string): VkShaderModule;
var
  LSpv: TBytes;
begin
  LSpv := VdxLoadShader(AName);
  Result := FCompute.CreateShaderModule(@LSpv[0], NativeUInt(Length(LSpv)));
end;

procedure TVdxFFN.Cleanup();
begin
  // Cleanup is safe on partially-constructed state. Every handle is
  // checked against its null sentinel before destroy. FCompute may
  // still be nil if Init never ran — guard at the top.
  if FCompute = nil then
  begin
    FInitialized := False;
    Exit;
  end;

  // Scratch buffers first
  if FGateBuf.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(FGateBuf);
  if FUpBuf.Buffer <> VK_NULL_HANDLE then
    FCompute.DestroyGpuBuffer(FUpBuf);

  // Descriptor pool — frees all sets it allocated (matvec + gelu-mul)
  if FDescPool <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorPoolHandle(FDescPool);

  // Pipelines — tear down before their layouts + shaders
  FCompute.DestroyComputePipelineBundle(FMatVecBundle);
  FCompute.DestroyComputePipelineBundle(FMatVecQ8Bundle);
  FCompute.DestroyComputePipelineBundle(FMatVecQ4Bundle);
  FCompute.DestroyComputePipelineBundle(FMatMulF16Bundle);
  FCompute.DestroyComputePipelineBundle(FMatMulQ8Bundle);
  FCompute.DestroyComputePipelineBundle(FMatMulQ4Bundle);
  FCompute.DestroyComputePipelineBundle(FGeluMulBundle);

  // Descriptor layouts
  if FMatVecDescLayout <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorSetLayoutHandle(FMatVecDescLayout);
  if FGeluMulDescLayout <> VK_NULL_HANDLE then
    FCompute.DestroyDescriptorSetLayoutHandle(FGeluMulDescLayout);

  // Shader modules
  if FMatVecShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FMatVecShader);
  if FMatVecQ8Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FMatVecQ8Shader);
  if FMatVecQ4Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FMatVecQ4Shader);
  if FMatMulF16Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FMatMulF16Shader);
  if FMatMulQ8Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FMatMulQ8Shader);
  if FMatMulQ4Shader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FMatMulQ4Shader);
  if FGeluMulShader <> VK_NULL_HANDLE then
    FCompute.DestroyShaderModuleHandle(FGeluMulShader);

  FInitialized := False;
  FCompute     := nil;
end;

function TVdxFFN.Init(const ACompute: TVdxCompute;
  const AHiddenDim: UInt32;
  const AFFNWidth: UInt32): Boolean;
var
  LDummyBuf:    TVdxGpuBuffer;
  LMaxQ8Blocks: UInt32;
  LMaxSliceF16: UInt64;
begin
  Result := False;

  if FInitialized then
  begin
    FErrors.Add(esError, VDX_ERROR_FFN_ALREADY_INIT, RSFFNAlreadyInit);
    Exit;
  end;

  if ACompute = nil then
  begin
    FErrors.Add(esFatal, VDX_ERROR_FFN_COMPUTE_NIL, RSFFNComputeNil);
    Exit;
  end;

  FCompute   := ACompute;
  FHiddenDim := AHiddenDim;
  FFFNWidth  := AFFNWidth;

  // Max Q8_0 blocks per row across all our matvec uses. Down projection
  // has in_dim = FFNWidth, which is larger than HiddenDim — it sets the
  // ceiling. Passed as spec constant to MATVEC_Q8_0 / MATMUL_Q8_0 so
  // their shared-memory array is sized correctly.
  LMaxQ8Blocks := FFFNWidth div 32;

  LDummyBuf := Default(TVdxGpuBuffer);

  Status('FFN: Init (H=%d F=%d)', [AHiddenDim, AFFNWidth]);

  // Outer try/finally guarantees Cleanup runs on any early-exit. Inner
  // try/except converts raises from VdxLoadShader or any other RTL call
  // into esFatal errors without propagating past this boundary.
  try
    try
      //------------------------------------------------------------------
      // Shaders — matvec (decode) + matmul (batch) + gelu-mul
      //------------------------------------------------------------------
      FMatVecShader := LoadShader('MATVEC_F16');
      if FErrors.HasFatal() then Exit;
      FMatVecQ8Shader := LoadShader('MATVEC_Q8_0');
      if FErrors.HasFatal() then Exit;
      FMatVecQ4Shader := LoadShader('MATVEC_Q4_0');
      if FErrors.HasFatal() then Exit;

      FMatMulF16Shader := LoadShader('MATMUL_F16');
      if FErrors.HasFatal() then Exit;
      FMatMulQ8Shader := LoadShader('MATMUL_Q8_0');
      if FErrors.HasFatal() then Exit;
      FMatMulQ4Shader := LoadShader('MATMUL_Q4_0');
      if FErrors.HasFatal() then Exit;

      FGeluMulShader := LoadShader('GELU_MUL');
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Descriptor set layouts
      //------------------------------------------------------------------
      FMatVecDescLayout  := FCompute.CreateStorageDescriptorSetLayout(3);
      if FErrors.HasFatal() then Exit;
      FGeluMulDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Pipelines — matvec single-token
      //------------------------------------------------------------------
      FMatVecBundle := FCompute.CreateComputePipelineWithPush(
        FMatVecShader, 'main', FMatVecDescLayout,
        SizeOf(TVdxFFNMatVecPush));
      if FErrors.HasFatal() then Exit;
      FMatVecQ8Bundle := FCompute.CreateComputePipelineWithPushAndSpec(
        FMatVecQ8Shader, 'main', FMatVecDescLayout,
        SizeOf(TVdxFFNMatVecPush), LMaxQ8Blocks);
      if FErrors.HasFatal() then Exit;
      FMatVecQ4Bundle := FCompute.CreateComputePipelineWithPush(
        FMatVecQ4Shader, 'main', FMatVecDescLayout,
        SizeOf(TVdxFFNMatVecPush));
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Pipelines — matmul batch. Reuse FMatVecDescLayout (same 3-binding
      // shape: weight, input, output — matches the matmul shader slots).
      //------------------------------------------------------------------
      FMatMulF16Bundle := FCompute.CreateComputePipelineWithPush(
        FMatMulF16Shader, 'main', FMatVecDescLayout,
        SizeOf(TVdxFFNMatMulPush));
      if FErrors.HasFatal() then Exit;
      FMatMulQ8Bundle := FCompute.CreateComputePipelineWithPushAndSpec(
        FMatMulQ8Shader, 'main', FMatVecDescLayout,
        SizeOf(TVdxFFNMatMulPush), LMaxQ8Blocks);
      if FErrors.HasFatal() then Exit;
      FMatMulQ4Bundle := FCompute.CreateComputePipelineWithPush(
        FMatMulQ4Shader, 'main', FMatVecDescLayout,
        SizeOf(TVdxFFNMatMulPush));
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Pipeline — GELU-mul (2 bindings: in-place gate, read-only up)
      //------------------------------------------------------------------
      FGeluMulBundle := FCompute.CreateComputePipelineWithPush(
        FGeluMulShader, 'main', FGeluMulDescLayout,
        SizeOf(TVdxGeluMulPush));
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Scratch buffers — gate and up, both [FFNWidth] F32. Reused every
      // single-token Forward call. Batch path uses caller workspaces.
      //------------------------------------------------------------------
      FGateBuf := FCompute.CreateGpuBuffer(
        UInt64(FFFNWidth) * SizeOf(Single),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vcBuffer);
      if FErrors.HasFatal() then Exit;

      FUpBuf := FCompute.CreateGpuBuffer(
        UInt64(FFFNWidth) * SizeOf(Single),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vcBuffer);
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Descriptor pool + two reusable sets (matvec, gelu-mul).
      // Total descriptors: 3 + 2 = 5 storage buffers across 2 sets.
      //------------------------------------------------------------------
      FDescPool := FCompute.CreateDescriptorPoolForStorage(2, 5);
      if FErrors.HasFatal() then Exit;

      FMatVecDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FDescPool, FMatVecDescLayout, [LDummyBuf, LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;
      FGeluMulDescSet := FCompute.AllocateDescriptorSetForBuffers(
        FDescPool, FGeluMulDescLayout, [LDummyBuf, LDummyBuf]);
      if FErrors.HasFatal() then Exit;

      //------------------------------------------------------------------
      // Grow the streaming staging pool on TVdxCompute to fit the largest
      // gate / up / down projection slice. Three pairs because all three
      // projections are in flight within one FFN batch. Sized for F16
      // upper bound — Q8_0 / Q4_0 slices are smaller and fit without
      // regrowth. The pool is shared with TVdxAttention (4 pairs) — after
      // both units have initialized, the pool settles at max(4, 3) = 4
      // pairs of max(attn_max, ffn_max) bytes per pair.
      //
      //   Gate : [HiddenDim x FFNWidth]   = HiddenDim * FFNWidth * 2 bytes
      //   Up   : same shape as gate       = same bytes
      //   Down : [FFNWidth x HiddenDim]   = same bytes (transposed shape)
      //------------------------------------------------------------------
      LMaxSliceF16 := UInt64(FHiddenDim) * FFFNWidth * SizeOf(Word);
      if not FCompute.EnsureStagingPool(3, LMaxSliceF16) then Exit;

      FInitialized := True;
      Result       := True;
      Status('FFN: Ready (staging=3 pairs x %d bytes)', [LMaxSliceF16]);
    except
      on E: Exception do
        FErrors.Add(esFatal, VDX_ERROR_FFN_INIT_EXCEPTION,
          RSFFNInitException, [E.Message]);
    end;
  finally
    if not Result then
      Cleanup();
  end;
end;

function TVdxFFN.ResolveFFNWeights(const AReader: TVdxGGUFReader;
  const ALayerIndex: Integer;
  out AWeights: TVdxFFNLayerWeights): Boolean;

  // Resolve one gate / up / down tensor. Returns False with FErrors
  // populated on any failure (tensor missing, unsupported type, empty
  // slice). Outputs the mmap pointer and the computed byte size.
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
      FErrors.Add(esFatal, VDX_ERROR_FFN_TENSOR_NOT_FOUND,
        RSFFNTensorNotFound, [ATensorName]);
      Exit;
    end;

    if not AReader.GetTensorInfo(ATensorName, LInfo) then Exit;

    AType  := LInfo.TensorType;
    ABytes := VdxGGMLTensorBytes(LInfo.TensorType,
      LInfo.Dimensions[0], LInfo.Dimensions[1]);
    if ABytes = 0 then
    begin
      FErrors.Add(esFatal, VDX_ERROR_FFN_UNSUPPORTED_TYPE,
        RSFFNUnsupportedType,
        [ATensorName, VdxGGMLTypeName(LInfo.TensorType)]);
      Exit;
    end;

    APtr := AReader.GetTensorDataPtr(ATensorName);
    if APtr = nil then Exit;  // GGUFReader already logged the reason

    Result := True;
  end;

var
  LTypeGate: TVdxGGMLType;
  LTypeUp:   TVdxGGMLType;
  LTypeDown: TVdxGGMLType;
begin
  Result   := False;
  AWeights := Default(TVdxFFNLayerWeights);

  if not FInitialized then
  begin
    FErrors.Add(esError, VDX_ERROR_FFN_NOT_INIT, RSFFNNotInit);
    Exit;
  end;

  if AReader = nil then
  begin
    FErrors.Add(esFatal, VDX_ERROR_FFN_RESOLVE_EXCEPTION,
      RSFFNResolveException, ['AReader is nil']);
    Exit;
  end;

  try
    if not ResolveOne(
      Format('blk.%d.ffn_gate.weight', [ALayerIndex]),
      AWeights.GateWeightPtr, AWeights.GateWeightBytes, LTypeGate) then Exit;
    if not ResolveOne(
      Format('blk.%d.ffn_up.weight', [ALayerIndex]),
      AWeights.UpWeightPtr, AWeights.UpWeightBytes, LTypeUp) then Exit;
    if not ResolveOne(
      Format('blk.%d.ffn_down.weight', [ALayerIndex]),
      AWeights.DownWeightPtr, AWeights.DownWeightBytes, LTypeDown) then Exit;

    // All three projections share a single quant type in every model we
    // support. If a future model mixes quants across projections within
    // one layer, this check surfaces it loudly instead of silently
    // picking one.
    if (LTypeGate <> LTypeUp) or (LTypeGate <> LTypeDown) then
    begin
      FErrors.Add(esFatal, VDX_ERROR_FFN_UNSUPPORTED_TYPE,
        RSFFNUnsupportedType,
        [Format('layer %d mixed quant types', [ALayerIndex]),
         Format('gate=%s up=%s down=%s',
           [VdxGGMLTypeName(LTypeGate), VdxGGMLTypeName(LTypeUp),
            VdxGGMLTypeName(LTypeDown)])]);
      Exit;
    end;

    AWeights.WeightType := LTypeGate;
    Result := True;
  except
    on E: Exception do
    begin
      FErrors.Add(esFatal, VDX_ERROR_FFN_RESOLVE_EXCEPTION,
        RSFFNResolveException, [E.Message]);
      AWeights := Default(TVdxFFNLayerWeights);
    end;
  end;
end;

procedure TVdxFFN.DispatchMatVec(const AWeightBuf: TVdxGpuBuffer;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ATensorType: TVdxGGMLType);
var
  LPush:           TVdxFFNMatVecPush;
  LPipeline:       VkPipeline;
  LPipelineLayout: VkPipelineLayout;
begin
  // Rebind buffers to the pre-allocated matvec descriptor set.
  FCompute.UpdateDescriptorSetBuffers(FMatVecDescSet,
    [AWeightBuf, AInputBuf, AOutputBuf]);

  // Pipeline selection by quant: Q4_0 / Q8_0 use full in_dim, F16 uses
  // in_dim/2 (two halves packed per uint32 in the shader's load path).
  if ATensorType = gtQ4_0 then
  begin
    LPush.InDimParam := AInDim;
    LPipeline        := FMatVecQ4Bundle.Pipeline;
    LPipelineLayout  := FMatVecQ4Bundle.PipelineLayout;
  end
  else if ATensorType = gtQ8_0 then
  begin
    LPush.InDimParam := AInDim;
    LPipeline        := FMatVecQ8Bundle.Pipeline;
    LPipelineLayout  := FMatVecQ8Bundle.PipelineLayout;
  end
  else
  begin
    LPush.InDimParam := AInDim div 2;
    LPipeline        := FMatVecBundle.Pipeline;
    LPipelineLayout  := FMatVecBundle.PipelineLayout;
  end;

  LPush.OutDim := AOutDim;

  // One workgroup (256 threads) per output row.
  FCompute.DispatchComputeWithPush(
    LPipeline, LPipelineLayout, FMatVecDescSet,
    @LPush, SizeOf(LPush), AOutDim);
end;

procedure TVdxFFN.StreamAndDispatchMatVec(const AStagingIndex: UInt32;
  const AWeightPtr: PByte; const AWeightBytes: UInt64;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ATensorType: TVdxGGMLType);
var
  LHost:   TVdxGpuBuffer;
  LDevice: TVdxGpuBuffer;
begin
  // The indexed staging pair is the key to correct batched streaming:
  // each concurrent in-flight slice (gate at 0, up at 1, down at 2) owns
  // its own host memory, so the CPU memcpy for a later slice never
  // overwrites bytes the GPU will later copy from an earlier slice when
  // the batch submits.
  LHost   := FCompute.GetStagingHost(AStagingIndex);
  LDevice := FCompute.GetStagingDevice(AStagingIndex);

  // Step 1 — CPU memcpy from mmap region into host-visible staging.
  // The OS page-faults in weight pages on demand via the mmap; this is
  // the disk -> RAM hop.
  FCompute.UploadToBuffer(LHost, AWeightPtr, AWeightBytes);

  // Step 2 — GPU copy host -> device-local (batch-recorded).
  FCompute.CopyBuffer(LHost, LDevice, AWeightBytes);

  // Step 3 — barrier so the matvec dispatch waits for the copy.
  FCompute.BatchBarrier();

  // Step 4 — matvec against the device-local staging buffer.
  DispatchMatVec(LDevice, AInputBuf, AOutputBuf, AInDim, AOutDim, ATensorType);
end;

procedure TVdxFFN.DispatchBatchMatMul(const AWeightBuf: TVdxGpuBuffer;
  const AInputBuf: TVdxGpuBuffer; const AOutputBuf: TVdxGpuBuffer;
  const AInDim: UInt32; const AOutDim: UInt32;
  const ANumTokens: UInt32; const ATensorType: TVdxGGMLType);
var
  LPush:           TVdxFFNMatMulPush;
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

procedure TVdxFFN.StreamAndDispatchBatchMatMul(const AStagingIndex: UInt32;
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

procedure TVdxFFN.Forward(const AInputBuf: TVdxGpuBuffer;
  const AWeights: TVdxFFNLayerWeights;
  const AOutputBuf: TVdxGpuBuffer);
var
  LGeluPush: TVdxGeluMulPush;
begin
  //----------------------------------------------------------------
  // Step 1 — gate(x) → FGateBuf, streamed via pair 0.
  // Reads AInputBuf (pre-normed residual) [HiddenDim] F32.
  //----------------------------------------------------------------
  StreamAndDispatchMatVec(0,
    AWeights.GateWeightPtr, AWeights.GateWeightBytes,
    AInputBuf, FGateBuf,
    FHiddenDim, FFFNWidth, AWeights.WeightType);

  //----------------------------------------------------------------
  // Step 2 — up(x) → FUpBuf, streamed via pair 1.
  // Reads the same AInputBuf as gate, writes to a different output
  // buffer — no dependency on gate's output, so no inter-projection
  // barrier is needed beyond the one StreamAndDispatchMatVec already
  // inserts between its own copy and dispatch.
  //----------------------------------------------------------------
  StreamAndDispatchMatVec(1,
    AWeights.UpWeightPtr, AWeights.UpWeightBytes,
    AInputBuf, FUpBuf,
    FHiddenDim, FFFNWidth, AWeights.WeightType);
  FCompute.BatchBarrier();  // Gate/Up buffers ready for GELU-mul

  //----------------------------------------------------------------
  // Step 3 — FGateBuf = GELU-tanh(FGateBuf) * FUpBuf (in-place).
  // The shader reads binding 0 (gate) + binding 1 (up), computes
  // GELU-tanh(gate[i]) * up[i], and writes back to binding 0.
  //----------------------------------------------------------------
  LGeluPush.Count := FFFNWidth;
  FCompute.UpdateDescriptorSetBuffers(FGeluMulDescSet, [FGateBuf, FUpBuf]);
  FCompute.DispatchComputeWithPush(
    FGeluMulBundle.Pipeline, FGeluMulBundle.PipelineLayout,
    FGeluMulDescSet, @LGeluPush, SizeOf(LGeluPush),
    (FFFNWidth + 255) div 256);
  FCompute.BatchBarrier();  // GELU-mul result in FGateBuf, ready for down

  //----------------------------------------------------------------
  // Step 4 — down(hidden) → AOutputBuf, streamed via pair 2.
  // Reads FGateBuf (now holding GELU-mul result) [FFNWidth] F32,
  // writes [HiddenDim] F32 to caller-provided output.
  //----------------------------------------------------------------
  StreamAndDispatchMatVec(2,
    AWeights.DownWeightPtr, AWeights.DownWeightBytes,
    FGateBuf, AOutputBuf,
    FFFNWidth, FHiddenDim, AWeights.WeightType);
end;

procedure TVdxFFN.ForwardBatch(const AInputMat: TVdxGpuBuffer;
  const AWeights: TVdxFFNLayerWeights;
  const ANumTokens: UInt32;
  const AGateMat: TVdxGpuBuffer;
  const AUpMat: TVdxGpuBuffer;
  const AOutputMat: TVdxGpuBuffer);
var
  LGeluPush: TVdxGeluMulPush;
begin
  //----------------------------------------------------------------
  // Step 1 — gate matmul: AInputMat → AGateMat, streamed via pair 0.
  //----------------------------------------------------------------
  StreamAndDispatchBatchMatMul(0,
    AWeights.GateWeightPtr, AWeights.GateWeightBytes,
    AInputMat, AGateMat,
    FHiddenDim, FFFNWidth, ANumTokens, AWeights.WeightType);

  //----------------------------------------------------------------
  // Step 2 — up matmul: AInputMat → AUpMat, streamed via pair 1.
  //----------------------------------------------------------------
  StreamAndDispatchBatchMatMul(1,
    AWeights.UpWeightPtr, AWeights.UpWeightBytes,
    AInputMat, AUpMat,
    FHiddenDim, FFFNWidth, ANumTokens, AWeights.WeightType);
  FCompute.BatchBarrier();  // Gate/Up matrices ready for GELU-mul

  //----------------------------------------------------------------
  // Step 3 — AGateMat = GELU-tanh(AGateMat) * AUpMat (in-place).
  // Count is flattened across all tokens.
  //----------------------------------------------------------------
  LGeluPush.Count := ANumTokens * FFFNWidth;
  FCompute.UpdateDescriptorSetBuffers(FGeluMulDescSet, [AGateMat, AUpMat]);
  FCompute.DispatchComputeWithPush(
    FGeluMulBundle.Pipeline, FGeluMulBundle.PipelineLayout,
    FGeluMulDescSet, @LGeluPush, SizeOf(LGeluPush),
    (ANumTokens * FFFNWidth + 255) div 256);
  FCompute.BatchBarrier();  // GELU-mul result in AGateMat, ready for down

  //----------------------------------------------------------------
  // Step 4 — down matmul: AGateMat → AOutputMat, streamed via pair 2.
  //----------------------------------------------------------------
  StreamAndDispatchBatchMatMul(2,
    AWeights.DownWeightPtr, AWeights.DownWeightBytes,
    AGateMat, AOutputMat,
    FFFNWidth, FHiddenDim, ANumTokens, AWeights.WeightType);
end;

end.
