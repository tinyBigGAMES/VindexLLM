unit UVdxTestbed;

interface

procedure RunVdxTestbed();

implementation

uses
  System.SysUtils,
  System.IOUtils,
  VindexLLM.Utils,
  VindexLLM.VulkanCompute,
  VindexLLM.GGUFReader,
  VindexLLM.Vindex,
  VindexLLM.KNNWalk,
  VindexLLM.LayerNorm,
  VindexLLM.Attention,
  VindexLLM.Tokenizer;

// ============================================================================
//  Embedded SPIR-V: "double every float" compute shader
//  GLSL equivalent:
//    #version 450
//    layout(local_size_x = 4) in;
//    layout(std430, binding = 0) buffer DataBuf { float data[]; };
//    void main() {
//      uint idx = gl_GlobalInvocationID.x;
//      data[idx] = data[idx] * 2.0;
//    }
// ============================================================================

const
  CDoubleShaderSize = 544;
  CDoubleShader: array[0..135] of UInt32 = (
    $07230203, $00010000, $00000000, $00000017, $00000000, $00020011,
    $00000001, $0003000E, $00000000, $00000001, $0006000F, $00000005,
    $00000010, $6E69616D, $00000000, $0000000E, $00060010, $00000010,
    $00000011, $00000004, $00000001, $00000001, $00040047, $0000000E,
    $0000000B, $0000001C, $00030047, $00000009, $00000003, $00050048,
    $00000009, $00000000, $00000023, $00000000, $00040047, $00000008,
    $00000006, $00000004, $00040047, $0000000F, $00000022, $00000000,
    $00040047, $0000000F, $00000021, $00000000, $00020013, $00000001,
    $00030021, $00000002, $00000001, $00040015, $00000003, $00000020,
    $00000000, $00040017, $00000004, $00000003, $00000003, $00040020,
    $00000005, $00000001, $00000004, $00040020, $00000006, $00000001,
    $00000003, $00030016, $00000007, $00000020, $0003001D, $00000008,
    $00000007, $0003001E, $00000009, $00000008, $00040020, $0000000A,
    $00000002, $00000009, $00040020, $0000000B, $00000002, $00000007,
    $0004002B, $00000007, $0000000C, $40000000, $0004002B, $00000003,
    $0000000D, $00000000, $0004003B, $00000005, $0000000E, $00000001,
    $0004003B, $0000000A, $0000000F, $00000002, $00050036, $00000001,
    $00000010, $00000000, $00000002, $000200F8, $00000011, $00050041,
    $00000006, $00000012, $0000000E, $0000000D, $0004003D, $00000003,
    $00000013, $00000012, $00060041, $0000000B, $00000014, $0000000F,
    $0000000D, $00000013, $0004003D, $00000007, $00000015, $00000014,
    $00050085, $00000007, $00000016, $00000015, $0000000C, $0003003E,
    $00000014, $00000016, $000100FD, $00010038
  );

procedure StatusCallback(const AText: string; const AUserData: Pointer);
begin
  TVdxUtils.PrintLn(AText);
end;

// ============================================================================
//  Test 01 — Basic Vulkan init (already verified)
// ============================================================================

procedure Test01();
var
  LCompute: TVdxVulkanCompute;
begin
  LCompute := TVdxVulkanCompute.Create();
  try
    LCompute.SetStatusCallback(StatusCallback);
    LCompute.Init();
    TvdxUtils.Pause();
  finally
    LCompute.Free();
  end;
end;

// ============================================================================
//  Test 02 — GPU compute round-trip: upload, dispatch, download, verify
// ============================================================================

procedure Test02();
const
  CFloatCount = 4;
  CBufferSize = CFloatCount * SizeOf(Single);
  CInputData: array[0..CFloatCount - 1] of Single = (1.0, 2.0, 3.0, 4.0);
  CExpected:  array[0..CFloatCount - 1] of Single = (2.0, 4.0, 6.0, 8.0);
var
  LCompute: TVdxVulkanCompute;
  LBuffer: TVdxGpuBuffer;
  LShader: VkShaderModule;
  LDescLayout: VkDescriptorSetLayout;
  LPipeline: TVdxComputePipelineBundle;
  LDescPool: VkDescriptorPool;
  LDescSet: VkDescriptorSet;
  LOutput: array[0..CFloatCount - 1] of Single;
  LI: Integer;
  LPassed: Boolean;
begin
  LCompute := TVdxVulkanCompute.Create();
  try
    LCompute.SetStatusCallback(StatusCallback);
    LCompute.Init();

    // Create host-visible storage buffer
    TVdxUtils.PrintLn('Creating GPU buffer (%d bytes)...', [CBufferSize]);
    LBuffer := LCompute.CreateGpuBuffer(
      CBufferSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Upload input data
    TVdxUtils.PrintLn('Uploading [%.1f, %.1f, %.1f, %.1f]...',
      [CInputData[0], CInputData[1], CInputData[2], CInputData[3]]);
    LCompute.UploadToBuffer(LBuffer, @CInputData[0], CBufferSize);

    // Create shader module from embedded SPIR-V
    TVdxUtils.PrintLn('Creating shader module (%d bytes SPIR-V)...', [CDoubleShaderSize]);
    LShader := LCompute.CreateShaderModule(@CDoubleShader[0], CDoubleShaderSize);

    // Create descriptor set layout (1 storage buffer binding)
    LDescLayout := LCompute.CreateStorageDescriptorSetLayout(1);

    // Create compute pipeline
    TVdxUtils.PrintLn('Creating compute pipeline...');
    LPipeline := LCompute.CreateComputePipelineSimple(LShader, 'main', LDescLayout);

    // Create descriptor pool and allocate descriptor set
    LDescPool := LCompute.CreateDescriptorPoolForStorage(1, 1);
    LDescSet := LCompute.AllocateDescriptorSetForBuffers(LDescPool, LDescLayout, [LBuffer]);

    // Dispatch: 1 workgroup (local_size_x=4 covers all 4 floats)
    TVdxUtils.PrintLn('Dispatching compute shader (1 workgroup)...');
    LCompute.DispatchCompute(
      LPipeline.Pipeline,
      LPipeline.PipelineLayout,
      LDescSet,
      1  // 1 workgroup of 4 invocations
    );

    // Download results
    FillChar(LOutput, SizeOf(LOutput), 0);
    LCompute.DownloadFromBuffer(LBuffer, @LOutput[0], CBufferSize);

    TVdxUtils.PrintLn('Result: [%.1f, %.1f, %.1f, %.1f]',
      [LOutput[0], LOutput[1], LOutput[2], LOutput[3]]);

    // Verify
    LPassed := True;
    for LI := 0 to CFloatCount - 1 do
    begin
      if Abs(LOutput[LI] - CExpected[LI]) > 0.001 then
      begin
        TVdxUtils.PrintLn(COLOR_RED + 'MISMATCH at [%d]: expected %.1f, got %.1f',
          [LI, CExpected[LI], LOutput[LI]]);
        LPassed := False;
      end;
    end;

    if LPassed then
      TVdxUtils.PrintLn(COLOR_GREEN + 'TEST 02 PASSED: GPU compute round-trip verified!')
    else
      TVdxUtils.PrintLn(COLOR_RED + 'TEST 02 FAILED: Output mismatch');

    // Cleanup Vulkan objects (reverse order of creation)
    LCompute.DestroyDescriptorPoolHandle(LDescPool);
    LCompute.DestroyComputePipelineBundle(LPipeline);
    LCompute.DestroyDescriptorSetLayoutHandle(LDescLayout);
    LCompute.DestroyShaderModuleHandle(LShader);
    LCompute.DestroyGpuBuffer(LBuffer);

    TVdxUtils.Pause();
  finally
    LCompute.Free();
  end;
end;

// ============================================================================
//  Test 03 — GPU compute round-trip with SPIR-V loaded from disk
// ============================================================================

procedure Test03();
const
  CFloatCount = 4;
  CBufferSize = CFloatCount * SizeOf(Single);
  CInputData: array[0..CFloatCount - 1] of Single = (1.0, 2.0, 3.0, 4.0);
  CExpected:  array[0..CFloatCount - 1] of Single = (2.0, 4.0, 6.0, 8.0);
var
  LCompute: TVdxVulkanCompute;
  LBuffer: TVdxGpuBuffer;
  LShader: VkShaderModule;
  LDescLayout: VkDescriptorSetLayout;
  LPipeline: TVdxComputePipelineBundle;
  LDescPool: VkDescriptorPool;
  LDescSet: VkDescriptorSet;
  LOutput: array[0..CFloatCount - 1] of Single;
  LSpvPath: string;
  LSpvBytes: TBytes;
  LI: Integer;
  LPassed: Boolean;
begin
  LCompute := TVdxVulkanCompute.Create();
  try
    LCompute.SetStatusCallback(StatusCallback);
    LCompute.Init();

    // Resolve path to .spv relative to exe location
    LSpvPath := TPath.Combine(
      TPath.GetDirectoryName(ParamStr(0)),
      '..\shaders\double_floats.spv'
    );
    LSpvPath := TPath.GetFullPath(LSpvPath);

    TVdxUtils.PrintLn('Loading SPIR-V from: %s', [LSpvPath]);
    TVdxUtils.FailIf(not TFile.Exists(LSpvPath),
      'SPIR-V file not found: %s', [LSpvPath]);

    LSpvBytes := TFile.ReadAllBytes(LSpvPath);
    TVdxUtils.PrintLn('Loaded %d bytes of SPIR-V', [Length(LSpvBytes)]);

    // Create buffer and upload
    LBuffer := LCompute.CreateGpuBuffer(
      CBufferSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    TVdxUtils.PrintLn('Uploading [%.1f, %.1f, %.1f, %.1f]...',
      [CInputData[0], CInputData[1], CInputData[2], CInputData[3]]);
    LCompute.UploadToBuffer(LBuffer, @CInputData[0], CBufferSize);

    // Create shader from file bytes
    LShader := LCompute.CreateShaderModule(@LSpvBytes[0], NativeUInt(Length(LSpvBytes)));

    // Pipeline setup
    LDescLayout := LCompute.CreateStorageDescriptorSetLayout(1);
    LPipeline := LCompute.CreateComputePipelineSimple(LShader, 'main', LDescLayout);
    LDescPool := LCompute.CreateDescriptorPoolForStorage(1, 1);
    LDescSet := LCompute.AllocateDescriptorSetForBuffers(LDescPool, LDescLayout, [LBuffer]);

    // Dispatch
    TVdxUtils.PrintLn('Dispatching compute shader (1 workgroup)...');
    LCompute.DispatchCompute(
      LPipeline.Pipeline,
      LPipeline.PipelineLayout,
      LDescSet,
      1
    );

    // Download and verify
    FillChar(LOutput, SizeOf(LOutput), 0);
    LCompute.DownloadFromBuffer(LBuffer, @LOutput[0], CBufferSize);

    TVdxUtils.PrintLn('Result: [%.1f, %.1f, %.1f, %.1f]',
      [LOutput[0], LOutput[1], LOutput[2], LOutput[3]]);

    LPassed := True;
    for LI := 0 to CFloatCount - 1 do
    begin
      if Abs(LOutput[LI] - CExpected[LI]) > 0.001 then
      begin
        TVdxUtils.PrintLn(COLOR_RED + 'MISMATCH at [%d]: expected %.1f, got %.1f',
          [LI, CExpected[LI], LOutput[LI]]);
        LPassed := False;
      end;
    end;

    if LPassed then
      TVdxUtils.PrintLn(COLOR_GREEN + 'TEST 03 PASSED: File-loaded SPIR-V round-trip verified!')
    else
      TVdxUtils.PrintLn(COLOR_RED + 'TEST 03 FAILED: Output mismatch');

    // Cleanup
    LCompute.DestroyDescriptorPoolHandle(LDescPool);
    LCompute.DestroyComputePipelineBundle(LPipeline);
    LCompute.DestroyDescriptorSetLayoutHandle(LDescLayout);
    LCompute.DestroyShaderModuleHandle(LShader);
    LCompute.DestroyGpuBuffer(LBuffer);

    TVdxUtils.Pause();
  finally
    LCompute.Free();
  end;
end;

// ============================================================================
//  Test 04 — GGUF Reader: parse Gemma 3 4B F16
// ============================================================================

procedure Test04();
const
  CGGUFPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';
var
  LReader: TVdxGGUFReader;
  LTensor: TVdxGGUFTensorInfo;
  LPtr: Pointer;
begin
  LReader := TVdxGGUFReader.Create();
  try
    LReader.SetStatusCallback(StatusCallback);

    TVdxUtils.PrintLn(COLOR_CYAN + '=== Test 04: GGUF Reader ===');
    TVdxUtils.PrintLn('File: %s', [CGGUFPath]);
    TVdxUtils.PrintLn('');

    if not LReader.Open(CGGUFPath) then
    begin
      TVdxUtils.PrintLn(COLOR_RED + 'TEST 04 FAILED: Could not open GGUF file');
      Exit;
    end;

    // Summary
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_GREEN + '--- Summary ---');
    TVdxUtils.PrintLn('  Version:    %d', [LReader.GetVersion()]);
    TVdxUtils.PrintLn('  Tensors:    %d', [LReader.GetTensorCount()]);
    TVdxUtils.PrintLn('  Metadata:   %d', [LReader.GetMetadataCount()]);
    TVdxUtils.PrintLn('  Alignment:  %d', [LReader.GetAlignment()]);
    TVdxUtils.PrintLn('  File size:  %.2f GB', [LReader.GetFileSize() / (1024.0 * 1024.0 * 1024.0)]);

    // Check key metadata
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_GREEN + '--- Key Metadata ---');
    TVdxUtils.PrintLn('  Architecture: %s', [LReader.GetMetadataString('general.architecture')]);
    TVdxUtils.PrintLn('  Model name:   %s', [LReader.GetMetadataString('general.name')]);

    // Spot-check a known FFN tensor
    if LReader.HasTensor('blk.0.ffn_gate.weight') then
    begin
      TVdxUtils.PrintLn('');
      TVdxUtils.PrintLn(COLOR_GREEN + '--- Tensor Spot Check ---');
      LTensor := LReader.GetTensorInfo('blk.0.ffn_gate.weight');
      TVdxUtils.PrintLn('  blk.0.ffn_gate.weight: %s [%d x %d] offset=%d',
        [VdxGGMLTypeName(LTensor.TensorType),
         LTensor.Dimensions[0], LTensor.Dimensions[1],
         LTensor.DataOffset]);

      LPtr := LReader.GetTensorDataPtr('blk.0.ffn_gate.weight');
      TVdxUtils.PrintLn('  Data pointer: $%p', [LPtr]);

      if LPtr <> nil then
        TVdxUtils.PrintLn(COLOR_GREEN + 'TEST 04 PASSED: GGUF parsed and tensor data accessible!')
      else
        TVdxUtils.PrintLn(COLOR_RED + 'TEST 04 FAILED: Tensor data pointer is nil');
    end
    else
      TVdxUtils.PrintLn(COLOR_RED + 'TEST 04 FAILED: blk.0.ffn_gate.weight not found');

    LReader.Close();
    TVdxUtils.Pause();
  finally
    LReader.Free();
  end;
end;

// ============================================================================
//  Test 05 — Vindex: build FFN layer view + GPU upload round-trip
// ============================================================================

procedure Test05();
const
  CGGUFPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';
  CVerifyCount = 16;  // Verify first 16 float16 values (32 bytes)
  CVerifyBytes = CVerifyCount * 2;
var
  LReader: TVdxGGUFReader;
  LCompute: TVdxVulkanCompute;
  LVindex: TVdxVindex;
  LLayer: TVdxFFNLayerView;
  LStaging: TVdxGpuBuffer;
  LReadback: array[0..CVerifyCount - 1] of UInt16;
  LPassed: Boolean;
begin
  TVdxUtils.PrintLn(COLOR_CYAN + '=== Test 05: Vindex — FFN Layer View + GPU Upload ===');
  TVdxUtils.PrintLn('');

  // Phase 1: Parse GGUF and build vindex
  LReader := TVdxGGUFReader.Create();
  LCompute := TVdxVulkanCompute.Create();
  LVindex := TVdxVindex.Create();
  try
    LReader.SetStatusCallback(StatusCallback);
    LCompute.SetStatusCallback(StatusCallback);

    TVdxUtils.FailIf(not LReader.Open(CGGUFPath),
      'Failed to open GGUF: %s', [CGGUFPath]);

    TVdxUtils.PrintLn('Building vindex from GGUF...');
    TVdxUtils.FailIf(not LVindex.BuildFromGGUF(LReader),
      'Failed to build vindex from GGUF', []);

    // Print summary
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_GREEN + '--- Vindex Summary ---');
    TVdxUtils.PrintLn('  Layers:       %d', [LVindex.GetLayerCount()]);
    TVdxUtils.PrintLn('  Hidden dim:   %d', [LVindex.GetHiddenDim()]);
    TVdxUtils.PrintLn('  FFN width:    %d', [LVindex.GetFFNWidth()]);

    LLayer := LVindex.GetLayer(0);
    TVdxUtils.PrintLn('  Gate type:    %s', [VdxGGMLTypeName(LLayer.GateType)]);
    TVdxUtils.PrintLn('  Down type:    %s', [VdxGGMLTypeName(LLayer.DownType)]);
    TVdxUtils.PrintLn('  Gate size:    %.2f MB', [LLayer.GateSizeBytes / (1024.0 * 1024.0)]);
    TVdxUtils.PrintLn('  Down size:    %.2f MB', [LLayer.DownSizeBytes / (1024.0 * 1024.0)]);

    // Phase 2: Init Vulkan and upload layer 0
    TVdxUtils.PrintLn('');
    LCompute.Init();

    TVdxUtils.PrintLn('Uploading layer 0 gate+down to device-local VRAM...');
    LVindex.UploadLayer(0, LCompute);
    TVdxUtils.PrintLn(COLOR_GREEN + 'Upload complete.');

    // Phase 3: Read back gate data and verify against mmap'd source
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('Verifying GPU data (reading back first %d float16 values)...', [CVerifyCount]);

    LLayer := LVindex.GetLayer(0);

    // Create staging buffer, copy device-local → staging, download to CPU
    LStaging := LCompute.CreateGpuBuffer(
      CVerifyBytes,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    try
      LCompute.CopyBuffer(LLayer.GateGpuBuffer, LStaging, CVerifyBytes);
      FillChar(LReadback, SizeOf(LReadback), 0);
      LCompute.DownloadFromBuffer(LStaging, @LReadback[0], CVerifyBytes);
    finally
      LCompute.DestroyGpuBuffer(LStaging);
    end;

    // Compare with source mmap data using direct memory comparison
    LPassed := CompareMem(@LReadback[0], LLayer.GatePtr, CVerifyBytes);

    if LPassed then
    begin
      TVdxUtils.PrintLn(COLOR_GREEN + '  All %d values match!', [CVerifyCount]);
      TVdxUtils.PrintLn('  First 4 raw F16 values: $%04X $%04X $%04X $%04X',
        [LReadback[0], LReadback[1], LReadback[2], LReadback[3]]);
      TVdxUtils.PrintLn('');
      TVdxUtils.PrintLn(COLOR_GREEN + 'TEST 05 PASSED: Vindex build + GPU upload round-trip verified!');
    end
    else
      TVdxUtils.PrintLn(COLOR_RED + 'TEST 05 FAILED: GPU readback mismatch');

    // Cleanup GPU resources
    LVindex.FreeAllGpu(LCompute);
    LReader.Close();

  finally
    LVindex.Free();
    LCompute.Free();
    LReader.Free();
  end;
end;

// ============================================================================
//  Test 06 — KNNWalk: gate scan → top-K → accumulate on layer 0
// ============================================================================

procedure Test06();
const
  CGGUFPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';
  CTopK = 128;
var
  LReader: TVdxGGUFReader;
  LCompute: TVdxVulkanCompute;
  LVindex: TVdxVindex;
  LWalk: TVdxKNNWalk;
  LLayer: TVdxFFNLayerView;
  LResidualIn: array of Single;
  LResidualOut: array of Single;
  LHiddenDim: UInt64;
  LI: Integer;
  LNormIn: Double;
  LNormOut: Double;
  LDelta: Double;
  LNonZero: Integer;
begin
  TVdxUtils.PrintLn(COLOR_CYAN + '=== Test 06: KNNWalk — FFN Walk on Layer 0 ===');
  TVdxUtils.PrintLn('');

  LReader := TVdxGGUFReader.Create();
  LCompute := TVdxVulkanCompute.Create();
  LVindex := TVdxVindex.Create();
  LWalk := TVdxKNNWalk.Create();
  try
    LReader.SetStatusCallback(StatusCallback);
    LCompute.SetStatusCallback(StatusCallback);
    LWalk.SetStatusCallback(StatusCallback);

    // Parse GGUF and build vindex
    TVdxUtils.FailIf(not LReader.Open(CGGUFPath),
      'Failed to open GGUF: %s', [CGGUFPath]);
    TVdxUtils.FailIf(not LVindex.BuildFromGGUF(LReader),
      'Failed to build vindex', []);

    LHiddenDim := LVindex.GetHiddenDim();

    // Init Vulkan and upload layer 0 (gate + down for now)
    LCompute.Init();
    TVdxUtils.PrintLn('Uploading layer 0 to VRAM...');
    LVindex.UploadLayer(0, LCompute);

    // Init KNN walk engine
    LWalk.Init(LCompute, LVindex.GetHiddenDim(), LVindex.GetFFNWidth(), CTopK);

    // Create input residual: unit vector along dim 0
    SetLength(LResidualIn, LHiddenDim);
    SetLength(LResidualOut, LHiddenDim);
    FillChar(LResidualIn[0], LHiddenDim * SizeOf(Single), 0);
    LResidualIn[0] := 1.0;

    // Compute input L2 norm
    LNormIn := 0.0;
    for LI := 0 to LHiddenDim - 1 do
      LNormIn := LNormIn + LResidualIn[LI] * LResidualIn[LI];
    LNormIn := Sqrt(LNormIn);

    // Run the walk
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('Setting residual (unit vector dim 0)...');
    LWalk.SetResidual(@LResidualIn[0]);

    TVdxUtils.PrintLn('Walking layer 0 (topK=%d)...', [CTopK]);
    LWalk.WalkLayer(LVindex.GetLayer(0));

    TVdxUtils.PrintLn('Reading result...');
    LWalk.GetResidual(@LResidualOut[0]);

    // Compute output L2 norm and delta
    LNormOut := 0.0;
    LNonZero := 0;
    for LI := 0 to LHiddenDim - 1 do
    begin
      LNormOut := LNormOut + LResidualOut[LI] * LResidualOut[LI];
      if Abs(LResidualOut[LI] - LResidualIn[LI]) > 1e-9 then
        Inc(LNonZero);
    end;
    LNormOut := Sqrt(LNormOut);
    LDelta := LNormOut - LNormIn;

    // Report
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_GREEN + '--- Results ---');
    TVdxUtils.PrintLn('  Input L2 norm:    %.6f', [LNormIn]);
    TVdxUtils.PrintLn('  Output L2 norm:   %.6f', [LNormOut]);
    TVdxUtils.PrintLn('  Norm delta:       %.6f', [LDelta]);
    TVdxUtils.PrintLn('  Dims changed:     %d / %d', [LNonZero, LHiddenDim]);

    // Print first 8 output residual values
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  First 8 output values:');
    for LI := 0 to 7 do
      TVdxUtils.PrintLn('    [%d] = %.8f', [LI, LResidualOut[LI]]);

    if LNonZero > 0 then
      TVdxUtils.PrintLn(COLOR_GREEN + 'TEST 06 PASSED: KNNWalk modified the residual (%d dims changed)', [LNonZero])
    else
      TVdxUtils.PrintLn(COLOR_RED + 'TEST 06 FAILED: Residual unchanged after walk');

    // Cleanup
    LWalk.Cleanup();
    LVindex.FreeAllGpu(LCompute);
    LReader.Close();
  finally
    LWalk.Free();
    LVindex.Free();
    LCompute.Free();
    LReader.Free();
  end;
end;

// ============================================================================
//  Test 07 — LayerNorm: RMSNorm on GPU vs CPU reference
// ============================================================================

procedure Test07();
const
  CGGUFPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';
var
  LReader: TVdxGGUFReader;
  LCompute: TVdxVulkanCompute;
  LNorm: TVdxLayerNorm;
  LNormWeights: TVdxNormLayerWeights;
  LHiddenDim: UInt32;
  LResidualBuf: TVdxGpuBuffer;
  LResidualIn: array of Single;
  LResidualGpu: array of Single;
  LResidualCpu: array of Single;
  LWeightData: array of Single;
  LWeightPtr: Pointer;
  LSumSq: Double;
  LRms: Double;
  LMaxErr: Double;
  LErr: Double;
  LI: Integer;
  LPassed: Boolean;
begin
  TVdxUtils.PrintLn(COLOR_CYAN + '=== Test 07: LayerNorm — RMSNorm GPU vs CPU ===');
  TVdxUtils.PrintLn('');

  LReader := TVdxGGUFReader.Create();
  LCompute := TVdxVulkanCompute.Create();
  LNorm := TVdxLayerNorm.Create();
  try
    LReader.SetStatusCallback(StatusCallback);
    LCompute.SetStatusCallback(StatusCallback);
    LNorm.SetStatusCallback(StatusCallback);

    // Open GGUF
    TVdxUtils.FailIf(not LReader.Open(CGGUFPath),
      'Failed to open GGUF: %s', [CGGUFPath]);

    // Get hidden dim from metadata
    LHiddenDim := LReader.GetMetadataUInt32(
      'gemma3.embedding_length', 2560);
    TVdxUtils.PrintLn('Hidden dim: %d', [LHiddenDim]);

    // Init Vulkan + LayerNorm
    LCompute.Init();
    LNorm.Init(LCompute);

    // Upload layer 0 norm weights
    TVdxUtils.PrintLn('Uploading layer 0 norm weights...');
    LNorm.UploadNormWeights(LReader, 0, LNormWeights);

    // Create a test residual: ramp [1, 2, 3, ..., 2560] / 2560
    SetLength(LResidualIn, LHiddenDim);
    for LI := 0 to LHiddenDim - 1 do
      LResidualIn[LI] := (LI + 1) / LHiddenDim;

    // Upload residual to host-visible GPU buffer
    LResidualBuf := LCompute.CreateGpuBuffer(
      UInt64(LHiddenDim) * SizeOf(Single),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    LCompute.UploadToBuffer(LResidualBuf, @LResidualIn[0],
      UInt64(LHiddenDim) * SizeOf(Single));

    // Apply RMSNorm on GPU
    TVdxUtils.PrintLn('Applying RMSNorm on GPU...');
    LNorm.Apply(LResidualBuf, LNormWeights.AttnNormGpu, LHiddenDim);

    // Download GPU result
    SetLength(LResidualGpu, LHiddenDim);
    LCompute.DownloadFromBuffer(LResidualBuf, @LResidualGpu[0],
      UInt64(LHiddenDim) * SizeOf(Single));

    // Compute CPU reference
    // 1. Read attn_norm weights from GGUF
    LWeightPtr := LReader.GetTensorDataPtr('blk.0.attn_norm.weight');
    SetLength(LWeightData, LHiddenDim);
    Move(LWeightPtr^, LWeightData[0],
      UInt64(LHiddenDim) * SizeOf(Single));

    // 2. Compute RMS on CPU
    SetLength(LResidualCpu, LHiddenDim);
    LSumSq := 0.0;
    for LI := 0 to LHiddenDim - 1 do
      LSumSq := LSumSq + Double(LResidualIn[LI]) * Double(LResidualIn[LI]);
    LRms := Sqrt(LSumSq / LHiddenDim + 1e-6);

    for LI := 0 to LHiddenDim - 1 do
      LResidualCpu[LI] := Single(
        (Double(LResidualIn[LI]) / LRms) * Double(LWeightData[LI]));

    // Compare GPU vs CPU
    LMaxErr := 0.0;
    LPassed := True;
    for LI := 0 to LHiddenDim - 1 do
    begin
      LErr := Abs(Double(LResidualGpu[LI]) - Double(LResidualCpu[LI]));
      if LErr > LMaxErr then
        LMaxErr := LErr;
      if LErr > 1e-3 then
      begin
        TVdxUtils.PrintLn(COLOR_RED +
          '  MISMATCH [%d]: GPU=%.8f CPU=%.8f err=%.8f',
          [LI, LResidualGpu[LI], LResidualCpu[LI], LErr]);
        LPassed := False;
        if LI > 10 then
        begin
          TVdxUtils.PrintLn('  ... (showing first 10 mismatches)');
          Break;
        end;
      end;
    end;

    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_GREEN + '--- Results ---');
    TVdxUtils.PrintLn('  RMS (CPU):    %.8f', [LRms]);
    TVdxUtils.PrintLn('  Max error:    %.10f', [LMaxErr]);
    TVdxUtils.PrintLn('  First 4 GPU:  [%.6f, %.6f, %.6f, %.6f]',
      [LResidualGpu[0], LResidualGpu[1], LResidualGpu[2],
       LResidualGpu[3]]);
    TVdxUtils.PrintLn('  First 4 CPU:  [%.6f, %.6f, %.6f, %.6f]',
      [LResidualCpu[0], LResidualCpu[1], LResidualCpu[2],
       LResidualCpu[3]]);

    if LPassed then
      TVdxUtils.PrintLn(COLOR_GREEN +
        'TEST 07 PASSED: RMSNorm GPU matches CPU (max err=%.2e)',
        [LMaxErr])
    else
      TVdxUtils.PrintLn(COLOR_RED +
        'TEST 07 FAILED: RMSNorm GPU/CPU mismatch');

    // Cleanup
    LCompute.DestroyGpuBuffer(LResidualBuf);
    LNorm.FreeNormWeights(LNormWeights);
    LNorm.Cleanup();
    LReader.Close();
  finally
    LNorm.Free();
    LCompute.Free();
    LReader.Free();
  end;
end;

// ============================================================================
//  Test 08 — Attention MatVec F16: Q projection GPU vs CPU
// ============================================================================

procedure Test08();
const
  CGGUFPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';
  CHiddenDim = 2560;
  CQOutDim = 2048;  // 8 Q heads * 256 head_dim
  CNumQHeads = 8;
  CNumKVHeads = 4;
  CHeadDim = 256;
  CNumLayers = 34;
  CMaxSeqLen = 128;  // small for test
var
  LReader: TVdxGGUFReader;
  LCompute: TVdxVulkanCompute;
  LAttn: TVdxAttention;
  LAttnWeights: TVdxAttnLayerWeights;
  LInputBuf: TVdxGpuBuffer;
  LOutputBuf: TVdxGpuBuffer;
  LInputData: array of Single;
  LOutputGpu: array of Single;
  LOutputCpu: array of Single;
  LWeightPtr: Pointer;
  LRow: UInt32;
  LCol: UInt32;
  LAcc: Double;
  LF16Val: Single;
  LMaxErr: Double;
  LErr: Double;
  LPassed: Boolean;
  LI: Integer;

  // Convert a single F16 (UInt16) to F32
  function F16ToF32(const AVal: UInt16): Single;
  const
    CSignBit: UInt32 = $80000000;
  var
    LExp: UInt32;
    LMant: UInt32;
    LBits: UInt32;
  begin
    LExp := (UInt32(AVal) shr 10) and $1F;
    LMant := UInt32(AVal) and $3FF;

    if LExp = 0 then
    begin
      if LMant = 0 then
        // Positive or negative zero
        LBits := 0
      else
      begin
        // Denormalized: normalize by shifting mantissa left
        LExp := 1;
        while (LMant and $400) = 0 do
        begin
          LMant := LMant shl 1;
          Inc(LExp);
        end;
        LMant := LMant and $3FF;
        LBits := UInt32((113 - LExp) shl 23) or (LMant shl 13);
      end;
    end
    else if LExp = $1F then
      // Inf or NaN
      LBits := UInt32($FF shl 23) or (LMant shl 13)
    else
      // Normalized
      LBits := UInt32((LExp + 112) shl 23) or (LMant shl 13);

    // Apply sign bit separately to avoid shl 31 overflow
    if (AVal and $8000) <> 0 then
      LBits := LBits or CSignBit;

    Move(LBits, Result, 4);
  end;

begin
  TVdxUtils.PrintLn(COLOR_CYAN + '=== Test 08: Attention MatVec F16 (Q Projection) GPU vs CPU ===');
  TVdxUtils.PrintLn('');

  LReader := TVdxGGUFReader.Create();
  LCompute := TVdxVulkanCompute.Create();
  LAttn := TVdxAttention.Create();
  try
    LReader.SetStatusCallback(StatusCallback);
    LCompute.SetStatusCallback(StatusCallback);

    TVdxUtils.FailIf(not LReader.Open(CGGUFPath),
      'Failed to open GGUF: %s', [CGGUFPath]);

    LCompute.Init();
    LAttn.Init(LCompute, CHiddenDim, CNumQHeads, CNumKVHeads,
      CHeadDim, CNumLayers, CMaxSeqLen);

    TVdxUtils.PrintLn('Uploading layer 0 attention weights...');
    LAttn.UploadAttnWeights(LReader, 0, LAttnWeights);

    // Create test input: ramp [1, 2, ..., 2560] / 2560
    SetLength(LInputData, CHiddenDim);
    for LI := 0 to CHiddenDim - 1 do
      LInputData[LI] := (LI + 1) / CHiddenDim;

    LInputBuf := LCompute.CreateGpuBuffer(
      UInt64(CHiddenDim) * SizeOf(Single),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    LCompute.UploadToBuffer(LInputBuf, @LInputData[0],
      UInt64(CHiddenDim) * SizeOf(Single));

    LOutputBuf := LCompute.CreateGpuBuffer(
      UInt64(CQOutDim) * SizeOf(Single),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    TVdxUtils.PrintLn('Running GPU MatVec F16 (Q projection: %d x %d)...',
      [CHiddenDim, CQOutDim]);
    LAttn.TestMatVec(LAttnWeights.QWeightGpu, LInputBuf, LOutputBuf,
      CHiddenDim, CQOutDim);

    SetLength(LOutputGpu, CQOutDim);
    LCompute.DownloadFromBuffer(LOutputBuf, @LOutputGpu[0],
      UInt64(CQOutDim) * SizeOf(Single));

    // CPU reference: F16 weight dot F32 input
    TVdxUtils.PrintLn('Computing CPU reference...');
    LWeightPtr := LReader.GetTensorDataPtr('blk.0.attn_q.weight');
    SetLength(LOutputCpu, CQOutDim);

    for LRow := 0 to CQOutDim - 1 do
    begin
      LAcc := 0.0;
      for LCol := 0 to CHiddenDim - 1 do
      begin
        LF16Val := F16ToF32(PWord(PByte(LWeightPtr) + (UInt64(LRow) * CHiddenDim + UInt64(LCol)) * 2)^);
        LAcc := LAcc + Double(LF16Val) * Double(LInputData[LCol]);
      end;
      LOutputCpu[LRow] := Single(LAcc);
    end;

    // Compare
    LMaxErr := 0.0;
    LPassed := True;
    for LI := 0 to CQOutDim - 1 do
    begin
      LErr := Abs(Double(LOutputGpu[LI]) - Double(LOutputCpu[LI]));
      if LErr > LMaxErr then
        LMaxErr := LErr;
      if LErr > 0.1 then
      begin
        TVdxUtils.PrintLn(COLOR_RED +
          '  MISMATCH [%d]: GPU=%.6f CPU=%.6f err=%.6f',
          [LI, LOutputGpu[LI], LOutputCpu[LI], LErr]);
        LPassed := False;
        if LI > 10 then
        begin
          TVdxUtils.PrintLn('  ... (showing first mismatches)');
          Break;
        end;
      end;
    end;

    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_GREEN + '--- Results ---');
    TVdxUtils.PrintLn('  Output dim:   %d', [CQOutDim]);
    TVdxUtils.PrintLn('  Max error:    %.10f', [LMaxErr]);
    TVdxUtils.PrintLn('  First 4 GPU:  [%.6f, %.6f, %.6f, %.6f]',
      [LOutputGpu[0], LOutputGpu[1], LOutputGpu[2], LOutputGpu[3]]);
    TVdxUtils.PrintLn('  First 4 CPU:  [%.6f, %.6f, %.6f, %.6f]',
      [LOutputCpu[0], LOutputCpu[1], LOutputCpu[2], LOutputCpu[3]]);

    if LPassed then
      TVdxUtils.PrintLn(COLOR_GREEN +
        'TEST 08 PASSED: MatVec F16 GPU matches CPU (max err=%.2e)', [LMaxErr])
    else
      TVdxUtils.PrintLn(COLOR_RED + 'TEST 08 FAILED: MatVec F16 GPU/CPU mismatch');

    LCompute.DestroyGpuBuffer(LOutputBuf);
    LCompute.DestroyGpuBuffer(LInputBuf);
    LAttn.FreeAttnWeights(LAttnWeights);
    LAttn.Cleanup();
    LReader.Close();
  finally
    LAttn.Free();
    LCompute.Free();
    LReader.Free();
  end;
end;

// ============================================================================
//  TestInference — FULL INFERENCE: "The capital of France is" → ???
//  Runs all 34 layers (attention + FFN walk) for each prompt token, then
//  unembeds the final residual to predict the next token.
// ============================================================================

procedure TestInference();
const
  CGGUFPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';
  CHiddenDim = 2560;
  CFFNWidth  = 10240;
  CNumQHeads = 8;
  CNumKVHeads = 4;
  CHeadDim   = 256;
  CNumLayers = 34;
  CMaxSeqLen = 32;
  CTopK      = 128;
  CEnableAttn = True;
  CVocabSize = 262144;

  // Gemma 3 IT chat template — hardcoded token IDs from Python tokenizer
  // <bos><start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n
  // Verified via sentencepiece vocab lookup from GGUF
  CPromptIds: array[0..15] of Integer = (
    2,        // <bos>
    105,      // <start_of_turn>
    2364,     // user
    107,      // \n
    3689,     // What
    563,      // ▁is
    506,      // ▁the
    5279,     // ▁capital
    529,      // ▁of
    7001,     // ▁France
    236881,   // ?
    106,      // <end_of_turn>
    107,      // \n
    105,      // <start_of_turn>
    4368,     // model
    107       // \n
  );
var
  LReader: TVdxGGUFReader;
  LCompute: TVdxVulkanCompute;
  LNorm: TVdxLayerNorm;
  LAttn: TVdxAttention;
  LWalk: TVdxKNNWalk;
  LVindex: TVdxVindex;
  LTokenizer: TVdxTokenizer;

  // Per-layer weight storage
  LAttnWeights: array[0..CNumLayers - 1] of TVdxAttnLayerWeights;
  LNormWeights: array[0..CNumLayers - 1] of TVdxNormLayerWeights;

  // Global weights
  LOutputNormGpu: TVdxGpuBuffer;

  // Work GPU buffers (host-visible coherent, CHiddenDim floats each)
  LWorkBufA: TVdxGpuBuffer;
  LAttnOutBuf: TVdxGpuBuffer;

  // CPU-side arrays
  LResidual: array of Single;
  LSavedResidual: array of Single;
  LNormedInput: array of Single;
  LTempVec: array of Single;

  // Token IDs
  LBosTokenId: Integer;
  LPromptIds: array[0..14] of Integer;
  LAllTokenIds: TArray<Integer>;
  LTokenCount: Integer;

  // Embedding pointer (mmap'd from GGUF)
  LEmbedPtr: PByte;

  // Vocab
  LVocab: TVdxGGUFMetaValue;
  LVocabCount: Integer;

  // Unembedding results
  LTop5Ids: array[0..4] of Integer;
  LTop5Scores: array[0..4] of Double;
  LAcc: Double;
  LRowPtr: PByte;
  LTmpId: Integer;
  LTmpScore: Double;
  LTopToken: string;

  // Loop / temp vars
  LI: Integer;
  LJ: Integer;
  LLayer: Integer;
  LPos: Integer;
  LEmbedScale: Single;
  LBufSize: UInt64;
  LStartTick: UInt64;
  LElapsed: UInt64;

  // Dense FFN variables
  LUpWeights: array[0..CNumLayers - 1] of TVdxGpuBuffer;
  LGateBuf: TVdxGpuBuffer;     // F32 x FFNWidth (10240), device-local scratch
  LUpBuf: TVdxGpuBuffer;       // F32 x FFNWidth (10240), device-local scratch
  LFFNOutBuf: TVdxGpuBuffer;   // F32 x HiddenDim (2560), host-visible
  LSiluMulShader: VkShaderModule;
  LSiluMulBundle: TVdxComputePipelineBundle;
  LSiluMulDescLayout: VkDescriptorSetLayout;
  LSiluMulPush: record Count: UInt32; end;
  LSiluDescPool: VkDescriptorPool;
  LSiluDescSet: VkDescriptorSet;
  LSpvPath: string;
  LSpvData: TBytes;
  LTheta: Single;

  // Convert a single F16 (UInt16) to F32 (same as Test08)
  function F16ToF32(const AVal: UInt16): Single;
  const
    CSignBit: UInt32 = $80000000;
  var
    LExp: UInt32;
    LMant: UInt32;
    LBits: UInt32;
  begin
    LExp := (UInt32(AVal) shr 10) and $1F;
    LMant := UInt32(AVal) and $3FF;
    if LExp = 0 then
    begin
      if LMant = 0 then
        LBits := 0
      else
      begin
        LExp := 1;
        while (LMant and $400) = 0 do
        begin
          LMant := LMant shl 1;
          Inc(LExp);
        end;
        LMant := LMant and $3FF;
        LBits := UInt32((113 - LExp) shl 23) or (LMant shl 13);
      end;
    end
    else if LExp = $1F then
      LBits := UInt32($FF shl 23) or (LMant shl 13)
    else
      LBits := UInt32((LExp + 112) shl 23) or (LMant shl 13);
    if (AVal and $8000) <> 0 then
      LBits := LBits or CSignBit;
    Move(LBits, Result, 4);
  end;

  // Upload F32 weight tensor from GGUF to GPU, adding +1 offset (Gemma RMSNorm
  // stores offset-from-1 values; the actual scale is 1 + stored_weight)
  function UploadNormWeight(const ATensorName: string;
    const ACount: UInt32): TVdxGpuBuffer;
  var
    LPtr: Pointer;
    LData: array of Single;
    LK: Integer;
  begin
    LPtr := LReader.GetTensorDataPtr(ATensorName);
    SetLength(LData, ACount);
    Move(LPtr^, LData[0], ACount * SizeOf(Single));
    // GGUF stores effective weight (already includes Gemma +1 offset)
    Result := LCompute.CreateGpuBuffer(
      UInt64(ACount) * SizeOf(Single),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    LCompute.UploadToBuffer(Result, @LData[0], UInt64(ACount) * SizeOf(Single));
  end;

  // Upload raw F16 tensor from GGUF to device-local GPU via staging
  function UploadF16Tensor(const ATensorName: string): TVdxGpuBuffer;
  var
    LInfo: TVdxGGUFTensorInfo;
    LPtr: Pointer;
    LSize: UInt64;
    LStaging: TVdxGpuBuffer;
  begin
    LInfo := LReader.GetTensorInfo(ATensorName);
    LPtr := LReader.GetTensorDataPtr(ATensorName);
    LSize := UInt64(LInfo.Dimensions[0]) * UInt64(LInfo.Dimensions[1]) * 2;
    LStaging := LCompute.CreateGpuBuffer(LSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    try
      LCompute.UploadToBuffer(LStaging, LPtr, LSize);
      Result := LCompute.CreateGpuBuffer(LSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      LCompute.CopyBuffer(LStaging, Result, LSize);
    finally
      LCompute.DestroyGpuBuffer(LStaging);
    end;
  end;

  // Find token ID by linear search through GGUF vocab array
  function FindTokenId(const ATokenStr: string): Integer;
  var
    LK: Integer;
  begin
    for LK := 0 to LVocabCount - 1 do
    begin
      if LVocab.ArrayItems[LK].AsString = ATokenStr then
        Exit(LK);
    end;
    Result := -1;
  end;

begin
  TVdxUtils.PrintLn(COLOR_CYAN + '===============================================================');
  TVdxUtils.PrintLn(COLOR_CYAN + '  TestInference: FULL INFERENCE — "What is the capital of France?"');
  TVdxUtils.PrintLn(COLOR_CYAN + '===============================================================');
  TVdxUtils.PrintLn('');

  LBufSize := UInt64(CHiddenDim) * SizeOf(Single);
  LEmbedScale := Sqrt(Single(CHiddenDim));  // Gemma scales embeddings by sqrt(d_model)

  // Allocate CPU arrays
  SetLength(LResidual, CHiddenDim);
  SetLength(LSavedResidual, CHiddenDim);
  SetLength(LNormedInput, CHiddenDim);
  SetLength(LTempVec, CHiddenDim);

  // Zero all weight records
  FillChar(LAttnWeights, SizeOf(LAttnWeights), 0);
  FillChar(LNormWeights, SizeOf(LNormWeights), 0);
  FillChar(LOutputNormGpu, SizeOf(LOutputNormGpu), 0);
  FillChar(LWorkBufA, SizeOf(LWorkBufA), 0);
  FillChar(LAttnOutBuf, SizeOf(LAttnOutBuf), 0);

  LReader := TVdxGGUFReader.Create();
  LCompute := TVdxVulkanCompute.Create();
  LNorm := TVdxLayerNorm.Create();
  LAttn := TVdxAttention.Create();
  LWalk := TVdxKNNWalk.Create();
  LVindex := TVdxVindex.Create();
  LTokenizer := TVdxTokenizer.Create();
  try
    LReader.SetStatusCallback(StatusCallback);
    LCompute.SetStatusCallback(StatusCallback);
    LNorm.SetStatusCallback(StatusCallback);
    LWalk.SetStatusCallback(StatusCallback);

    // ==================================================================
    //  Phase 1: Open GGUF + Init all GPU subsystems
    // ==================================================================
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Phase 1: Init ---');
    LStartTick := TVdxUtils.GetTickCount64();

    TVdxUtils.FailIf(not LReader.Open(CGGUFPath),
      'Failed to open GGUF: %s', [CGGUFPath]);
    LCompute.Init();
    LNorm.Init(LCompute);
    LAttn.Init(LCompute, CHiddenDim, CNumQHeads, CNumKVHeads,
      CHeadDim, CNumLayers, CMaxSeqLen);
    TVdxUtils.FailIf(not LVindex.BuildFromGGUF(LReader),
      'Failed to build vindex', []);
    LWalk.Init(LCompute, LVindex.GetHiddenDim(), LVindex.GetFFNWidth(), CTopK);
    LWalk.UseUpProjection := False;  // Toggle for debugging

    // Create work buffers (host-visible coherent for easy upload/download)
    LWorkBufA := LCompute.CreateGpuBuffer(LBufSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    LAttnOutBuf := LCompute.CreateGpuBuffer(LBufSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    LElapsed := TVdxUtils.GetTickCount64() - LStartTick;
    TVdxUtils.PrintLn('  Init complete (%d ms)', [LElapsed]);

    // Dense FFN scratch buffers
    LGateBuf := LCompute.CreateGpuBuffer(
      UInt64(CFFNWidth) * SizeOf(Single),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    LUpBuf := LCompute.CreateGpuBuffer(
      UInt64(CFFNWidth) * SizeOf(Single),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    LFFNOutBuf := LCompute.CreateGpuBuffer(LBufSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // GELU-mul pipeline (GELU with tanh approximation — matches Gemma 3)
    LSpvPath := TPath.Combine(TPath.GetDirectoryName(ParamStr(0)),
      '..\shaders\gelu_mul.spv');
    LSpvPath := TPath.GetFullPath(LSpvPath);
    LSpvData := TFile.ReadAllBytes(LSpvPath);
    LSiluMulShader := LCompute.CreateShaderModule(@LSpvData[0], NativeUInt(Length(LSpvData)));
    LSiluMulDescLayout := LCompute.CreateStorageDescriptorSetLayout(2);
    LSiluMulBundle := LCompute.CreateComputePipelineWithPush(
      LSiluMulShader, 'main', LSiluMulDescLayout, SizeOf(LSiluMulPush));

    // ==================================================================
    //  Phase 2: Upload ALL weights to VRAM
    // ==================================================================
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Phase 2: Upload weights ---');
    LStartTick := TVdxUtils.GetTickCount64();

    // Gate vectors (all 34 layers, device-local via staging)
    TVdxUtils.PrintLn('  Uploading gate vectors (34 layers, ~1.7 GB)...');
    LVindex.UploadAll(LCompute);

    // Attention weights (Q/K/V/O for all 34 layers, ~1 GB)
    TVdxUtils.PrintLn('  Uploading attention weights (34 layers, ~1 GB)...');
    for LLayer := 0 to CNumLayers - 1 do
    begin
      LAttn.UploadAttnWeights(LReader, LLayer, LAttnWeights[LLayer]);
      if (LLayer mod 10 = 0) or (LLayer = CNumLayers - 1) then
        TVdxUtils.PrintLn('    Layer %d/%d done', [LLayer + 1, CNumLayers]);
    end;

    // Norm weights with Gemma +1 offset
    TVdxUtils.PrintLn('  Uploading norm weights (34 layers)...');

    // Upload FFN up weights for dense FFN
    TVdxUtils.PrintLn('  Uploading FFN up weights (34 layers)...');
    FillChar(LUpWeights, SizeOf(LUpWeights), 0);
    for LLayer := 0 to CNumLayers - 1 do
    begin
      LUpWeights[LLayer] := UploadF16Tensor(
        Format('blk.%d.ffn_up.weight', [LLayer]));
    end;

    for LLayer := 0 to CNumLayers - 1 do
    begin
      LNormWeights[LLayer].AttnNormGpu := UploadNormWeight(
        Format('blk.%d.attn_norm.weight', [LLayer]), CHiddenDim);
      LNormWeights[LLayer].PostAttnNormGpu := UploadNormWeight(
        Format('blk.%d.post_attention_norm.weight', [LLayer]), CHiddenDim);
      LNormWeights[LLayer].FFNNormGpu := UploadNormWeight(
        Format('blk.%d.ffn_norm.weight', [LLayer]), CHiddenDim);
      LNormWeights[LLayer].PostFFNNormGpu := UploadNormWeight(
        Format('blk.%d.post_ffw_norm.weight', [LLayer]), CHiddenDim);
      LNormWeights[LLayer].QNormGpu := UploadNormWeight(
        Format('blk.%d.attn_q_norm.weight', [LLayer]), CHeadDim);
      LNormWeights[LLayer].KNormGpu := UploadNormWeight(
        Format('blk.%d.attn_k_norm.weight', [LLayer]), CHeadDim);
    end;

    // Output norm (global, not per-layer)
    LOutputNormGpu := UploadNormWeight('output_norm.weight', CHiddenDim);

    LElapsed := TVdxUtils.GetTickCount64() - LStartTick;
    TVdxUtils.PrintLn('  All weights uploaded (%d ms)', [LElapsed]);

    // ==================================================================
    //  Phase 3: Tokenize the prompt using proper BPE tokenizer
    // ==================================================================
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Phase 3: Tokenize ---');

    TVdxUtils.FailIf(not LTokenizer.LoadFromGGUF(LReader),
      'Failed to load tokenizer from GGUF', []);
    TVdxUtils.PrintLn('  Vocab size: %d tokens', [LTokenizer.GetVocabSize()]);
    TVdxUtils.PrintLn('  BOS id: %d, EOS id: %d',
      [LTokenizer.GetBosId(), LTokenizer.GetEosId()]);

    // Gemma 3 IT prompt format
    LAllTokenIds := LTokenizer.Encode(
      '<start_of_turn>user'#10 +
      'What is the capital of France?<end_of_turn>'#10 +
      '<start_of_turn>model'#10,
      True  // add BOS
    );
    LTokenCount := Length(LAllTokenIds);

    TVdxUtils.PrintLn('  Token count: %d', [LTokenCount]);
    for LI := 0 to LTokenCount - 1 do
      TVdxUtils.PrintLn('    [%d] id=%-6d  "%s"',
        [LI, LAllTokenIds[LI], LTokenizer.GetTokenStr(LAllTokenIds[LI])]);

    // Read vocab for unembedding display
    LVocab := LReader.GetMetadata('tokenizer.ggml.tokens');
    LVocabCount := Length(LVocab.ArrayItems);

    // Get embedding table pointer (mmap'd F16 data)
    LEmbedPtr := PByte(LReader.GetTensorDataPtr('token_embd.weight'));
    TVdxUtils.FailIf(LEmbedPtr = nil,
      'token_embd.weight not found in GGUF', []);

    // ==================================================================
    //  Phase 4: Inference — process all prompt tokens
    // ==================================================================
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Phase 4: Inference (%d tokens x %d layers) ---',
      [LTokenCount, CNumLayers]);
    LStartTick := TVdxUtils.GetTickCount64();

    for LPos := 0 to LTokenCount - 1 do
    begin
      TVdxUtils.PrintLn('  Token %d/%d (id=%d pos=%d)',
        [LPos + 1, LTokenCount, LAllTokenIds[LPos], LPos]);

      // 4a: Embedding lookup — read F16 row from mmap, convert to F32, scale
      for LI := 0 to CHiddenDim - 1 do
        LResidual[LI] := F16ToF32(
          PWord(LEmbedPtr +
            UInt64(LAllTokenIds[LPos]) * UInt64(CHiddenDim) * 2 +
            UInt64(LI) * 2)^
        ) * LEmbedScale;

      // 4b: Process through all 34 transformer layers
      for LLayer := 0 to CNumLayers - 1 do
      begin
        // ============================================================
        //  Attention branch: x = x + PostAttnNorm(Attn(PreAttnNorm(x)))
        // ============================================================
        if CEnableAttn then
        begin

        // Save residual for the residual connection
        Move(LResidual[0], LSavedResidual[0], LBufSize);

        // Upload residual to GPU, apply PreAttnNorm in-place
        LCompute.UploadToBuffer(LWorkBufA, @LResidual[0], LBufSize);
        LNorm.Apply(LWorkBufA, LNormWeights[LLayer].AttnNormGpu, CHiddenDim);

        // Run full attention: normed input -> attention output
        // Full layers [5,11,17,23,29] use theta=1M, sliding layers use theta=10K
        if LLayer mod 6 = 5 then
          LTheta := 1000000.0
        else
          LTheta := 10000.0;
        LAttn.Forward(
          LWorkBufA,
          LAttnWeights[LLayer],
          LNormWeights[LLayer].QNormGpu,
          LNormWeights[LLayer].KNormGpu,
          LLayer,
          LPos,
          LTheta,
          LAttnOutBuf
        );

        // Apply PostAttnNorm on attention output in-place
        LNorm.Apply(LAttnOutBuf,
          LNormWeights[LLayer].PostAttnNormGpu, CHiddenDim);

        // Download post-normed attention output, add to saved residual
        LCompute.DownloadFromBuffer(LAttnOutBuf, @LTempVec[0], LBufSize);
        for LI := 0 to CHiddenDim - 1 do
          LResidual[LI] := LSavedResidual[LI] + LTempVec[LI];

        // Diagnostic: attention contribution norm (last token only)
        if LPos = LTokenCount - 1 then
        begin
          LAcc := 0.0;
          for LI := 0 to CHiddenDim - 1 do
            LAcc := LAcc + Double(LTempVec[LI]) * Double(LTempVec[LI]);
          TVdxUtils.PrintLn('    L%02d  attn_contrib=%.1f', [LLayer, Sqrt(LAcc)]);
        end;

        end; // if CEnableAttn

        // ============================================================
        //  FFN branch: x = x + PostFFNNorm(down(GELU(gate(x)) * up(x)))
        //  Dense standard transformer FFN using matvec_f16 shader
        // ============================================================

        // Save residual for the residual connection
        Move(LResidual[0], LSavedResidual[0], LBufSize);

        // Upload residual to GPU, apply PreFFNNorm in-place
        LCompute.UploadToBuffer(LWorkBufA, @LResidual[0], LBufSize);
        LNorm.Apply(LWorkBufA, LNormWeights[LLayer].FFNNormGpu, CHiddenDim);

        // gate(x) → LGateBuf [10240]
        LAttn.TestMatVec(LVindex.GetLayer(LLayer).GateGpuBuffer,
          LWorkBufA, LGateBuf, CHiddenDim, CFFNWidth);

        // up(x) → LUpBuf [10240]
        LAttn.TestMatVec(LUpWeights[LLayer],
          LWorkBufA, LUpBuf, CHiddenDim, CFFNWidth);

        // GELU(gate) * up → LGateBuf [10240] in-place
        LSiluMulPush.Count := CFFNWidth;
        LSiluDescPool := LCompute.CreateDescriptorPoolForStorage(1, 2);
        try
          LSiluDescSet := LCompute.AllocateDescriptorSetForBuffers(
            LSiluDescPool, LSiluMulDescLayout, [LGateBuf, LUpBuf]);
          LCompute.DispatchComputeWithPush(
            LSiluMulBundle.Pipeline, LSiluMulBundle.PipelineLayout,
            LSiluDescSet, @LSiluMulPush, SizeOf(LSiluMulPush),
            (CFFNWidth + 255) div 256);
        finally
          LCompute.DestroyDescriptorPoolHandle(LSiluDescPool);
        end;

        // down(hidden) → LFFNOutBuf [2560]
        LAttn.TestMatVec(LVindex.GetLayer(LLayer).DownGpuBuffer,
          LGateBuf, LFFNOutBuf, CFFNWidth, CHiddenDim);

        // Apply PostFFNNorm to FFN output
        LNorm.Apply(LFFNOutBuf,
          LNormWeights[LLayer].PostFFNNormGpu, CHiddenDim);

        // Download post-normed FFN output, add to saved residual
        LCompute.DownloadFromBuffer(LFFNOutBuf, @LTempVec[0], LBufSize);
        for LI := 0 to CHiddenDim - 1 do
          LResidual[LI] := LSavedResidual[LI] + LTempVec[LI];

        // Diagnostic: FFN contribution + residual norms (last token only)
        if LPos = LTokenCount - 1 then
        begin
          LAcc := 0.0;
          for LI := 0 to CHiddenDim - 1 do
            LAcc := LAcc + Double(LTempVec[LI]) * Double(LTempVec[LI]);
          LTmpScore := Sqrt(LAcc);
          LAcc := 0.0;
          for LI := 0 to CHiddenDim - 1 do
            LAcc := LAcc + Double(LResidual[LI]) * Double(LResidual[LI]);
          TVdxUtils.PrintLn('         ffn_contrib=%.1f  residual=%.1f',
            [LTmpScore, Sqrt(LAcc)]);
        end;

      end; // end layer loop

      LElapsed := TVdxUtils.GetTickCount64() - LStartTick;
      TVdxUtils.PrintLn('    Elapsed so far: %d ms', [LElapsed]);

    end; // end token loop

    LElapsed := TVdxUtils.GetTickCount64() - LStartTick;
    TVdxUtils.PrintLn('  Inference complete (%d ms)', [LElapsed]);

    // ==================================================================
    //  Phase 5: Final norm + Unembedding (CPU)
    // ==================================================================
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Phase 5: Final norm + Unembedding ---');
    LStartTick := TVdxUtils.GetTickCount64();

    // Apply final RMSNorm (output_norm) to the last residual
    LCompute.UploadToBuffer(LWorkBufA, @LResidual[0], LBufSize);
    LNorm.Apply(LWorkBufA, LOutputNormGpu, CHiddenDim);
    LCompute.DownloadFromBuffer(LWorkBufA, @LResidual[0], LBufSize);

    // Unembedding: dot product of final residual with each vocab embedding
    // (tied weights — same token_embd.weight used for output projection)
    TVdxUtils.PrintLn('  Computing logits (%d dot products x %d dims)...',
      [CVocabSize, CHiddenDim]);

    // Initialize top-5 tracking
    for LI := 0 to 4 do
    begin
      LTop5Ids[LI] := 0;
      LTop5Scores[LI] := -1e30;
    end;

    for LI := 0 to CVocabSize - 1 do
    begin
      // Dot product: residual[2560] . embed_row[2560] (F16->F32)
      LAcc := 0.0;
      LRowPtr := LEmbedPtr + UInt64(LI) * UInt64(CHiddenDim) * 2;
      for LJ := 0 to CHiddenDim - 1 do
        LAcc := LAcc +
          Double(LResidual[LJ]) *
          Double(F16ToF32(PWord(LRowPtr + UInt64(LJ) * 2)^));

      // Maintain top-5 by checking against the lowest score in our list
      if LAcc > LTop5Scores[4] then
      begin
        LTop5Ids[4] := LI;
        LTop5Scores[4] := LAcc;
        // Bubble up to keep sorted (descending)
        for LJ := 3 downto 0 do
        begin
          if LTop5Scores[LJ + 1] > LTop5Scores[LJ] then
          begin
            LTmpId := LTop5Ids[LJ];
            LTmpScore := LTop5Scores[LJ];
            LTop5Ids[LJ] := LTop5Ids[LJ + 1];
            LTop5Scores[LJ] := LTop5Scores[LJ + 1];
            LTop5Ids[LJ + 1] := LTmpId;
            LTop5Scores[LJ + 1] := LTmpScore;
          end;
        end;
      end;

      // Progress indicator
      if (LI > 0) and (LI mod 50000 = 0) then
        TVdxUtils.PrintLn('    %d / %d ...', [LI, CVocabSize]);
    end;

    LElapsed := TVdxUtils.GetTickCount64() - LStartTick;
    TVdxUtils.PrintLn('  Unembedding complete (%d ms)', [LElapsed]);

    // ==================================================================
    //  Phase 6: Print results
    // ==================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_GREEN + '========================================================');
    TVdxUtils.PrintLn(COLOR_GREEN + '  INFERENCE RESULTS');
    TVdxUtils.PrintLn(COLOR_GREEN + '========================================================');
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  Prompt: "What is the capital of France?" (Gemma 3 IT chat format)');
    TVdxUtils.PrintLn('  Embedding scale: sqrt(%d) = %.3f', [CHiddenDim, LEmbedScale]);
    TVdxUtils.PrintLn('  FFN top-K: %d / %d', [CTopK, CFFNWidth]);
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  Top 5 predictions:');
    for LI := 0 to 4 do
    begin
      if (LTop5Ids[LI] >= 0) and (LTop5Ids[LI] < LVocabCount) then
        LTopToken := LVocab.ArrayItems[LTop5Ids[LI]].AsString
      else
        LTopToken := '<unknown>';
      TVdxUtils.PrintLn('    #%d  id=%-6d  score=%12.4f  token="%s"',
        [LI + 1, LTop5Ids[LI], LTop5Scores[LI], LTopToken]);
    end;

    TVdxUtils.PrintLn('');
    if (LTop5Ids[0] >= 0) and (LTop5Ids[0] < LVocabCount) then
      LTopToken := LVocab.ArrayItems[LTop5Ids[0]].AsString
    else
      LTopToken := '<unknown>';
    TVdxUtils.PrintLn(COLOR_CYAN + '  >>> Answer: "%s"', [LTopToken]);
    TVdxUtils.PrintLn('');

    // Print residual diagnostics
    LAcc := 0.0;
    for LI := 0 to CHiddenDim - 1 do
      LAcc := LAcc + Double(LResidual[LI]) * Double(LResidual[LI]);
    TVdxUtils.PrintLn('  Final residual L2 norm: %.4f', [Sqrt(LAcc)]);
    TVdxUtils.PrintLn('  First 4 residual values: [%.6f, %.6f, %.6f, %.6f]',
      [LResidual[0], LResidual[1], LResidual[2], LResidual[3]]);

    // ==================================================================
    //  Phase 7: Cleanup
    // ==================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Cleanup ---');

    LCompute.DestroyGpuBuffer(LWorkBufA);
    LCompute.DestroyGpuBuffer(LAttnOutBuf);
    LCompute.DestroyGpuBuffer(LOutputNormGpu);
    LCompute.DestroyGpuBuffer(LGateBuf);
    LCompute.DestroyGpuBuffer(LUpBuf);
    LCompute.DestroyGpuBuffer(LFFNOutBuf);
    LCompute.DestroyComputePipelineBundle(LSiluMulBundle);
    LCompute.DestroyDescriptorSetLayoutHandle(LSiluMulDescLayout);
    LCompute.DestroyShaderModuleHandle(LSiluMulShader);

    for LLayer := 0 to CNumLayers - 1 do
    begin
      LAttn.FreeAttnWeights(LAttnWeights[LLayer]);
      if LUpWeights[LLayer].Buffer <> VK_NULL_HANDLE then
        LCompute.DestroyGpuBuffer(LUpWeights[LLayer]);
      // Free norm weight buffers (uploaded manually with +1 offset)
      LCompute.DestroyGpuBuffer(LNormWeights[LLayer].AttnNormGpu);
      LCompute.DestroyGpuBuffer(LNormWeights[LLayer].PostAttnNormGpu);
      LCompute.DestroyGpuBuffer(LNormWeights[LLayer].FFNNormGpu);
      LCompute.DestroyGpuBuffer(LNormWeights[LLayer].PostFFNNormGpu);
      LCompute.DestroyGpuBuffer(LNormWeights[LLayer].QNormGpu);
      LCompute.DestroyGpuBuffer(LNormWeights[LLayer].KNormGpu);
    end;

    LWalk.Cleanup();
    LVindex.FreeAllGpu(LCompute);
    LAttn.Cleanup();
    LNorm.Cleanup();
    LReader.Close();
    TVdxUtils.PrintLn('  Done.');
  finally
    LTokenizer.Free();
    LVindex.Free();
    LWalk.Free();
    LAttn.Free();
    LNorm.Free();
    LCompute.Free();
    LReader.Free();
  end;
end;

// ============================================================================
//  Test10_QKNorm — Verify QK-norm GPU vs CPU reference
//  Checks: are QK-norm weights correct? Does GPU match CPU?
//  Prints raw weight values, pre/post norm vectors, score magnitude estimate.
// ============================================================================

// ============================================================================
//  Test_AttnTrace — Trace attention on real BOS token at position 0
//  At pos=0, seq_len=1: softmax([single_score]) = [1.0], so output = O × V.
//  This bypasses Q/K/scores entirely and tests V+O path + GQA layout.
// ============================================================================

// ============================================================================
//  Test_FFNCompare — Standard FFN vs our approximation on CPU
//  Standard: down(SiLU(gate(x)) * up(x))
//  Ours:     down^T @ gate(x)  (missing up + SiLU)
//  Runs on layer 0, BOS token embedding after PreFFNNorm.
// ============================================================================

procedure Test_FFNCompare();
const
  CGGUFPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';
  CHiddenDim = 2560;
  CFFNWidth = 10240;
var
  LReader: TVdxGGUFReader;
  LCompute: TVdxVulkanCompute;
  LNorm: TVdxLayerNorm;
  LNormBuf: TVdxGpuBuffer;
  LResBuf: TVdxGpuBuffer;
  LResidual: array of Single;
  LNormedInput: array of Single;
  LGateOut: array of Single;
  LUpOut: array of Single;
  LHidden: array of Single;
  LStdFFNOut: array of Single;
  LApproxFFNOut: array of Single;
  LGatePtr: PByte;
  LUpPtr: PByte;
  LDownPtr: PByte;
  LEmbedPtr: PByte;
  LBosTokenId: Integer;
  LEmbedScale: Single;
  LBufSize: UInt64;
  LI: Integer;
  LJ: Integer;
  LAcc: Double;
  LCosNum: Double;
  LCosDenA: Double;
  LCosDenB: Double;
  LL2Std: Double;
  LL2Approx: Double;
  LL2Diff: Double;
  LWeightPtr: Pointer;
  LNormWeightData: array of Single;
  LSumSq: Double;
  LRms: Double;
  LSilu: Double;

  function F16ToF32(const AVal: UInt16): Single;
  const
    CSignBit: UInt32 = $80000000;
  var
    LExp: UInt32;
    LMant: UInt32;
    LBits: UInt32;
  begin
    LExp := (UInt32(AVal) shr 10) and $1F;
    LMant := UInt32(AVal) and $3FF;
    if LExp = 0 then
    begin
      if LMant = 0 then
        LBits := 0
      else
      begin
        LExp := 1;
        while (LMant and $400) = 0 do
        begin
          LMant := LMant shl 1;
          Inc(LExp);
        end;
        LMant := LMant and $3FF;
        LBits := UInt32((113 - LExp) shl 23) or (LMant shl 13);
      end;
    end
    else if LExp = $1F then
      LBits := UInt32($FF shl 23) or (LMant shl 13)
    else
      LBits := UInt32((LExp + 112) shl 23) or (LMant shl 13);
    if (AVal and $8000) <> 0 then
      LBits := LBits or CSignBit;
    Move(LBits, Result, 4);
  end;

begin
  TVdxUtils.PrintLn(COLOR_CYAN + '=== Test_FFNCompare: Standard FFN vs Approximate ===');
  TVdxUtils.PrintLn(COLOR_CYAN + '  Layer 0, BOS token');
  TVdxUtils.PrintLn('');

  LBufSize := UInt64(CHiddenDim) * SizeOf(Single);
  LEmbedScale := Sqrt(Single(CHiddenDim));

  LReader := TVdxGGUFReader.Create();
  LCompute := TVdxVulkanCompute.Create();
  LNorm := TVdxLayerNorm.Create();
  try
    LReader.SetStatusCallback(StatusCallback);
    LCompute.SetStatusCallback(StatusCallback);
    LNorm.SetStatusCallback(StatusCallback);

    TVdxUtils.FailIf(not LReader.Open(CGGUFPath),
      'Failed to open GGUF: %s', [CGGUFPath]);
    LCompute.Init();
    LNorm.Init(LCompute);

    // ================================================================
    //  Embed BOS + PreFFNNorm
    // ================================================================
    LBosTokenId := Integer(LReader.GetMetadataUInt32(
      'tokenizer.ggml.bos_token_id', 2));
    LEmbedPtr := PByte(LReader.GetTensorDataPtr('token_embd.weight'));

    SetLength(LResidual, CHiddenDim);
    for LI := 0 to CHiddenDim - 1 do
      LResidual[LI] := F16ToF32(PWord(LEmbedPtr +
        UInt64(LBosTokenId) * UInt64(CHiddenDim) * 2 +
        UInt64(LI) * 2)^) * LEmbedScale;

    // Apply PreFFNNorm on CPU (with +1 offset)
    LWeightPtr := LReader.GetTensorDataPtr('blk.0.ffn_norm.weight');
    SetLength(LNormWeightData, CHiddenDim);
    Move(LWeightPtr^, LNormWeightData[0], LBufSize);

    LSumSq := 0.0;
    for LI := 0 to CHiddenDim - 1 do
      LSumSq := LSumSq + Double(LResidual[LI]) * Double(LResidual[LI]);
    LRms := Sqrt(LSumSq / CHiddenDim + 1e-6);

    SetLength(LNormedInput, CHiddenDim);
    for LI := 0 to CHiddenDim - 1 do
      LNormedInput[LI] := Single(
        (Double(LResidual[LI]) / LRms) * (1.0 + Double(LNormWeightData[LI])));

    TVdxUtils.PrintLn('  Normed input L2: %.4f', [LRms]);

    // ================================================================
    //  Gate projection: gate_out = gate_weight @ normed_input
    // ================================================================
    TVdxUtils.PrintLn('  Computing gate projection (10240 dot products)...');
    LGatePtr := PByte(LReader.GetTensorDataPtr('blk.0.ffn_gate.weight'));
    SetLength(LGateOut, CFFNWidth);

    for LI := 0 to CFFNWidth - 1 do
    begin
      LAcc := 0.0;
      for LJ := 0 to CHiddenDim - 1 do
        LAcc := LAcc + Double(F16ToF32(
          PWord(LGatePtr + (UInt64(LI) * CHiddenDim + UInt64(LJ)) * 2)^)) *
          Double(LNormedInput[LJ]);
      LGateOut[LI] := Single(LAcc);
    end;

    // ================================================================
    //  Up projection: up_out = up_weight @ normed_input
    // ================================================================
    TVdxUtils.PrintLn('  Computing up projection (10240 dot products)...');
    LUpPtr := PByte(LReader.GetTensorDataPtr('blk.0.ffn_up.weight'));
    SetLength(LUpOut, CFFNWidth);

    for LI := 0 to CFFNWidth - 1 do
    begin
      LAcc := 0.0;
      for LJ := 0 to CHiddenDim - 1 do
        LAcc := LAcc + Double(F16ToF32(
          PWord(LUpPtr + (UInt64(LI) * CHiddenDim + UInt64(LJ)) * 2)^)) *
          Double(LNormedInput[LJ]);
      LUpOut[LI] := Single(LAcc);
    end;

    // ================================================================
    //  Hidden = SiLU(gate_out) * up_out
    // ================================================================
    TVdxUtils.PrintLn('  Computing SiLU(gate) * up...');
    SetLength(LHidden, CFFNWidth);
    for LI := 0 to CFFNWidth - 1 do
    begin
      // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
      LSilu := Double(LGateOut[LI]) / (1.0 + Exp(-Double(LGateOut[LI])));
      LHidden[LI] := Single(LSilu * Double(LUpOut[LI]));
    end;

    // ================================================================
    //  Standard FFN: down_weight @ hidden
    // ================================================================
    TVdxUtils.PrintLn('  Computing standard down projection (2560 dot products of 10240)...');
    LDownPtr := PByte(LReader.GetTensorDataPtr('blk.0.ffn_down.weight'));
    SetLength(LStdFFNOut, CHiddenDim);

    for LI := 0 to CHiddenDim - 1 do
    begin
      LAcc := 0.0;
      for LJ := 0 to CFFNWidth - 1 do
        LAcc := LAcc + Double(F16ToF32(
          PWord(LDownPtr + (UInt64(LI) * CFFNWidth + UInt64(LJ)) * 2)^)) *
          Double(LHidden[LJ]);
      LStdFFNOut[LI] := Single(LAcc);
    end;

    // ================================================================
    //  Approximate FFN: down^T @ gate_out (what VindexLLM does at K=all)
    // ================================================================
    TVdxUtils.PrintLn('  Computing approximate FFN (down^T @ gate_out)...');
    SetLength(LApproxFFNOut, CHiddenDim);

    for LI := 0 to CHiddenDim - 1 do
    begin
      LAcc := 0.0;
      for LJ := 0 to CFFNWidth - 1 do
        LAcc := LAcc + Double(F16ToF32(
          PWord(LDownPtr + (UInt64(LI) * CFFNWidth + UInt64(LJ)) * 2)^)) *
          Double(LGateOut[LJ]);
      LApproxFFNOut[LI] := Single(LAcc);
    end;

    // ================================================================
    //  Compare
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Results ---');

    // L2 norms
    LL2Std := 0.0;
    LL2Approx := 0.0;
    LL2Diff := 0.0;
    LCosNum := 0.0;
    LCosDenA := 0.0;
    LCosDenB := 0.0;
    for LI := 0 to CHiddenDim - 1 do
    begin
      LL2Std := LL2Std + Double(LStdFFNOut[LI]) * Double(LStdFFNOut[LI]);
      LL2Approx := LL2Approx + Double(LApproxFFNOut[LI]) * Double(LApproxFFNOut[LI]);
      LL2Diff := LL2Diff + Sqr(Double(LStdFFNOut[LI]) - Double(LApproxFFNOut[LI]));
      LCosNum := LCosNum + Double(LStdFFNOut[LI]) * Double(LApproxFFNOut[LI]);
      LCosDenA := LCosDenA + Double(LStdFFNOut[LI]) * Double(LStdFFNOut[LI]);
      LCosDenB := LCosDenB + Double(LApproxFFNOut[LI]) * Double(LApproxFFNOut[LI]);
    end;

    TVdxUtils.PrintLn('  Standard FFN L2:     %.4f', [Sqrt(LL2Std)]);
    TVdxUtils.PrintLn('  Approximate FFN L2:  %.4f', [Sqrt(LL2Approx)]);
    TVdxUtils.PrintLn('  Difference L2:       %.4f', [Sqrt(LL2Diff)]);
    TVdxUtils.PrintLn('  Cosine similarity:   %.8f', [LCosNum / (Sqrt(LCosDenA) * Sqrt(LCosDenB))]);
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  Standard first 4:    [%.6f, %.6f, %.6f, %.6f]',
      [LStdFFNOut[0], LStdFFNOut[1], LStdFFNOut[2], LStdFFNOut[3]]);
    TVdxUtils.PrintLn('  Approx first 4:      [%.6f, %.6f, %.6f, %.6f]',
      [LApproxFFNOut[0], LApproxFFNOut[1], LApproxFFNOut[2], LApproxFFNOut[3]]);

    TVdxUtils.PrintLn('');
    if LCosNum / (Sqrt(LCosDenA) * Sqrt(LCosDenB)) > 0.9 then
      TVdxUtils.PrintLn(COLOR_GREEN +
        'Approximation is directionally similar — up projection helps but may not be the only issue')
    else
      TVdxUtils.PrintLn(COLOR_RED +
        'Approximation is VERY different — up projection is CRITICAL, must be added');

    LNorm.Cleanup();
    LReader.Close();
  finally
    LNorm.Free();
    LCompute.Free();
    LReader.Free();
  end;
end;

// ============================================================================
//  Test_Basics — Verify fundamental assumptions: tensor dims, token IDs,
//  embedding values, and down vector layout. No GPU needed.
// ============================================================================

procedure Test_Basics();
const
  CGGUFPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';
var
  LReader: TVdxGGUFReader;
  LGateInfo: TVdxGGUFTensorInfo;
  LUpInfo: TVdxGGUFTensorInfo;
  LDownInfo: TVdxGGUFTensorInfo;
  LQInfo: TVdxGGUFTensorInfo;
  LOInfo: TVdxGGUFTensorInfo;
  LVocab: TVdxGGUFMetaValue;
  LVocabCount: Integer;
  LBosId: Integer;
  LI: Integer;
  LToken: string;
begin
  TVdxUtils.PrintLn(COLOR_CYAN + '=== Test_Basics: Verify Fundamental Assumptions ===');
  TVdxUtils.PrintLn('');

  LReader := TVdxGGUFReader.Create();
  try
    LReader.SetStatusCallback(StatusCallback);
    TVdxUtils.FailIf(not LReader.Open(CGGUFPath),
      'Failed to open GGUF', []);

    // ================================================================
    //  Tensor dimensions — the critical question
    // ================================================================
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Tensor Dimensions (ne0 x ne1) ---');
    TVdxUtils.PrintLn('  ne0 = contiguous dim, ne1 = strided dim');
    TVdxUtils.PrintLn('');

    LGateInfo := LReader.GetTensorInfo('blk.0.ffn_gate.weight');
    TVdxUtils.PrintLn('  ffn_gate:   [%d x %d]  %s',
      [LGateInfo.Dimensions[0], LGateInfo.Dimensions[1],
       VdxGGMLTypeName(LGateInfo.TensorType)]);

    LUpInfo := LReader.GetTensorInfo('blk.0.ffn_up.weight');
    TVdxUtils.PrintLn('  ffn_up:     [%d x %d]  %s',
      [LUpInfo.Dimensions[0], LUpInfo.Dimensions[1],
       VdxGGMLTypeName(LUpInfo.TensorType)]);

    LDownInfo := LReader.GetTensorInfo('blk.0.ffn_down.weight');
    TVdxUtils.PrintLn('  ffn_down:   [%d x %d]  %s',
      [LDownInfo.Dimensions[0], LDownInfo.Dimensions[1],
       VdxGGMLTypeName(LDownInfo.TensorType)]);

    LQInfo := LReader.GetTensorInfo('blk.0.attn_q.weight');
    TVdxUtils.PrintLn('  attn_q:     [%d x %d]  %s',
      [LQInfo.Dimensions[0], LQInfo.Dimensions[1],
       VdxGGMLTypeName(LQInfo.TensorType)]);

    LOInfo := LReader.GetTensorInfo('blk.0.attn_output.weight');
    TVdxUtils.PrintLn('  attn_output:[%d x %d]  %s',
      [LOInfo.Dimensions[0], LOInfo.Dimensions[1],
       VdxGGMLTypeName(LOInfo.TensorType)]);

    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  If ffn_down is [10240 x 2560]: output-dim major (need strided gather)');
    TVdxUtils.PrintLn('  If ffn_down is [2560 x 10240]: feature-major (contiguous copy OK)');

    // ================================================================
    //  Tokenization check
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Tokenization ---');

    LVocab := LReader.GetMetadata('tokenizer.ggml.tokens');
    LVocabCount := Length(LVocab.ArrayItems);
    LBosId := Integer(LReader.GetMetadataUInt32('tokenizer.ggml.bos_token_id', 2));

    TVdxUtils.PrintLn('  Vocab size: %d', [LVocabCount]);
    TVdxUtils.PrintLn('  BOS id: %d  token: "%s"',
      [LBosId, LVocab.ArrayItems[LBosId].AsString]);

    // Look up our prompt tokens
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  Searching for prompt tokens:');

    for LI := 0 to LVocabCount - 1 do
    begin
      LToken := LVocab.ArrayItems[LI].AsString;
      if (LToken = #$2581'The') or
         (LToken = #$2581'capital') or
         (LToken = #$2581'of') or
         (LToken = #$2581'France') or
         (LToken = #$2581'is') or
         (LToken = #$2581'Paris') or
         (LToken = 'Paris') or
         (LToken = #$2581'paris') then
        TVdxUtils.PrintLn('    id=%-6d  "%s"', [LI, LToken]);
    end;

    // Also print some known token IDs to verify
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  Token ID spot checks:');
    TVdxUtils.PrintLn('    id=0: "%s"', [LVocab.ArrayItems[0].AsString]);
    TVdxUtils.PrintLn('    id=1: "%s"', [LVocab.ArrayItems[1].AsString]);
    TVdxUtils.PrintLn('    id=2: "%s"', [LVocab.ArrayItems[2].AsString]);
    TVdxUtils.PrintLn('    id=3: "%s"', [LVocab.ArrayItems[3].AsString]);
    TVdxUtils.PrintLn('    id=106: "%s"', [LVocab.ArrayItems[106].AsString]);
    TVdxUtils.PrintLn('    id=107: "%s"', [LVocab.ArrayItems[107].AsString]);
    TVdxUtils.PrintLn('    id=108: "%s"', [LVocab.ArrayItems[108].AsString]);

    // Verify exact bytes for key tokens
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  Byte-level check for "user" token:');
    for LI := 0 to LVocabCount - 1 do
    begin
      LToken := LVocab.ArrayItems[LI].AsString;
      if LToken = 'user' then
        TVdxUtils.PrintLn('    FOUND "user" at id=%d (len=%d)', [LI, Length(LToken)]);
      if LToken = 'model' then
        TVdxUtils.PrintLn('    FOUND "model" at id=%d (len=%d)', [LI, Length(LToken)]);
      if LToken = 'What' then
        TVdxUtils.PrintLn('    FOUND "What" at id=%d (len=%d)', [LI, Length(LToken)]);
      if (Length(LToken) = 1) and (Ord(LToken[1]) = 10) then
        TVdxUtils.PrintLn('    FOUND newline (chr 10) at id=%d', [LI]);
    end;

    LReader.Close();
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_GREEN + 'Test_Basics complete.');
  finally
    LReader.Free();
  end;
end;

procedure Test_AttnTrace();
const
  CGGUFPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';
  CHiddenDim = 2560;
  CNumQHeads = 8;
  CNumKVHeads = 4;
  CHeadDim = 256;
  CNumLayers = 34;
  CMaxSeqLen = 32;
  CQOutDim = 2048;
  CKVOutDim = 1024;
var
  LReader: TVdxGGUFReader;
  LCompute: TVdxVulkanCompute;
  LNorm: TVdxLayerNorm;
  LAttn: TVdxAttention;
  LAttnWeights: TVdxAttnLayerWeights;
  LNormWeights: TVdxNormLayerWeights;
  LResidualBuf: TVdxGpuBuffer;
  LAttnOutBuf: TVdxGpuBuffer;
  LVOutBuf: TVdxGpuBuffer;
  LResidual: array of Single;
  LAttnOut: array of Single;
  LVProj: array of Single;
  LExpectedAttnIn: array of Single;
  LExpectedOut: array of Single;
  LEmbedPtr: PByte;
  LWeightPtr: Pointer;
  LBosTokenId: Integer;
  LEmbedScale: Single;
  LBufSize: UInt64;
  LI: Integer;
  LJ: Integer;
  LHead: Integer;
  LKVHead: Integer;
  LAcc: Double;
  LMaxErr: Double;
  LErr: Double;
  LL2Attn: Double;
  LL2Resid: Double;

  function F16ToF32(const AVal: UInt16): Single;
  const
    CSignBit: UInt32 = $80000000;
  var
    LExp: UInt32;
    LMant: UInt32;
    LBits: UInt32;
  begin
    LExp := (UInt32(AVal) shr 10) and $1F;
    LMant := UInt32(AVal) and $3FF;
    if LExp = 0 then
    begin
      if LMant = 0 then
        LBits := 0
      else
      begin
        LExp := 1;
        while (LMant and $400) = 0 do
        begin
          LMant := LMant shl 1;
          Inc(LExp);
        end;
        LMant := LMant and $3FF;
        LBits := UInt32((113 - LExp) shl 23) or (LMant shl 13);
      end;
    end
    else if LExp = $1F then
      LBits := UInt32($FF shl 23) or (LMant shl 13)
    else
      LBits := UInt32((LExp + 112) shl 23) or (LMant shl 13);
    if (AVal and $8000) <> 0 then
      LBits := LBits or CSignBit;
    Move(LBits, Result, 4);
  end;

  function UploadNormWeight(const ATensorName: string;
    const ACount: UInt32): TVdxGpuBuffer;
  var
    LPtr: Pointer;
    LData: array of Single;
    LK: Integer;
  begin
    LPtr := LReader.GetTensorDataPtr(ATensorName);
    SetLength(LData, ACount);
    Move(LPtr^, LData[0], ACount * SizeOf(Single));
    for LK := 0 to ACount - 1 do
      LData[LK] := LData[LK] + 1.0;
    Result := LCompute.CreateGpuBuffer(
      UInt64(ACount) * SizeOf(Single),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    LCompute.UploadToBuffer(Result, @LData[0], UInt64(ACount) * SizeOf(Single));
  end;

begin
  TVdxUtils.PrintLn(COLOR_CYAN + '=== Test_AttnTrace: Layer 0 Attention on BOS Token ===');
  TVdxUtils.PrintLn(COLOR_CYAN + '  Position 0, seq_len=1 => output must equal O x V');
  TVdxUtils.PrintLn('');

  LBufSize := UInt64(CHiddenDim) * SizeOf(Single);
  LEmbedScale := Sqrt(Single(CHiddenDim));

  LReader := TVdxGGUFReader.Create();
  LCompute := TVdxVulkanCompute.Create();
  LNorm := TVdxLayerNorm.Create();
  LAttn := TVdxAttention.Create();
  try
    LReader.SetStatusCallback(StatusCallback);
    LCompute.SetStatusCallback(StatusCallback);
    LNorm.SetStatusCallback(StatusCallback);

    TVdxUtils.FailIf(not LReader.Open(CGGUFPath),
      'Failed to open GGUF: %s', [CGGUFPath]);
    LCompute.Init();
    LNorm.Init(LCompute);
    LAttn.Init(LCompute, CHiddenDim, CNumQHeads, CNumKVHeads,
      CHeadDim, CNumLayers, CMaxSeqLen);

    // Upload layer 0 weights
    TVdxUtils.PrintLn('Uploading layer 0 weights...');
    LAttn.UploadAttnWeights(LReader, 0, LAttnWeights);
    LNormWeights.AttnNormGpu := UploadNormWeight(
      'blk.0.attn_norm.weight', CHiddenDim);
    LNormWeights.QNormGpu := UploadNormWeight(
      'blk.0.attn_q_norm.weight', CHeadDim);
    LNormWeights.KNormGpu := UploadNormWeight(
      'blk.0.attn_k_norm.weight', CHeadDim);

    // ================================================================
    //  Step 1: Embed BOS token
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Step 1: Embed BOS Token ---');

    LBosTokenId := Integer(LReader.GetMetadataUInt32(
      'tokenizer.ggml.bos_token_id', 2));
    LEmbedPtr := PByte(LReader.GetTensorDataPtr('token_embd.weight'));
    TVdxUtils.PrintLn('  BOS token ID: %d', [LBosTokenId]);

    SetLength(LResidual, CHiddenDim);
    for LI := 0 to CHiddenDim - 1 do
      LResidual[LI] := F16ToF32(
        PWord(LEmbedPtr +
          UInt64(LBosTokenId) * UInt64(CHiddenDim) * 2 +
          UInt64(LI) * 2)^
      ) * LEmbedScale;

    LL2Resid := 0.0;
    for LI := 0 to CHiddenDim - 1 do
      LL2Resid := LL2Resid + Double(LResidual[LI]) * Double(LResidual[LI]);
    LL2Resid := Sqrt(LL2Resid);
    TVdxUtils.PrintLn('  Residual L2: %.4f', [LL2Resid]);
    TVdxUtils.PrintLn('  First 4: [%.6f, %.6f, %.6f, %.6f]',
      [LResidual[0], LResidual[1], LResidual[2], LResidual[3]]);

    // ================================================================
    //  Step 2: PreAttnNorm + Attention Forward at pos=0
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Step 2: PreAttnNorm + Attention Forward ---');

    LResidualBuf := LCompute.CreateGpuBuffer(LBufSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    LAttnOutBuf := LCompute.CreateGpuBuffer(LBufSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    LCompute.UploadToBuffer(LResidualBuf, @LResidual[0], LBufSize);
    LNorm.Apply(LResidualBuf, LNormWeights.AttnNormGpu, CHiddenDim);

    // Dump normed input
    LCompute.DownloadFromBuffer(LResidualBuf, @LResidual[0], LBufSize);
    LL2Resid := 0.0;
    for LI := 0 to CHiddenDim - 1 do
      LL2Resid := LL2Resid + Double(LResidual[LI]) * Double(LResidual[LI]);
    LL2Resid := Sqrt(LL2Resid);
    TVdxUtils.PrintLn('  Post-PreAttnNorm L2: %.4f', [LL2Resid]);

    // Re-upload (download may have changed buffer? No — host-visible is coherent)
    // Actually the data is still there, Forward reads it directly
    LAttn.Forward(LResidualBuf, LAttnWeights, LNormWeights.QNormGpu,
      LNormWeights.KNormGpu, 0, 0, 10000.0, LAttnOutBuf);

    // Download GPU attention output
    SetLength(LAttnOut, CHiddenDim);
    LCompute.DownloadFromBuffer(LAttnOutBuf, @LAttnOut[0], LBufSize);

    LL2Attn := 0.0;
    for LI := 0 to CHiddenDim - 1 do
      LL2Attn := LL2Attn + Double(LAttnOut[LI]) * Double(LAttnOut[LI]);
    LL2Attn := Sqrt(LL2Attn);
    TVdxUtils.PrintLn('  GPU attention output L2: %.4f', [LL2Attn]);
    TVdxUtils.PrintLn('  First 4: [%.6f, %.6f, %.6f, %.6f]',
      [LAttnOut[0], LAttnOut[1], LAttnOut[2], LAttnOut[3]]);

    // ================================================================
    //  Step 3: CPU reference — V projection then expand GQA then O
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Step 3: CPU Reference (O x V, pos=0) ---');

    // 3a: V projection on CPU: V_weight[1024 x 2560] x normed_input[2560]
    LWeightPtr := LReader.GetTensorDataPtr('blk.0.attn_v.weight');
    SetLength(LVProj, CKVOutDim);

    for LI := 0 to CKVOutDim - 1 do
    begin
      LAcc := 0.0;
      for LJ := 0 to CHiddenDim - 1 do
        LAcc := LAcc + Double(F16ToF32(
          PWord(PByte(LWeightPtr) +
            (UInt64(LI) * CHiddenDim + UInt64(LJ)) * 2)^)) *
          Double(LResidual[LJ]);
      LVProj[LI] := Single(LAcc);
    end;

    TVdxUtils.PrintLn('  V proj (CPU) first 4: [%.6f, %.6f, %.6f, %.6f]',
      [LVProj[0], LVProj[1], LVProj[2], LVProj[3]]);

    // 3b: Expand V to GQA attention output (2048 elements)
    // Q heads 0,1 -> KV head 0; Q heads 2,3 -> KV head 1; etc.
    SetLength(LExpectedAttnIn, CQOutDim);
    for LHead := 0 to CNumQHeads - 1 do
    begin
      LKVHead := LHead div (CNumQHeads div CNumKVHeads);
      for LI := 0 to CHeadDim - 1 do
        LExpectedAttnIn[LHead * CHeadDim + LI] :=
          LVProj[LKVHead * CHeadDim + LI];
    end;

    // 3c: O projection on CPU: O_weight[2560 x 2048] x attn_concat[2048]
    LWeightPtr := LReader.GetTensorDataPtr('blk.0.attn_output.weight');
    SetLength(LExpectedOut, CHiddenDim);

    for LI := 0 to CHiddenDim - 1 do
    begin
      LAcc := 0.0;
      for LJ := 0 to CQOutDim - 1 do
        LAcc := LAcc + Double(F16ToF32(
          PWord(PByte(LWeightPtr) +
            (UInt64(LI) * CQOutDim + UInt64(LJ)) * 2)^)) *
          Double(LExpectedAttnIn[LJ]);
      LExpectedOut[LI] := Single(LAcc);
    end;

    LL2Resid := 0.0;
    for LI := 0 to CHiddenDim - 1 do
      LL2Resid := LL2Resid + Double(LExpectedOut[LI]) * Double(LExpectedOut[LI]);
    LL2Resid := Sqrt(LL2Resid);
    TVdxUtils.PrintLn('  CPU expected output L2: %.4f', [LL2Resid]);
    TVdxUtils.PrintLn('  First 4: [%.6f, %.6f, %.6f, %.6f]',
      [LExpectedOut[0], LExpectedOut[1], LExpectedOut[2], LExpectedOut[3]]);

    // ================================================================
    //  Step 4: Compare GPU vs CPU
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Step 4: GPU vs CPU Comparison ---');

    LMaxErr := 0.0;
    for LI := 0 to CHiddenDim - 1 do
    begin
      LErr := Abs(Double(LAttnOut[LI]) - Double(LExpectedOut[LI]));
      if LErr > LMaxErr then
        LMaxErr := LErr;
    end;

    TVdxUtils.PrintLn('  Max error: %.10f', [LMaxErr]);

    // Show first few comparisons
    for LI := 0 to 7 do
      TVdxUtils.PrintLn('  [%d] GPU=%.8f  CPU=%.8f  err=%.8f',
        [LI, LAttnOut[LI], LExpectedOut[LI],
         Abs(Double(LAttnOut[LI]) - Double(LExpectedOut[LI]))]);

    if LMaxErr < 0.1 then
      TVdxUtils.PrintLn(COLOR_GREEN +
        'PASSED: pos=0 attention output matches O x V (max err=%.2e)', [LMaxErr])
    else
      TVdxUtils.PrintLn(COLOR_RED +
        'FAILED: pos=0 attention output does NOT match O x V (max err=%.2e)', [LMaxErr]);

    // Cleanup
    LCompute.DestroyGpuBuffer(LResidualBuf);
    LCompute.DestroyGpuBuffer(LAttnOutBuf);
    LCompute.DestroyGpuBuffer(LNormWeights.AttnNormGpu);
    LCompute.DestroyGpuBuffer(LNormWeights.QNormGpu);
    LCompute.DestroyGpuBuffer(LNormWeights.KNormGpu);
    LAttn.FreeAttnWeights(LAttnWeights);
    LAttn.Cleanup();
    LNorm.Cleanup();
    LReader.Close();
  finally
    LAttn.Free();
    LNorm.Free();
    LCompute.Free();
    LReader.Free();
  end;
end;

// ============================================================================
//  Test_AttnScores — Dump real attention scores at pos=1 (2 positions)
//  After running Forward for BOS (pos=0) and "The" (pos=1), download
//  Q, K cache, and scores to verify the scoring path.
// ============================================================================

procedure Test_AttnScores();
const
  CGGUFPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';
  CHiddenDim = 2560;
  CNumQHeads = 8;
  CNumKVHeads = 4;
  CHeadDim = 256;
  CNumLayers = 34;
  CMaxSeqLen = 32;
  CQOutDim = 2048;
  CKVOutDim = 1024;
var
  LReader: TVdxGGUFReader;
  LCompute: TVdxVulkanCompute;
  LNorm: TVdxLayerNorm;
  LAttn: TVdxAttention;
  LAttnWeights: TVdxAttnLayerWeights;
  LNormWeights: TVdxNormLayerWeights;
  LResidualBuf: TVdxGpuBuffer;
  LAttnOutBuf: TVdxGpuBuffer;
  LResidual: array of Single;
  LQData: array of Single;
  LKCacheData: array of Single;
  LScoresData: array of Single;
  LAttnOutData: array of Single;
  LAttnOut: array of Single;
  LEmbedPtr: PByte;
  LBosTokenId: Integer;
  LTheTokenId: Integer;
  LEmbedScale: Single;
  LBufSize: UInt64;
  LI: Integer;
  LAcc: Double;
  LScale: Double;
  LVocab: TVdxGGUFMetaValue;
  LVocabCount: Integer;
  LKCacheSize: UInt64;

  function F16ToF32(const AVal: UInt16): Single;
  const
    CSignBit: UInt32 = $80000000;
  var
    LExp: UInt32;
    LMant: UInt32;
    LBits: UInt32;
  begin
    LExp := (UInt32(AVal) shr 10) and $1F;
    LMant := UInt32(AVal) and $3FF;
    if LExp = 0 then
    begin
      if LMant = 0 then
        LBits := 0
      else
      begin
        LExp := 1;
        while (LMant and $400) = 0 do
        begin
          LMant := LMant shl 1;
          Inc(LExp);
        end;
        LMant := LMant and $3FF;
        LBits := UInt32((113 - LExp) shl 23) or (LMant shl 13);
      end;
    end
    else if LExp = $1F then
      LBits := UInt32($FF shl 23) or (LMant shl 13)
    else
      LBits := UInt32((LExp + 112) shl 23) or (LMant shl 13);
    if (AVal and $8000) <> 0 then
      LBits := LBits or CSignBit;
    Move(LBits, Result, 4);
  end;

  function UploadNormWeight(const ATensorName: string;
    const ACount: UInt32): TVdxGpuBuffer;
  var
    LPtr: Pointer;
    LData: array of Single;
    LK: Integer;
  begin
    LPtr := LReader.GetTensorDataPtr(ATensorName);
    SetLength(LData, ACount);
    Move(LPtr^, LData[0], ACount * SizeOf(Single));
    for LK := 0 to ACount - 1 do
      LData[LK] := LData[LK] + 1.0;
    Result := LCompute.CreateGpuBuffer(
      UInt64(ACount) * SizeOf(Single),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    LCompute.UploadToBuffer(Result, @LData[0], UInt64(ACount) * SizeOf(Single));
  end;

  function FindTokenId(const ATokenStr: string): Integer;
  var
    LK: Integer;
  begin
    for LK := 0 to LVocabCount - 1 do
    begin
      if LVocab.ArrayItems[LK].AsString = ATokenStr then
        Exit(LK);
    end;
    Result := -1;
  end;

  // Read back from device-local buffer via staging copy
  procedure ReadbackDeviceLocal(const ASrc: TVdxGpuBuffer;
    const ADest: Pointer; const ASize: UInt64);
  var
    LStaging: TVdxGpuBuffer;
  begin
    LStaging := LCompute.CreateGpuBuffer(ASize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    try
      LCompute.CopyBuffer(ASrc, LStaging, ASize);
      LCompute.DownloadFromBuffer(LStaging, ADest, ASize);
    finally
      LCompute.DestroyGpuBuffer(LStaging);
    end;
  end;

begin
  TVdxUtils.PrintLn(COLOR_CYAN + '=== Test_AttnScores: Attention Scores at pos=1 ===');
  TVdxUtils.PrintLn('');

  LBufSize := UInt64(CHiddenDim) * SizeOf(Single);
  LEmbedScale := Sqrt(Single(CHiddenDim));
  LScale := 1.0 / Sqrt(Double(CHeadDim));  // 1/sqrt(256) = 1/16

  LReader := TVdxGGUFReader.Create();
  LCompute := TVdxVulkanCompute.Create();
  LNorm := TVdxLayerNorm.Create();
  LAttn := TVdxAttention.Create();
  try
    LReader.SetStatusCallback(StatusCallback);
    LCompute.SetStatusCallback(StatusCallback);
    LNorm.SetStatusCallback(StatusCallback);

    TVdxUtils.FailIf(not LReader.Open(CGGUFPath),
      'Failed to open GGUF: %s', [CGGUFPath]);
    LCompute.Init();
    LNorm.Init(LCompute);
    LAttn.Init(LCompute, CHiddenDim, CNumQHeads, CNumKVHeads,
      CHeadDim, CNumLayers, CMaxSeqLen);

    // Upload layer 0 weights
    TVdxUtils.PrintLn('Uploading layer 0 weights...');
    LAttn.UploadAttnWeights(LReader, 0, LAttnWeights);
    FillChar(LNormWeights, SizeOf(LNormWeights), 0);
    LNormWeights.AttnNormGpu := UploadNormWeight(
      'blk.0.attn_norm.weight', CHiddenDim);
    LNormWeights.QNormGpu := UploadNormWeight(
      'blk.0.attn_q_norm.weight', CHeadDim);
    LNormWeights.KNormGpu := UploadNormWeight(
      'blk.0.attn_k_norm.weight', CHeadDim);

    // Get vocab for token lookup
    LVocab := LReader.GetMetadata('tokenizer.ggml.tokens');
    LVocabCount := Length(LVocab.ArrayItems);
    LBosTokenId := Integer(LReader.GetMetadataUInt32(
      'tokenizer.ggml.bos_token_id', 2));
    LTheTokenId := FindTokenId(#$2581'The');
    TVdxUtils.FailIf(LTheTokenId < 0, 'Token "The" not found', []);
    TVdxUtils.PrintLn('  BOS id=%d, "The" id=%d', [LBosTokenId, LTheTokenId]);

    LEmbedPtr := PByte(LReader.GetTensorDataPtr('token_embd.weight'));

    // Work buffers
    LResidualBuf := LCompute.CreateGpuBuffer(LBufSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    LAttnOutBuf := LCompute.CreateGpuBuffer(LBufSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    SetLength(LResidual, CHiddenDim);

    // ================================================================
    //  Position 0: BOS token — populate KV cache
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Position 0: BOS token ---');

    for LI := 0 to CHiddenDim - 1 do
      LResidual[LI] := F16ToF32(PWord(LEmbedPtr +
        UInt64(LBosTokenId) * UInt64(CHiddenDim) * 2 +
        UInt64(LI) * 2)^) * LEmbedScale;

    LCompute.UploadToBuffer(LResidualBuf, @LResidual[0], LBufSize);
    LNorm.Apply(LResidualBuf, LNormWeights.AttnNormGpu, CHiddenDim);
    LAttn.Forward(LResidualBuf, LAttnWeights, LNormWeights.QNormGpu,
      LNormWeights.KNormGpu, 0, 0, 10000.0, LAttnOutBuf);
    TVdxUtils.PrintLn('  Forward at pos=0 done (KV cache populated)');

    // ================================================================
    //  Position 1: "The" token — this is where scores matter
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Position 1: "The" token (seq_len=2) ---');

    for LI := 0 to CHiddenDim - 1 do
      LResidual[LI] := F16ToF32(PWord(LEmbedPtr +
        UInt64(LTheTokenId) * UInt64(CHiddenDim) * 2 +
        UInt64(LI) * 2)^) * LEmbedScale;

    LCompute.UploadToBuffer(LResidualBuf, @LResidual[0], LBufSize);
    LNorm.Apply(LResidualBuf, LNormWeights.AttnNormGpu, CHiddenDim);
    LAttn.Forward(LResidualBuf, LAttnWeights, LNormWeights.QNormGpu,
      LNormWeights.KNormGpu, 0, 1, 10000.0, LAttnOutBuf);

    // ================================================================
    //  Dump internal buffers after Forward at pos=1
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Post-Forward Dumps ---');

    // Download Q (post QK-norm + RoPE)
    SetLength(LQData, CQOutDim);
    ReadbackDeviceLocal(LAttn.QBuf, @LQData[0],
      UInt64(CQOutDim) * SizeOf(Single));

    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  Q head 0 (post QK-norm + RoPE) first 4:');
    TVdxUtils.PrintLn('    [%.6f, %.6f, %.6f, %.6f]',
      [LQData[0], LQData[1], LQData[2], LQData[3]]);
    LAcc := 0.0;
    for LI := 0 to CHeadDim - 1 do
      LAcc := LAcc + Double(LQData[LI]) * Double(LQData[LI]);
    TVdxUtils.PrintLn('    L2: %.4f', [Sqrt(LAcc)]);

    // Download K cache (all KV heads for layer 0)
    LKCacheSize := UInt64(CNumKVHeads) * CMaxSeqLen * CHeadDim * SizeOf(Single);
    SetLength(LKCacheData, CNumKVHeads * CMaxSeqLen * CHeadDim);
    ReadbackDeviceLocal(LAttn.GetKCache(0), @LKCacheData[0], LKCacheSize);

    // Print K cache entries for KV head 0, positions 0 and 1
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  K cache (KV head 0, pos 0) first 4:');
    TVdxUtils.PrintLn('    [%.6f, %.6f, %.6f, %.6f]',
      [LKCacheData[0 * CHeadDim + 0],
       LKCacheData[0 * CHeadDim + 1],
       LKCacheData[0 * CHeadDim + 2],
       LKCacheData[0 * CHeadDim + 3]]);

    TVdxUtils.PrintLn('  K cache (KV head 0, pos 1) first 4:');
    TVdxUtils.PrintLn('    [%.6f, %.6f, %.6f, %.6f]',
      [LKCacheData[1 * CHeadDim + 0],
       LKCacheData[1 * CHeadDim + 1],
       LKCacheData[1 * CHeadDim + 2],
       LKCacheData[1 * CHeadDim + 3]]);

    // ================================================================
    //  CPU reference: compute Q·K scores for Q head 0 vs K cache
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- CPU Score Computation (Q head 0) ---');

    // Q head 0 uses KV head 0 (GQA: head 0 div 2 = 0)
    // K cache layout: [KVHead][MaxSeqLen][HeadDim]
    // KV head 0: starts at index 0

    // Score for pos=0: Q[0..255] dot K_cache[KVHead0, pos0, 0..255]
    LAcc := 0.0;
    for LI := 0 to CHeadDim - 1 do
      LAcc := LAcc + Double(LQData[LI]) *
              Double(LKCacheData[0 * CHeadDim + LI]);
    TVdxUtils.PrintLn('  Raw score pos=0: %.6f (scaled: %.6f)',
      [LAcc, LAcc * LScale]);

    // Score for pos=1: Q[0..255] dot K_cache[KVHead0, pos1, 0..255]
    LAcc := 0.0;
    for LI := 0 to CHeadDim - 1 do
      LAcc := LAcc + Double(LQData[LI]) *
              Double(LKCacheData[1 * CHeadDim + LI]);
    TVdxUtils.PrintLn('  Raw score pos=1: %.6f (scaled: %.6f)',
      [LAcc, LAcc * LScale]);

    // Download GPU scores buffer (contains post-softmax weights from
    // the LAST Q head processed, which is head 7)
    SetLength(LScoresData, CMaxSeqLen);
    ReadbackDeviceLocal(LAttn.ScoresBuf, @LScoresData[0],
      UInt64(CMaxSeqLen) * SizeOf(Single));

    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  GPU ScoresBuf (post-softmax, LAST head = head 7):');
    TVdxUtils.PrintLn('    pos=0 weight: %.8f', [LScoresData[0]]);
    TVdxUtils.PrintLn('    pos=1 weight: %.8f', [LScoresData[1]]);
    TVdxUtils.PrintLn('    sum: %.8f (should be 1.0)',
      [Double(LScoresData[0]) + Double(LScoresData[1])]);

    // Download attention output for magnitude check
    SetLength(LAttnOutData, CQOutDim);
    ReadbackDeviceLocal(LAttn.AttnOutBufInternal, @LAttnOutData[0],
      UInt64(CQOutDim) * SizeOf(Single));
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  AttnOut (pre-O-projection) head 0 first 4:');
    TVdxUtils.PrintLn('    [%.6f, %.6f, %.6f, %.6f]',
      [LAttnOutData[0], LAttnOutData[1], LAttnOutData[2], LAttnOutData[3]]);

    // Final attention output
    SetLength(LAttnOut, CHiddenDim);
    LCompute.DownloadFromBuffer(LAttnOutBuf, @LAttnOut[0], LBufSize);
    LAcc := 0.0;
    for LI := 0 to CHiddenDim - 1 do
      LAcc := LAcc + Double(LAttnOut[LI]) * Double(LAttnOut[LI]);
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  Final attention output L2: %.4f', [Sqrt(LAcc)]);
    TVdxUtils.PrintLn('  First 4: [%.6f, %.6f, %.6f, %.6f]',
      [LAttnOut[0], LAttnOut[1], LAttnOut[2], LAttnOut[3]]);

    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_GREEN + 'Test_AttnScores complete. Review scores above.');

    // Cleanup
    LCompute.DestroyGpuBuffer(LResidualBuf);
    LCompute.DestroyGpuBuffer(LAttnOutBuf);
    LCompute.DestroyGpuBuffer(LNormWeights.AttnNormGpu);
    LCompute.DestroyGpuBuffer(LNormWeights.QNormGpu);
    LCompute.DestroyGpuBuffer(LNormWeights.KNormGpu);
    LAttn.FreeAttnWeights(LAttnWeights);
    LAttn.Cleanup();
    LNorm.Cleanup();
    LReader.Close();
  finally
    LAttn.Free();
    LNorm.Free();
    LCompute.Free();
    LReader.Free();
  end;
end;

procedure Test10_QKNorm();
const
  CGGUFPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';
  CHiddenDim = 2560;
  CNumQHeads = 8;
  CNumKVHeads = 4;
  CHeadDim = 256;
  CNumLayers = 34;
  CMaxSeqLen = 32;
  CQOutDim = 2048;  // 8 heads * 256 head_dim
var
  LReader: TVdxGGUFReader;
  LCompute: TVdxVulkanCompute;
  LAttn: TVdxAttention;
  LAttnWeights: TVdxAttnLayerWeights;
  LInputBuf: TVdxGpuBuffer;
  LQOutBuf: TVdxGpuBuffer;
  LInputData: array of Single;
  LQPreNorm: array of Single;
  LQPostNormGpu: array of Single;
  LQPostNormCpu: array of Single;
  LQNormWeightRaw: array of Single;
  LQNormWeightPlus1: array of Single;
  LWeightPtr: Pointer;
  LQNormBuf: TVdxGpuBuffer;

  // QK-norm dispatch vars
  LDescPool: VkDescriptorPool;
  LDescSet: VkDescriptorSet;
  LQKNormPush: TVdxQKNormPush;
  LQKNormDescLayout: VkDescriptorSetLayout;
  LQKNormShader: VkShaderModule;
  LQKNormBundle: TVdxComputePipelineBundle;

  LSumSq: Double;
  LRms: Double;
  LInvRms: Double;
  LMaxErr: Double;
  LErr: Double;
  LPreNormL2: Double;
  LPostNormL2: Double;
  LEstScore: Double;
  LPassed: Boolean;
  LI: Integer;
  LHead: Integer;
  LOffset: Integer;
  LSpvPath: string;
  LSpvData: TBytes;

  function F16ToF32(const AVal: UInt16): Single;
  const
    CSignBit: UInt32 = $80000000;
  var
    LExp: UInt32;
    LMant: UInt32;
    LBits: UInt32;
  begin
    LExp := (UInt32(AVal) shr 10) and $1F;
    LMant := UInt32(AVal) and $3FF;
    if LExp = 0 then
    begin
      if LMant = 0 then
        LBits := 0
      else
      begin
        LExp := 1;
        while (LMant and $400) = 0 do
        begin
          LMant := LMant shl 1;
          Inc(LExp);
        end;
        LMant := LMant and $3FF;
        LBits := UInt32((113 - LExp) shl 23) or (LMant shl 13);
      end;
    end
    else if LExp = $1F then
      LBits := UInt32($FF shl 23) or (LMant shl 13)
    else
      LBits := UInt32((LExp + 112) shl 23) or (LMant shl 13);
    if (AVal and $8000) <> 0 then
      LBits := LBits or CSignBit;
    Move(LBits, Result, 4);
  end;

begin
  TVdxUtils.PrintLn(COLOR_CYAN + '=== Test 10: QK-Norm GPU vs CPU (Layer 0) ===');
  TVdxUtils.PrintLn('');

  LReader := TVdxGGUFReader.Create();
  LCompute := TVdxVulkanCompute.Create();
  LAttn := TVdxAttention.Create();
  try
    LReader.SetStatusCallback(StatusCallback);
    LCompute.SetStatusCallback(StatusCallback);

    TVdxUtils.FailIf(not LReader.Open(CGGUFPath),
      'Failed to open GGUF: %s', [CGGUFPath]);
    LCompute.Init();
    LAttn.Init(LCompute, CHiddenDim, CNumQHeads, CNumKVHeads,
      CHeadDim, CNumLayers, CMaxSeqLen);

    TVdxUtils.PrintLn('Uploading layer 0 attention weights...');
    LAttn.UploadAttnWeights(LReader, 0, LAttnWeights);

    // ================================================================
    //  Step 1: Print raw QK-norm weight values from GGUF
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Raw attn_q_norm.weight from GGUF (first 8) ---');
    LWeightPtr := LReader.GetTensorDataPtr('blk.0.attn_q_norm.weight');
    SetLength(LQNormWeightRaw, CHeadDim);
    Move(LWeightPtr^, LQNormWeightRaw[0], CHeadDim * SizeOf(Single));
    for LI := 0 to 7 do
      TVdxUtils.PrintLn('  [%d] = %.8f', [LI, LQNormWeightRaw[LI]]);

    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  If values ~0: needs +1 (offset-from-1 convention)');
    TVdxUtils.PrintLn('  If values ~1: already scaled, +1 would be WRONG');

    // Prepare weight+1 array (same as UploadNormWeight does)
    SetLength(LQNormWeightPlus1, CHeadDim);
    for LI := 0 to CHeadDim - 1 do
      LQNormWeightPlus1[LI] := LQNormWeightRaw[LI] + 1.0;

    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn('  After +1 (first 8):');
    for LI := 0 to 7 do
      TVdxUtils.PrintLn('  [%d] = %.8f', [LI, LQNormWeightPlus1[LI]]);

    // Upload QK-norm weight with +1
    LQNormBuf := LCompute.CreateGpuBuffer(
      UInt64(CHeadDim) * SizeOf(Single),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    LCompute.UploadToBuffer(LQNormBuf, @LQNormWeightPlus1[0],
      UInt64(CHeadDim) * SizeOf(Single));

    // ================================================================
    //  Step 2: Q projection on known input
    // ================================================================
    SetLength(LInputData, CHiddenDim);
    for LI := 0 to CHiddenDim - 1 do
      LInputData[LI] := (LI + 1) / CHiddenDim;

    LInputBuf := LCompute.CreateGpuBuffer(
      UInt64(CHiddenDim) * SizeOf(Single),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    LCompute.UploadToBuffer(LInputBuf, @LInputData[0],
      UInt64(CHiddenDim) * SizeOf(Single));

    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Q Projection (matvec F16) ---');

    LQOutBuf := LCompute.CreateGpuBuffer(
      UInt64(CQOutDim) * SizeOf(Single),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    LAttn.TestMatVec(LAttnWeights.QWeightGpu, LInputBuf, LQOutBuf,
      CHiddenDim, CQOutDim);

    // Download pre-QKnorm Q
    SetLength(LQPreNorm, CQOutDim);
    LCompute.DownloadFromBuffer(LQOutBuf, @LQPreNorm[0],
      UInt64(CQOutDim) * SizeOf(Single));

    LPreNormL2 := 0.0;
    for LI := 0 to CHeadDim - 1 do
      LPreNormL2 := LPreNormL2 + Double(LQPreNorm[LI]) * Double(LQPreNorm[LI]);
    LPreNormL2 := Sqrt(LPreNormL2);

    TVdxUtils.PrintLn('  Q head 0 pre-norm L2: %.6f', [LPreNormL2]);
    TVdxUtils.PrintLn('  Q head 0 first 4: [%.6f, %.6f, %.6f, %.6f]',
      [LQPreNorm[0], LQPreNorm[1], LQPreNorm[2], LQPreNorm[3]]);

    // ================================================================
    //  Step 3: Apply QK-norm on GPU
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- QK-Norm on GPU ---');

    LSpvPath := TPath.Combine(
      TPath.GetDirectoryName(ParamStr(0)),
      '..\shaders\qk_norm.spv');
    LSpvPath := TPath.GetFullPath(LSpvPath);
    LSpvData := TFile.ReadAllBytes(LSpvPath);
    LQKNormShader := LCompute.CreateShaderModule(
      @LSpvData[0], NativeUInt(Length(LSpvData)));
    LQKNormDescLayout := LCompute.CreateStorageDescriptorSetLayout(2);
    LQKNormBundle := LCompute.CreateComputePipelineWithPush(
      LQKNormShader, 'main', LQKNormDescLayout, SizeOf(TVdxQKNormPush));

    LQKNormPush.HeadDim := CHeadDim;
    LQKNormPush.NumHeads := CNumQHeads;
    LQKNormPush.Eps := 1e-6;

    LDescPool := LCompute.CreateDescriptorPoolForStorage(1, 2);
    try
      LDescSet := LCompute.AllocateDescriptorSetForBuffers(
        LDescPool, LQKNormDescLayout, [LQOutBuf, LQNormBuf]);
      LCompute.DispatchComputeWithPush(
        LQKNormBundle.Pipeline, LQKNormBundle.PipelineLayout,
        LDescSet, @LQKNormPush, SizeOf(LQKNormPush), CNumQHeads);
    finally
      LCompute.DestroyDescriptorPoolHandle(LDescPool);
    end;

    SetLength(LQPostNormGpu, CQOutDim);
    LCompute.DownloadFromBuffer(LQOutBuf, @LQPostNormGpu[0],
      UInt64(CQOutDim) * SizeOf(Single));

    LPostNormL2 := 0.0;
    for LI := 0 to CHeadDim - 1 do
      LPostNormL2 := LPostNormL2 + Double(LQPostNormGpu[LI]) * Double(LQPostNormGpu[LI]);
    LPostNormL2 := Sqrt(LPostNormL2);

    TVdxUtils.PrintLn('  Q head 0 post-norm L2 (GPU): %.6f', [LPostNormL2]);
    TVdxUtils.PrintLn('  Q head 0 first 4: [%.6f, %.6f, %.6f, %.6f]',
      [LQPostNormGpu[0], LQPostNormGpu[1], LQPostNormGpu[2], LQPostNormGpu[3]]);
    TVdxUtils.PrintLn('  Scale factor (post/pre L2): %.4f', [LPostNormL2 / LPreNormL2]);

    // ================================================================
    //  Step 4: CPU reference QK-norm (all heads)
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- CPU Reference QK-Norm ---');

    SetLength(LQPostNormCpu, CQOutDim);

    for LHead := 0 to CNumQHeads - 1 do
    begin
      LOffset := LHead * CHeadDim;

      LSumSq := 0.0;
      for LI := 0 to CHeadDim - 1 do
        LSumSq := LSumSq + Double(LQPreNorm[LOffset + LI]) *
                            Double(LQPreNorm[LOffset + LI]);
      LRms := Sqrt(LSumSq / CHeadDim + 1e-6);
      LInvRms := 1.0 / LRms;

      for LI := 0 to CHeadDim - 1 do
        LQPostNormCpu[LOffset + LI] := Single(
          Double(LQPreNorm[LOffset + LI]) * LInvRms *
          Double(LQNormWeightPlus1[LI]));

      if LHead = 0 then
      begin
        TVdxUtils.PrintLn('  Head 0 RMS: %.8f', [LRms]);
        TVdxUtils.PrintLn('  Head 0 InvRMS: %.8f', [LInvRms]);
      end;
    end;

    TVdxUtils.PrintLn('  CPU head 0 first 4: [%.6f, %.6f, %.6f, %.6f]',
      [LQPostNormCpu[0], LQPostNormCpu[1], LQPostNormCpu[2], LQPostNormCpu[3]]);

    // ================================================================
    //  Step 5: Compare GPU vs CPU (all heads)
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- GPU vs CPU Comparison ---');

    LMaxErr := 0.0;
    LPassed := True;
    for LI := 0 to CQOutDim - 1 do
    begin
      LErr := Abs(Double(LQPostNormGpu[LI]) - Double(LQPostNormCpu[LI]));
      if LErr > LMaxErr then
        LMaxErr := LErr;
      if (LErr > 0.01) and LPassed then
      begin
        TVdxUtils.PrintLn(COLOR_RED +
          '  MISMATCH [%d] (head %d, dim %d): GPU=%.8f CPU=%.8f err=%.8f',
          [LI, LI div CHeadDim, LI mod CHeadDim,
           LQPostNormGpu[LI], LQPostNormCpu[LI], LErr]);
        LPassed := False;
      end;
    end;

    TVdxUtils.PrintLn('  Max error across all heads: %.10f', [LMaxErr]);

    // ================================================================
    //  Step 6: Estimate attention score magnitude
    // ================================================================
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_YELLOW + '--- Attention Score Magnitude Estimate ---');

    LEstScore := LPostNormL2 * LPostNormL2 / Sqrt(CHeadDim);
    TVdxUtils.PrintLn('  Post-norm Q head 0 L2: %.4f', [LPostNormL2]);
    TVdxUtils.PrintLn('  Estimated max score (L2^2 / sqrt(d)): %.4f', [LEstScore]);
    TVdxUtils.PrintLn('  If > 50: softmax likely near one-hot (pathological)');
    TVdxUtils.PrintLn('  If 1-20: reasonable range for attention scores');

    // ================================================================
    //  Verdict
    // ================================================================
    TVdxUtils.PrintLn('');
    if LPassed then
      TVdxUtils.PrintLn(COLOR_GREEN +
        'TEST 10 PASSED: QK-norm GPU matches CPU (max err=%.2e)', [LMaxErr])
    else
      TVdxUtils.PrintLn(COLOR_RED +
        'TEST 10 FAILED: QK-norm GPU/CPU mismatch (max err=%.2e)', [LMaxErr]);

    // Cleanup
    LCompute.DestroyGpuBuffer(LQOutBuf);
    LCompute.DestroyGpuBuffer(LQNormBuf);
    LCompute.DestroyGpuBuffer(LInputBuf);
    LCompute.DestroyComputePipelineBundle(LQKNormBundle);
    LCompute.DestroyDescriptorSetLayoutHandle(LQKNormDescLayout);
    LCompute.DestroyShaderModuleHandle(LQKNormShader);
    LAttn.FreeAttnWeights(LAttnWeights);
    LAttn.Cleanup();
    LReader.Close();
  finally
    LAttn.Free();
    LCompute.Free();
    LReader.Free();
  end;
end;

procedure RunVdxTestbed();
var
  LIndex: Integer;
begin
  try
    LIndex := 9;

    case LIndex of
      1: Test01();
      2: Test02();
      3: Test03();
      4: Test04();
      5: Test05();
      6: Test06();
      7: Test07();
      8: Test08();
      9: TestInference();
      10: Test10_QKNorm();
      11: Test_AttnTrace();
      12: Test_AttnScores();
      13: Test_FFNCompare();
      14: Test_Basics();
    end;
  except
    on E: Exception do
    begin
      TVdxUtils.PrintLn('');
      TVdxUtils.PrintLn(COLOR_RED + 'EXCEPTION: %s', [E.Message]);
    end;
  end;

  if TVdxUtils.RunFromIDE() then
    TVdxUtils.Pause();
end;

end.
