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
  VindexLLM.LayerNorm;

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

procedure RunVdxTestbed();
var
  LIndex: Integer;
begin
  try
    LIndex := 7;

    case LIndex of
      1: Test01();
      2: Test02();
      3: Test03();
      4: Test04();
      5: Test05();
      6: Test06();
      7: Test07();
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
