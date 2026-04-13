unit UVdxTestbed;

interface

procedure RunVdxTestbed();

implementation

uses
  System.SysUtils,
  System.IOUtils,
  VindexLLM.Utils,
  VindexLLM.VulkanCompute;

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

procedure RunVdxTestbed();
var
  LIndex: Integer;
begin
  try
    LIndex := 3;

    case LIndex of
      1: Test01();
      2: Test02();
      3: Test03();
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
