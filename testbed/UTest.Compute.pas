{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.Compute;

interface

uses
  VindexLLM.TestCase,
  VindexLLM.Compute;

type

  { TComputeTest }
  TComputeTest = class(TVdxTestCase)
  private
    FCompute: TVdxCompute;

    procedure SecEnumerateGpus();
    procedure SecInitializeAuto();
    procedure SecInitializeInvalidIndex();
    procedure SecBufferRoundtrip();
    procedure SecMapPersistent();
    procedure SecCopyBuffer();
    procedure SecShaderModuleLifecycle();
    procedure SecPipelineBuildAndDispatch();
    procedure SecBatchMode();
    procedure SecBatchMisuseErrors();
  protected
    procedure Run(); override;
  public
    constructor Create(); override;
    destructor  Destroy(); override;
  end;

implementation

uses
  System.SysUtils,
  System.Generics.Collections,
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.Shaders;

{ TComputeTest }

constructor TComputeTest.Create();
begin
  inherited;
  Title := 'Test_Compute';
  FCompute := nil;
end;

destructor TComputeTest.Destroy();
begin
  FCompute.Free();
  inherited;
end;

procedure TComputeTest.Run();
begin
  SecEnumerateGpus();
  SecInitializeAuto();
  SecInitializeInvalidIndex();
  SecBufferRoundtrip();
  SecMapPersistent();
  SecCopyBuffer();
  SecShaderModuleLifecycle();
  SecPipelineBuildAndDispatch();
  SecBatchMode();
  SecBatchMisuseErrors();
end;

// ---------------------------------------------------------------------------
// SecEnumerateGpus
// ---------------------------------------------------------------------------
procedure TComputeTest.SecEnumerateGpus();
var
  LProbe: TVdxCompute;
  LGpus: TArray<TVdxGpuInfo>;
  LI: Integer;
  LInfo: TVdxGpuInfo;
  LAnyCompute: Boolean;
begin
  Section('EnumerateGpus');

  LProbe := TVdxCompute.Create();
  try
    Check(LProbe.EnumerateGpus(LGpus), 'EnumerateGpus returns True');
    FlushErrors(LProbe.GetErrors());

    Check(Length(LGpus) >= 1, 'At least one Vulkan device enumerated');

    LAnyCompute := False;
    for LI := 0 to High(LGpus) do
    begin
      LInfo := LGpus[LI];
      TVdxUtils.PrintLn(
        '    [%d] %s  type=%d  API=%s  VRAM=%d MB  compute=%s',
        [LInfo.Index, LInfo.Name, Ord(LInfo.DeviceType), LInfo.ApiVersionStr,
         LInfo.VRAMMB, BoolToStr(LInfo.HasComputeQueue, True)]);
      if LInfo.HasComputeQueue then
        LAnyCompute := True;
    end;

    Check(LAnyCompute, 'At least one GPU has a compute queue');
  finally
    LProbe.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecInitializeAuto
// ---------------------------------------------------------------------------
procedure TComputeTest.SecInitializeAuto();
begin
  Section('Init (auto-select)');

  FCompute := TVdxCompute.Create();
  FCompute.Init();
  Check(not FCompute.GetErrors().HasErrors(), 'Init() succeeds without errors');
  FlushErrors(FCompute.GetErrors());

  Check(FCompute.GetSelectedGpuIndex() >= 0,
    'GetSelectedGpuIndex >= 0 after successful init');
  Check(FCompute.GetDeviceName() <> '', 'GetDeviceName non-empty');
  Check(FCompute.GetVRAMSizeMB() > 0, 'GetVRAMSizeMB > 0');
  Check(FCompute.GetMaxComputeWorkGroupSize() > 0,
    'GetMaxComputeWorkGroupSize > 0');

  TVdxUtils.PrintLn('    Selected GPU: %s (%d MB VRAM, index %d)',
    [FCompute.GetDeviceName(), FCompute.GetVRAMSizeMB(),
     FCompute.GetSelectedGpuIndex()]);
end;

// ---------------------------------------------------------------------------
// SecInitializeInvalidIndex
// ---------------------------------------------------------------------------
procedure TComputeTest.SecInitializeInvalidIndex();
var
  LLocal: TVdxCompute;
begin
  Section('Init (invalid GPU index)');

  LLocal := TVdxCompute.Create();
  try
    LLocal.Init(999);
    Check(LLocal.GetErrors().HasErrors(),
      'Init(999) produces errors');
    FlushErrors(LLocal.GetErrors());
  finally
    LLocal.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecBufferRoundtrip
// ---------------------------------------------------------------------------
procedure TComputeTest.SecBufferRoundtrip();
const
  CSize: UInt64 = 4096;
var
  LBuf: TVdxGpuBuffer;
  LSrc, LDst: TBytes;
  LI: Integer;
  LMatches: Boolean;
begin
  Section('Buffer roundtrip (upload -> download)');

  SetLength(LSrc, CSize);
  SetLength(LDst, CSize);
  for LI := 0 to Integer(CSize) - 1 do
    LSrc[LI] := Byte(LI and $FF);

  LBuf := FCompute.CreateGpuBuffer(
    CSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  try
    Check(LBuf.Buffer <> VK_NULL_HANDLE, 'CreateGpuBuffer returned a handle');
    FlushErrors(FCompute.GetErrors());

    FCompute.UploadToBuffer(LBuf, @LSrc[0], CSize);
    FCompute.DownloadFromBuffer(LBuf, @LDst[0], CSize);
    Check(not FCompute.GetErrors().HasErrors(), 'Upload/Download without errors');
    FlushErrors(FCompute.GetErrors());

    LMatches := True;
    for LI := 0 to Integer(CSize) - 1 do
      if LDst[LI] <> LSrc[LI] then
      begin
        LMatches := False;
        Break;
      end;
    Check(LMatches, 'Downloaded bytes match uploaded pattern');
  finally
    FCompute.DestroyGpuBuffer(LBuf);
  end;
end;

// ---------------------------------------------------------------------------
// SecMapPersistent
// ---------------------------------------------------------------------------
procedure TComputeTest.SecMapPersistent();
const
  CCount: Integer = 256;
var
  LBuf: TVdxGpuBuffer;
  LPtr: Pointer;
  LDst: TBytes;
  LI: Integer;
  LMatches: Boolean;
begin
  Section('MapBufferPersistent (write -> unmap -> download -> verify)');

  LBuf := FCompute.CreateGpuBuffer(
    UInt64(CCount) * SizeOf(UInt32),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  try
    LPtr := FCompute.MapBufferPersistent(LBuf);
    Check(LPtr <> nil, 'Mapped pointer is non-nil');
    FlushErrors(FCompute.GetErrors());

    for LI := 0 to CCount - 1 do
      PUInt32Array(LPtr)^[LI] := UInt32(LI * 7 + 3);
    FCompute.UnmapBuffer(LBuf);

    SetLength(LDst, UInt64(CCount) * SizeOf(UInt32));
    FCompute.DownloadFromBuffer(LBuf, @LDst[0], UInt64(CCount) * SizeOf(UInt32));
    Check(not FCompute.GetErrors().HasErrors(), 'Download after persistent unmap');
    FlushErrors(FCompute.GetErrors());

    LMatches := True;
    for LI := 0 to CCount - 1 do
      if PUInt32Array(@LDst[0])^[LI] <> UInt32(LI * 7 + 3) then
      begin
        LMatches := False;
        Break;
      end;
    Check(LMatches, 'Round-tripped data matches pattern written through map');
  finally
    FCompute.DestroyGpuBuffer(LBuf);
  end;
end;

// ---------------------------------------------------------------------------
// SecCopyBuffer
// ---------------------------------------------------------------------------
procedure TComputeTest.SecCopyBuffer();
const
  CSize: UInt64 = 1024;
  CHalf: UInt64 = 512;
var
  LSrc, LDst: TVdxGpuBuffer;
  LSrcData, LDstData: TBytes;
  LI: Integer;
  LMatches: Boolean;
begin
  Section('CopyBuffer + CopyBufferRegion');

  SetLength(LSrcData, CSize);
  SetLength(LDstData, CSize);
  for LI := 0 to Integer(CSize) - 1 do
    LSrcData[LI] := Byte((LI * 3 + 17) and $FF);

  LSrc := FCompute.CreateGpuBuffer(CSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  LDst := FCompute.CreateGpuBuffer(CSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  try
    FlushErrors(FCompute.GetErrors());

    FCompute.UploadToBuffer(LSrc, @LSrcData[0], CSize);
    FCompute.CopyBuffer(LSrc, LDst, CSize);
    FCompute.DownloadFromBuffer(LDst, @LDstData[0], CSize);
    Check(not FCompute.GetErrors().HasErrors(), 'Copy + download without errors');
    FlushErrors(FCompute.GetErrors());

    LMatches := True;
    for LI := 0 to Integer(CSize) - 1 do
      if LDstData[LI] <> LSrcData[LI] then
      begin
        LMatches := False;
        Break;
      end;
    Check(LMatches, 'dst bytes match src after CopyBuffer');

    // CopyBufferRegion: zero dst, copy src[0..half] into dst[half..end]
    FillChar(LDstData[0], CSize, 0);
    FCompute.UploadToBuffer(LDst, @LDstData[0], CSize);
    FCompute.CopyBufferRegion(LSrc, 0, LDst, CHalf, CHalf);
    FCompute.DownloadFromBuffer(LDst, @LDstData[0], CSize);
    Check(not FCompute.GetErrors().HasErrors(), 'Region copy without errors');
    FlushErrors(FCompute.GetErrors());

    LMatches := True;
    for LI := 0 to Integer(CHalf) - 1 do
      if LDstData[LI] <> 0 then
      begin
        LMatches := False;
        Break;
      end;
    Check(LMatches, 'dst[0..half] unchanged (still zero)');

    LMatches := True;
    for LI := 0 to Integer(CHalf) - 1 do
      if LDstData[Integer(CHalf) + LI] <> LSrcData[LI] then
      begin
        LMatches := False;
        Break;
      end;
    Check(LMatches, 'dst[half..full] matches src[0..half]');
  finally
    FCompute.DestroyGpuBuffer(LSrc);
    FCompute.DestroyGpuBuffer(LDst);
  end;
end;

// ---------------------------------------------------------------------------
// SecShaderModuleLifecycle
// ---------------------------------------------------------------------------
procedure TComputeTest.SecShaderModuleLifecycle();
var
  LBytes: TBytes;
  LModule: VkShaderModule;
begin
  Section('Shader module lifecycle (VEC_ADD)');

  LBytes := VdxLoadShader('VEC_ADD');
  Check(Length(LBytes) > 0, 'VdxLoadShader returned non-empty bytes');
  Check(Length(LBytes) mod 4 = 0, 'SPIR-V length is multiple of 4 bytes');

  LModule := FCompute.CreateShaderModule(@LBytes[0], Length(LBytes));
  Check(LModule <> VK_NULL_HANDLE, 'CreateShaderModule returned a handle');
  FlushErrors(FCompute.GetErrors());

  FCompute.DestroyShaderModuleHandle(LModule);
end;

// ---------------------------------------------------------------------------
// SecPipelineBuildAndDispatch
// ---------------------------------------------------------------------------
procedure TComputeTest.SecPipelineBuildAndDispatch();
const
  CCount:  Integer = 1024;
  CLocalX: Integer = 256;
var
  LShaderBytes: TBytes;
  LModule:      VkShaderModule;
  LDescLayout:  VkDescriptorSetLayout;
  LDescPool:    VkDescriptorPool;
  LPipeline:    TVdxComputePipelineBundle;
  LBufA, LBufB: TVdxGpuBuffer;
  LDescSet:     VkDescriptorSet;
  LBufSize:     UInt64;
  LDataA, LDataB: array of Single;
  LResult: array of Single;
  LI: Integer;
  LGroupsX: UInt32;
  LPushCount: UInt32;
  LAllMatch: Boolean;
  LMaxErr: Single;
  LDiff: Single;
begin
  Section('Pipeline build + dispatch (VEC_ADD)');

  LBufSize := UInt64(CCount) * SizeOf(Single);
  LShaderBytes := VdxLoadShader('VEC_ADD');
  LModule      := FCompute.CreateShaderModule(@LShaderBytes[0], Length(LShaderBytes));
  LDescLayout  := FCompute.CreateStorageDescriptorSetLayout(2);
  LDescPool    := FCompute.CreateDescriptorPoolForStorage(1, 2);
  LPipeline    := FCompute.CreateComputePipelineWithPush(
    LModule, 'main', LDescLayout, SizeOf(UInt32));

  LBufA := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  LBufB := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  FlushErrors(FCompute.GetErrors());

  try
    SetLength(LDataA, CCount);
    SetLength(LDataB, CCount);
    for LI := 0 to CCount - 1 do
    begin
      LDataA[LI] := Single(LI) * 1.5;
      LDataB[LI] := Single(LI) * 0.25;
    end;
    FCompute.UploadToBuffer(LBufA, @LDataA[0], LBufSize);
    FCompute.UploadToBuffer(LBufB, @LDataB[0], LBufSize);

    LDescSet := FCompute.AllocateDescriptorSetForBuffers(
      LDescPool, LDescLayout, [LBufA, LBufB]);
    Check(LDescSet <> VK_NULL_HANDLE,
      'AllocateDescriptorSetForBuffers returned a set');
    FlushErrors(FCompute.GetErrors());

    LPushCount := UInt32(CCount);
    LGroupsX := (UInt32(CCount) + UInt32(CLocalX) - 1) div UInt32(CLocalX);
    FCompute.DispatchComputeWithPush(
      LPipeline.Pipeline, LPipeline.PipelineLayout, LDescSet,
      @LPushCount, SizeOf(UInt32), LGroupsX);
    Check(not FCompute.GetErrors().HasErrors(), 'Dispatch without errors');
    FlushErrors(FCompute.GetErrors());

    SetLength(LResult, CCount);
    FCompute.DownloadFromBuffer(LBufA, @LResult[0], LBufSize);
    FlushErrors(FCompute.GetErrors());

    LAllMatch := True;
    LMaxErr := 0;
    for LI := 0 to CCount - 1 do
    begin
      LDiff := Abs(LResult[LI] - (Single(LI) * 1.75));
      if LDiff > LMaxErr then
        LMaxErr := LDiff;
      if LDiff > 1e-3 then
        LAllMatch := False;
    end;
    TVdxUtils.PrintLn('    Max abs error: %.6g', [LMaxErr]);
    Check(LAllMatch, 'Dispatch output matches a[i] + b[i] within tol');
  finally
    FCompute.DestroyGpuBuffer(LBufA);
    FCompute.DestroyGpuBuffer(LBufB);
    FCompute.DestroyComputePipelineBundle(LPipeline);
    FCompute.DestroyDescriptorPoolHandle(LDescPool);
    FCompute.DestroyDescriptorSetLayoutHandle(LDescLayout);
    FCompute.DestroyShaderModuleHandle(LModule);
  end;
end;

// ---------------------------------------------------------------------------
// SecBatchMode
// ---------------------------------------------------------------------------
procedure TComputeTest.SecBatchMode();
const
  CCount:  Integer = 512;
  CLocalX: Integer = 256;
var
  LShaderBytes: TBytes;
  LModule:      VkShaderModule;
  LDescLayout:  VkDescriptorSetLayout;
  LDescPool:    VkDescriptorPool;
  LPipeline:    TVdxComputePipelineBundle;
  LBufA, LBufB: TVdxGpuBuffer;
  LDescSet:     VkDescriptorSet;
  LBufSize:     UInt64;
  LDataA, LDataB, LResult: array of Single;
  LI: Integer;
  LGroupsX: UInt32;
  LPushCount: UInt32;
  LAllMatch: Boolean;
begin
  Section('Batch mode (VEC_ADD recorded in batch)');

  LBufSize := UInt64(CCount) * SizeOf(Single);
  LShaderBytes := VdxLoadShader('VEC_ADD');
  LModule      := FCompute.CreateShaderModule(@LShaderBytes[0], Length(LShaderBytes));
  LDescLayout  := FCompute.CreateStorageDescriptorSetLayout(2);
  LDescPool    := FCompute.CreateDescriptorPoolForStorage(1, 2);
  LPipeline    := FCompute.CreateComputePipelineWithPush(
    LModule, 'main', LDescLayout, SizeOf(UInt32));

  LBufA := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  LBufB := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  FlushErrors(FCompute.GetErrors());

  try
    SetLength(LDataA, CCount);
    SetLength(LDataB, CCount);
    for LI := 0 to CCount - 1 do
    begin
      LDataA[LI] := 10.0;
      LDataB[LI] := Single(LI);
    end;
    FCompute.UploadToBuffer(LBufA, @LDataA[0], LBufSize);
    FCompute.UploadToBuffer(LBufB, @LDataB[0], LBufSize);

    LDescSet := FCompute.AllocateDescriptorSetForBuffers(
      LDescPool, LDescLayout, [LBufA, LBufB]);
    FlushErrors(FCompute.GetErrors());

    LPushCount := UInt32(CCount);
    LGroupsX   := (UInt32(CCount) + UInt32(CLocalX) - 1) div UInt32(CLocalX);

    FCompute.BeginBatch();
    FCompute.DispatchComputeWithPush(
      LPipeline.Pipeline, LPipeline.PipelineLayout, LDescSet,
      @LPushCount, SizeOf(UInt32), LGroupsX);
    FCompute.BatchBarrier();
    FCompute.EndBatch();
    Check(not FCompute.GetErrors().HasErrors(), 'Batch dispatch without errors');
    FlushErrors(FCompute.GetErrors());

    SetLength(LResult, CCount);
    FCompute.DownloadFromBuffer(LBufA, @LResult[0], LBufSize);
    FlushErrors(FCompute.GetErrors());

    LAllMatch := True;
    for LI := 0 to CCount - 1 do
      if Abs(LResult[LI] - (10.0 + Single(LI))) > 1e-3 then
      begin
        LAllMatch := False;
        Break;
      end;
    Check(LAllMatch, 'Batched dispatch output matches expected');
  finally
    FCompute.DestroyGpuBuffer(LBufA);
    FCompute.DestroyGpuBuffer(LBufB);
    FCompute.DestroyComputePipelineBundle(LPipeline);
    FCompute.DestroyDescriptorPoolHandle(LDescPool);
    FCompute.DestroyDescriptorSetLayoutHandle(LDescLayout);
    FCompute.DestroyShaderModuleHandle(LModule);
  end;
end;

// ---------------------------------------------------------------------------
// SecBatchMisuseErrors — verify batch misuse records errors (not exceptions)
// ---------------------------------------------------------------------------
procedure TComputeTest.SecBatchMisuseErrors();
var
  LLocal: TVdxCompute;
begin
  Section('Batch misuse errors');

  LLocal := TVdxCompute.Create();
  try
    LLocal.Init();
    Check(not LLocal.GetErrors().HasErrors(), 'Local compute Init()');
    FlushErrors(LLocal.GetErrors());

    // EndBatch without a prior BeginBatch
    LLocal.EndBatch();
    Check(LLocal.GetErrors().HasErrors(),
      'EndBatch without BeginBatch produces error');
    FlushErrors(LLocal.GetErrors());

    // BeginBatch twice
    LLocal.BeginBatch();
    LLocal.BeginBatch();
    Check(LLocal.GetErrors().HasErrors(),
      'Nested BeginBatch produces error');
    FlushErrors(LLocal.GetErrors());

    // Clean up — EndBatch to restore state (first BeginBatch succeeded)
    LLocal.EndBatch();
    FlushErrors(LLocal.GetErrors());
  finally
    LLocal.Free();
  end;
end;

end.
