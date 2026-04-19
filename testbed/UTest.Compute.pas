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
  // Full-surface coverage of TVdxCompute. Sections progress from
  // lightweight (enumeration) to heavy (dispatch) so a hardware
  // failure surfaces as early as possible. FCompute is created by
  // SecInitializeAuto and reused by every subsequent section;
  // negative-path sections create their own local instance.
  TComputeTest = class(TVdxTestCase)
  private
    FCompute: TVdxCompute;

    procedure SecEnumerateGpus();
    procedure SecInitializeAuto();
    procedure SecInitializeInvalidIndex();
    procedure SecBufferRoundtrip();
    procedure SecMapPersistent();
    procedure SecVRAMAccounting();
    procedure SecCopyBuffer();
    procedure SecStreamingStagingPool();
    procedure SecShaderModuleLifecycle();
    procedure SecPipelineBuildAndDispatch();
    procedure SecBatchMode();
    procedure SecBatchMisuseRaises();
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
  SecVRAMAccounting();
  SecCopyBuffer();
  SecStreamingStagingPool();
  SecShaderModuleLifecycle();
  SecPipelineBuildAndDispatch();
  SecBatchMode();
  SecBatchMisuseRaises();
end;

// ---------------------------------------------------------------------------
// SecEnumerateGpus — lightweight probe. Verifies EnumerateGpus returns
// at least one device and prints the descriptor for every entry. No
// dependency on FCompute having been initialized — uses a local probe
// that gets freed at the end of the section.
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
        '    [%d] %s  type=%d  API=%s  VRAM=%d MB  compute=%s  subgroup=%d',
        [LInfo.Index, LInfo.Name, Ord(LInfo.DeviceType), LInfo.ApiVersionStr,
         LInfo.VRAMMB, BoolToStr(LInfo.HasComputeQueue, True), LInfo.SubgroupSize]);
      TVdxUtils.PrintLn(
        '        fp16=%s  int8=%s  int16=%s  int64=%s  fp64=%s  maxInvoc=%d',
        [BoolToStr(LInfo.SupportsFp16, True),
         BoolToStr(LInfo.SupportsInt8, True),
         BoolToStr(LInfo.SupportsInt16, True),
         BoolToStr(LInfo.SupportsInt64, True),
         BoolToStr(LInfo.SupportsFp64, True),
         LInfo.MaxComputeWorkGroupInvocations]);

      if LInfo.HasComputeQueue then
        LAnyCompute := True;
    end;

    Check(LAnyCompute, 'At least one GPU has a compute queue');
  finally
    LProbe.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecInitializeAuto — create the shared FCompute with AGpuIndex = -1
// (auto-select). Subsequent sections reuse FCompute. If this section
// fails, downstream sections will cascade-fail, which is the intended
// behavior — nothing else is meaningful without a working device.
// ---------------------------------------------------------------------------
procedure TComputeTest.SecInitializeAuto();
begin
  Section('Initialize (auto-select)');

  FCompute := TVdxCompute.Create();
  Check(FCompute.Initialize(-1), 'Initialize(-1) returns True');
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
// SecInitializeInvalidIndex — fresh instance, Initialize(999). Expect
// False return and an esFatal VDX_ERROR_VK_GPU_INDEX_INVALID in the
// errors buffer. Local instance, not FCompute.
// ---------------------------------------------------------------------------
procedure TComputeTest.SecInitializeInvalidIndex();
var
  LLocal: TVdxCompute;
  LResult: Boolean;
  LErrors: TVdxErrors;
  LFoundCode: Boolean;
  LItems: TList<TVdxError>;
  LI: Integer;
begin
  Section('Initialize (invalid GPU index)');

  LLocal := TVdxCompute.Create();
  try
    LResult := LLocal.Initialize(999);
    Check(not LResult, 'Initialize(999) returns False');

    LErrors := LLocal.GetErrors();
    Check(LErrors.HasFatal(), 'FErrors.HasFatal = True after invalid index');

    LFoundCode := False;
    LItems := LErrors.GetItems();
    for LI := 0 to LItems.Count - 1 do
      if LItems[LI].Code = VDX_ERROR_VK_GPU_INDEX_INVALID then
      begin
        LFoundCode := True;
        Break;
      end;
    Check(LFoundCode, 'Error code VDX_ERROR_VK_GPU_INDEX_INVALID present');

    FlushErrors(LErrors);
  finally
    LLocal.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecBufferRoundtrip — allocate a host-visible+coherent buffer, upload
// a known byte pattern, download to a separate local buffer, verify
// byte-identical.
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
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  try
    Check(LBuf.Buffer <> VK_NULL_HANDLE, 'CreateGpuBuffer returned a handle');
    FlushErrors(FCompute.GetErrors());

    Check(FCompute.UploadToBuffer(LBuf, @LSrc[0], CSize),
      'UploadToBuffer returns True');
    Check(FCompute.DownloadFromBuffer(LBuf, @LDst[0], CSize),
      'DownloadFromBuffer returns True');
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
// SecMapPersistent — allocate host-visible buffer, MapBufferPersistent,
// write pattern through mapped pointer, Unmap, remap, verify data
// survived. Exercises the persistent-map code path distinct from the
// Upload/Download convenience wrappers.
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
  Section('MapBufferPersistent (write -> unmap -> remap -> read)');

  LBuf := FCompute.CreateGpuBuffer(
    UInt64(CCount) * SizeOf(UInt32),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  try
    // Write a pattern through a mapped pointer
    Check(FCompute.MapBufferPersistent(LBuf, LPtr),
      'MapBufferPersistent returns True');
    Check(LPtr <> nil, 'Mapped pointer is non-nil');
    FlushErrors(FCompute.GetErrors());

    for LI := 0 to CCount - 1 do
      PUInt32Array(LPtr)^[LI] := UInt32(LI * 7 + 3);
    FCompute.UnmapBuffer(LBuf);

    // Read back through DownloadFromBuffer (a separate code path)
    SetLength(LDst, UInt64(CCount) * SizeOf(UInt32));
    Check(FCompute.DownloadFromBuffer(LBuf, @LDst[0], UInt64(CCount) * SizeOf(UInt32)),
      'DownloadFromBuffer after persistent unmap');
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
// SecVRAMAccounting — create one buffer of each category with distinct
// sizes, verify GetVRAMUsage buckets match, destroy each and re-check.
// Running totals must return to zero after cleanup.
// ---------------------------------------------------------------------------
procedure TComputeTest.SecVRAMAccounting();
const
  CSizeWeight:  UInt64 = 4 * 1024;
  CSizeCache:   UInt64 = 8 * 1024;
  CSizeBuffer:  UInt64 = 12 * 1024;
var
  LBaseline: TVdxVRAMUsage;
  LW, LC, LB: TVdxGpuBuffer;
  LUsage: TVdxVRAMUsage;
begin
  Section('VRAM category accounting');

  LBaseline := FCompute.GetVRAMUsage();

  LW := FCompute.CreateGpuBuffer(CSizeWeight,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcWeight);
  LC := FCompute.CreateGpuBuffer(CSizeCache,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcCache);
  LB := FCompute.CreateGpuBuffer(CSizeBuffer,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  FlushErrors(FCompute.GetErrors());

  LUsage := FCompute.GetVRAMUsage();
  Check(LUsage.WeightsBytes = LBaseline.WeightsBytes + CSizeWeight,
    'Weights bucket incremented by weight buffer size');
  Check(LUsage.CacheBytes = LBaseline.CacheBytes + CSizeCache,
    'Cache bucket incremented by cache buffer size');
  Check(LUsage.BuffersBytes = LBaseline.BuffersBytes + CSizeBuffer,
    'Buffers bucket incremented by buffer size');
  Check(LUsage.TotalBytes = LUsage.WeightsBytes + LUsage.CacheBytes + LUsage.BuffersBytes,
    'TotalBytes = sum of buckets');

  FCompute.DestroyGpuBuffer(LW);
  FCompute.DestroyGpuBuffer(LC);
  FCompute.DestroyGpuBuffer(LB);

  LUsage := FCompute.GetVRAMUsage();
  Check(LUsage.WeightsBytes = LBaseline.WeightsBytes,
    'Weights bucket returns to baseline after destroy');
  Check(LUsage.CacheBytes = LBaseline.CacheBytes,
    'Cache bucket returns to baseline after destroy');
  Check(LUsage.BuffersBytes = LBaseline.BuffersBytes,
    'Buffers bucket returns to baseline after destroy');
end;

// ---------------------------------------------------------------------------
// SecCopyBuffer — allocate src and dst host-visible buffers, upload
// pattern to src, CopyBuffer src->dst, verify via download.
// Then test CopyBufferRegion with non-zero offsets.
// ---------------------------------------------------------------------------
procedure TComputeTest.SecCopyBuffer();
const
  CSize:    UInt64 = 1024;
  CHalf:    UInt64 = 512;
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
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  LDst := FCompute.CreateGpuBuffer(CSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  try
    FlushErrors(FCompute.GetErrors());

    Check(FCompute.UploadToBuffer(LSrc, @LSrcData[0], CSize),
      'Upload to src');
    Check(FCompute.CopyBuffer(LSrc, LDst, CSize),
      'CopyBuffer src -> dst');
    Check(FCompute.DownloadFromBuffer(LDst, @LDstData[0], CSize),
      'Download from dst');
    FlushErrors(FCompute.GetErrors());

    LMatches := True;
    for LI := 0 to Integer(CSize) - 1 do
      if LDstData[LI] <> LSrcData[LI] then
      begin
        LMatches := False;
        Break;
      end;
    Check(LMatches, 'dst bytes match src after CopyBuffer');

    // CopyBufferRegion: zero dst, copy src[0..CHalf] into dst[CHalf..CSize]
    FillChar(LDstData[0], CSize, 0);
    Check(FCompute.UploadToBuffer(LDst, @LDstData[0], CSize),
      'Zero dst via upload');
    Check(FCompute.CopyBufferRegion(LSrc, 0, LDst, CHalf, CHalf),
      'CopyBufferRegion src[0..half] -> dst[half..full]');
    Check(FCompute.DownloadFromBuffer(LDst, @LDstData[0], CSize),
      'Download dst after region copy');
    FlushErrors(FCompute.GetErrors());

    LMatches := True;
    // dst[0..CHalf] should be zero
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
// SecStreamingStagingPool — lifecycle + grow semantics of the shared
// staging pool owned by TVdxCompute, plus a functional data-path
// roundtrip that mirrors what TVdxAttention / TVdxFFN will do: fill
// staging host, CopyBuffer host -> device, then read back via a
// scratch buffer to prove bytes survived the round trip.
// ---------------------------------------------------------------------------
procedure TComputeTest.SecStreamingStagingPool();
const
  CCountSmall:  UInt32 = 2;
  CCountLarge:  UInt32 = 4;
  CBytesSmall:  UInt64 = 1024;
  CBytesLarge:  UInt64 = 4096;
var
  LSrcBytes: TBytes;
  LDstBytes: TBytes;
  LScratch:  TVdxGpuBuffer;
  LHost0AfterSmall: VkBuffer;
  LDev0AfterSmall:  VkBuffer;
  LI: Integer;
  LMatches: Boolean;
begin
  Section('Streaming staging pool');

  // Initial state
  Check(FCompute.GetStagingCapacity() = 0,
    'Capacity 0 before first EnsureStagingPool');
  Check(FCompute.GetStagingCount() = 0,
    'Count 0 before first EnsureStagingPool');
  Check(FCompute.GetStagingHost(0).Buffer = VK_NULL_HANDLE,
    'Host[0] null before first EnsureStagingPool');
  Check(FCompute.GetStagingDevice(0).Buffer = VK_NULL_HANDLE,
    'Device[0] null before first EnsureStagingPool');

  // Zero-arg requests are no-ops
  Check(FCompute.EnsureStagingPool(0, CBytesSmall),
    'EnsureStagingPool(0, bytes) returns True');
  Check(FCompute.EnsureStagingPool(CCountSmall, 0),
    'EnsureStagingPool(count, 0) returns True');
  Check(FCompute.GetStagingCapacity() = 0,
    'Capacity still 0 after zero-arg calls');
  Check(FCompute.GetStagingCount() = 0,
    'Count still 0 after zero-arg calls');
  FlushErrors(FCompute.GetErrors());

  // First real grow: 2 pairs × 1024 bytes
  Check(FCompute.EnsureStagingPool(CCountSmall, CBytesSmall),
    'EnsureStagingPool(2, CBytesSmall) grows pool');
  Check(FCompute.GetStagingCount() = CCountSmall,
    'Count == 2 after first grow');
  Check(FCompute.GetStagingCapacity() = CBytesSmall,
    'Capacity == CBytesSmall after first grow');
  Check(FCompute.GetStagingHost(0).Buffer <> VK_NULL_HANDLE,
    'Host[0] allocated');
  Check(FCompute.GetStagingHost(1).Buffer <> VK_NULL_HANDLE,
    'Host[1] allocated');
  Check(FCompute.GetStagingDevice(0).Buffer <> VK_NULL_HANDLE,
    'Device[0] allocated');
  Check(FCompute.GetStagingDevice(1).Buffer <> VK_NULL_HANDLE,
    'Device[1] allocated');
  Check(FCompute.GetStagingHost(0).Size = CBytesSmall,
    'Host[0] size == CBytesSmall');
  Check(FCompute.GetStagingDevice(0).Size = CBytesSmall,
    'Device[0] size == CBytesSmall');
  FlushErrors(FCompute.GetErrors());

  LHost0AfterSmall := FCompute.GetStagingHost(0).Buffer;
  LDev0AfterSmall  := FCompute.GetStagingDevice(0).Buffer;

  // Smaller request — no-op, handles unchanged
  Check(FCompute.EnsureStagingPool(1, CBytesSmall div 2),
    'EnsureStagingPool(smaller count + smaller bytes) returns True');
  Check(FCompute.GetStagingCount() = CCountSmall,
    'Count unchanged when smaller requested');
  Check(FCompute.GetStagingCapacity() = CBytesSmall,
    'Capacity unchanged when smaller requested');
  Check(FCompute.GetStagingHost(0).Buffer = LHost0AfterSmall,
    'Host[0] handle unchanged (no realloc)');
  Check(FCompute.GetStagingDevice(0).Buffer = LDev0AfterSmall,
    'Device[0] handle unchanged (no realloc)');
  FlushErrors(FCompute.GetErrors());

  // Grow both count and bytes
  Check(FCompute.EnsureStagingPool(CCountLarge, CBytesLarge),
    'EnsureStagingPool(4, CBytesLarge) grows pool');
  Check(FCompute.GetStagingCount() = CCountLarge,
    'Count == 4 after grow');
  Check(FCompute.GetStagingCapacity() = CBytesLarge,
    'Capacity == CBytesLarge after grow');
  Check(FCompute.GetStagingHost(3).Size = CBytesLarge,
    'Host[3] size == CBytesLarge after grow');
  Check(FCompute.GetStagingDevice(3).Size = CBytesLarge,
    'Device[3] size == CBytesLarge after grow');
  Check(FCompute.GetStagingHost(0).Buffer <> LHost0AfterSmall,
    'Host[0] handle replaced after grow');
  FlushErrors(FCompute.GetErrors());

  // Functional data-path roundtrip on pair index 2 (proves non-zero
  // indices work, not just pair 0). Mirrors what TVdxAttention will
  // do for each of Q/K/V/O.
  SetLength(LSrcBytes, CBytesLarge);
  SetLength(LDstBytes, CBytesLarge);
  for LI := 0 to Integer(CBytesLarge) - 1 do
    LSrcBytes[LI] := Byte((LI * 7 + 23) and $FF);

  LScratch := FCompute.CreateGpuBuffer(CBytesLarge,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  try
    FlushErrors(FCompute.GetErrors());
    Check(FCompute.UploadToBuffer(FCompute.GetStagingHost(2),
      @LSrcBytes[0], CBytesLarge),
      'Upload pattern into staging host pair[2]');
    Check(FCompute.CopyBuffer(FCompute.GetStagingHost(2),
      FCompute.GetStagingDevice(2), CBytesLarge),
      'CopyBuffer host[2] -> device[2]');
    Check(FCompute.CopyBuffer(FCompute.GetStagingDevice(2),
      LScratch, CBytesLarge),
      'CopyBuffer device[2] -> scratch');
    Check(FCompute.DownloadFromBuffer(LScratch, @LDstBytes[0], CBytesLarge),
      'Download scratch');
    FlushErrors(FCompute.GetErrors());

    LMatches := True;
    for LI := 0 to Integer(CBytesLarge) - 1 do
      if LDstBytes[LI] <> LSrcBytes[LI] then
      begin
        LMatches := False;
        Break;
      end;
    Check(LMatches,
      'Round-tripped bytes match source pattern');
  finally
    FCompute.DestroyGpuBuffer(LScratch);
  end;
end;

// ---------------------------------------------------------------------------
// SecShaderModuleLifecycle — VdxLoadShader('VEC_ADD') returns bytes,
// CreateShaderModule builds a module, DestroyShaderModuleHandle tears
// it down. No dispatch here — SecPipelineBuildAndDispatch does the
// full path.
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
// SecPipelineBuildAndDispatch — full end-to-end:
//   Load VEC_ADD -> create shader module -> descriptor set layout
//   (2 bindings) -> descriptor pool (1 set, 2 descriptors) ->
//   compute pipeline with push constants (1 UInt32) -> allocate
//   descriptor set bound to buffers A and B -> upload pattern ->
//   dispatch -> download -> verify a[i] == orig_a[i] + b[i].
//
// a[i] = i * 1.5, b[i] = i * 0.25, expected a[i] after = i * 1.75.
// ---------------------------------------------------------------------------
procedure TComputeTest.SecPipelineBuildAndDispatch();
const
  CCount:    Integer = 1024;
  CLocalX:   Integer = 256;
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
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  LBufB := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  FlushErrors(FCompute.GetErrors());

  try
    // Seed A and B with the test pattern
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
    Check(FCompute.DispatchComputeWithPush(
      LPipeline.Pipeline, LPipeline.PipelineLayout, LDescSet,
      @LPushCount, SizeOf(UInt32), LGroupsX),
      'DispatchComputeWithPush returns True');
    FlushErrors(FCompute.GetErrors());

    // Download A (mutated in place) and verify
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
// SecBatchMode — same VEC_ADD dispatch as SecPipelineBuildAndDispatch
// but recorded inside BeginBatch/EndBatch. Single submit + fence at
// the end. Verifies the batched code path produces identical output.
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
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  LBufB := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
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
    Check(FCompute.DispatchComputeWithPush(
      LPipeline.Pipeline, LPipeline.PipelineLayout, LDescSet,
      @LPushCount, SizeOf(UInt32), LGroupsX),
      'DispatchComputeWithPush inside batch returns True');
    FCompute.BatchBarrier();  // no-op expected since only one dispatch
    FCompute.EndBatch();
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
// SecBatchMisuseRaises — verify that the two programmer-error paths
// in BeginBatch / EndBatch do actually raise. Both are in the
// approved raise list per TASK-REFACTOR §2.3.
// ---------------------------------------------------------------------------
procedure TComputeTest.SecBatchMisuseRaises();
var
  LLocal: TVdxCompute;
  LRaised: Boolean;
begin
  Section('Batch misuse raises');

  LLocal := TVdxCompute.Create();
  try
    Check(LLocal.Initialize(-1), 'Local compute Initialize(-1)');
    FlushErrors(LLocal.GetErrors());

    // EndBatch without a prior BeginBatch
    LRaised := False;
    try
      LLocal.EndBatch();
    except
      on E: Exception do
        LRaised := True;
    end;
    Check(LRaised, 'EndBatch without BeginBatch raises Exception');

    // BeginBatch twice
    LLocal.BeginBatch();
    try
      LRaised := False;
      try
        LLocal.BeginBatch();
      except
        on E: Exception do
          LRaised := True;
      end;
      Check(LRaised, 'Nested BeginBatch raises Exception');
    finally
      LLocal.EndBatch();
    end;

    FlushErrors(LLocal.GetErrors());
  finally
    LLocal.Free();
  end;
end;

end.
