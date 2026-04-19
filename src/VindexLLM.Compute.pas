{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Compute;

{$I VindexLLM.Defines.inc}

interface

uses
  WinAPI.Windows,
  System.SysUtils,
  VindexLLM.Utils,
  VindexLLM.Vulkan;

const
  // Error codes — user-facing messages live in VindexLLM.Resources.
  VDX_ERROR_VK_LIB_LOAD_FAILED   = 'VK01';
  VDX_ERROR_VK_PROC_MISSING      = 'VK02';
  VDX_ERROR_VK_NO_GPU            = 'VK03';
  VDX_ERROR_VK_NO_COMPUTE_QUEUE  = 'VK04';
  VDX_ERROR_VK_GPU_INDEX_INVALID = 'VK05';
  VDX_ERROR_VK_NO_MEMORY_TYPE    = 'VK06';
  VDX_ERROR_VK_CALL_FAILED       = 'VK07';

type

  { TVdxVRAMCategory }
  // Allocation intent tag stamped on every TVdxGpuBuffer at create
  // time. Compute uses it to bucket the running VRAM tally so higher
  // layers never maintain their own accounting — just pass the right
  // category at CreateGpuBuffer time. vcBuffer is the default for
  // generic scratch / intermediate buffers.
  TVdxVRAMCategory = (
    vcWeight,
    vcCache,
    vcBuffer
  );

  { TVdxVRAMUsage }
  // Snapshot of per-category totals (plus their sum). Produced by
  // GetVRAMUsage. Numbers reflect byte sizes requested via
  // CreateGpuBuffer; driver-internal overhead is not included.
  TVdxVRAMUsage = record
    WeightsBytes: UInt64;
    CacheBytes:   UInt64;
    BuffersBytes: UInt64;
    TotalBytes:   UInt64;
  end;

  { TVdxGpuBuffer }
  // Owning record for one GPU buffer allocation. Category is stamped
  // at create time and read back at destroy time so Compute can
  // decrement the correct running total. Size is the requested byte
  // count (not the possibly-rounded-up allocationSize returned by
  // vkGetBufferMemoryRequirements).
  TVdxGpuBuffer = record
    Buffer:   VkBuffer;
    Memory:   VkDeviceMemory;
    Size:     VkDeviceSize;
    Category: TVdxVRAMCategory;
  end;

  { TVdxComputePipelineBundle }
  // Groups a compute pipeline with its pipeline layout so
  // DestroyComputePipelineBundle can tear both down atomically.
  TVdxComputePipelineBundle = record
    Pipeline:       VkPipeline;
    PipelineLayout: VkPipelineLayout;
  end;

  { TVdxGpuDeviceType }
  // Mirrors VkPhysicalDeviceType (values 0..4) but as a strongly-
  // typed Delphi enum so callers don't have to include raw Vulkan
  // constants in their code.
  TVdxGpuDeviceType = (
    gdtOther,
    gdtIntegrated,
    gdtDiscrete,
    gdtVirtual,
    gdtCpu
  );

  { TVdxGpuInfo }
  // Rich descriptor for one physical GPU, produced by
  // TVdxCompute.EnumerateGpus. Fields are grouped into six
  // sections: identity, stable IDs, versions, memory, queue caps,
  // compute limits, subgroup, and shader features. Index is the
  // slot to pass back into TVdxCompute.Initialize(AGpuIndex).
  TVdxGpuInfo = record
    // Identity (display + matching)
    Index:      Integer;
    Name:       string;
    DeviceType: TVdxGpuDeviceType;
    VendorID:   UInt32;
    DeviceID:   UInt32;

    // Stable IDs (survive reboots / driver updates)
    DeviceUUID:      array[0..15] of Byte;
    DriverUUID:      array[0..15] of Byte;
    DeviceLUID:      array[0..7] of Byte;
    DeviceLUIDValid: Boolean;

    // Versions
    ApiVersion:    UInt32;
    ApiVersionStr: string;
    DriverVersion: UInt32;

    // Memory (MB)
    VRAMMB:         UInt64;
    SharedMemoryMB: UInt64;
    TotalMemoryMB:  UInt64;

    // Queue capabilities
    HasComputeQueue:          Boolean;
    HasDedicatedComputeQueue: Boolean;
    HasTransferQueue:         Boolean;

    // Compute limits
    MaxComputeWorkGroupInvocations:  UInt32;
    MaxComputeWorkGroupSize:         array[0..2] of UInt32;
    MaxComputeWorkGroupCount:        array[0..2] of UInt32;
    MaxComputeSharedMemorySize:      UInt32;
    MaxStorageBufferBytes:           UInt64;
    MaxPushConstantsSize:            UInt32;
    MinStorageBufferOffsetAlignment: UInt64;

    // Subgroup (wavefront / warp)
    SubgroupSize: UInt32;

    // Shader feature flags
    SupportsFp16:  Boolean;
    SupportsInt8:  Boolean;
    SupportsInt16: Boolean;
    SupportsInt64: Boolean;
    SupportsFp64:  Boolean;
  end;

  { TVdxCompute }
  // Vulkan compute device manager. One instance owns one logical
  // device on one physical GPU. Construction is cheap and
  // parameterless; real setup happens in Initialize, which boots
  // the full Vulkan stack and returns False with FErrors populated
  // on any step's failure.
  //
  // Multi-GPU usage: create one TVdxCompute per GPU, call
  // Initialize(AGpuIndex) on each. Instances are fully independent —
  // separate VkInstance / VkDevice / queue / command pool / fence /
  // VRAM accounting, no shared state.
  //
  // The two batch-state misuse sites (BeginBatch while already
  // batching, EndBatch without a prior BeginBatch) raise — those
  // are programmer errors with no sensible default.
  TVdxCompute = class(TVdxBaseObject)
  private
    // Library
    FLibHandle: THandle;

    // Core Vulkan objects
    FInstance:           VkInstance;
    FPhysicalDevice:     VkPhysicalDevice;
    FDevice:             VkDevice;
    FComputeQueue:       VkQueue;
    FComputeQueueFamily: UInt32;
    FCommandPool:        VkCommandPool;
    FCommandBuffer:      VkCommandBuffer;
    FFence:              VkFence;

    // Batch mode — record multiple dispatches into one submit + fence
    FBatchMode:              Boolean;
    FBatchDeferredPools:     array of VkDescriptorPool;
    FBatchDeferredPoolCount: Integer;

    // Shared streaming-staging pool — a single host-visible +
    // device-local buffer pair that consumers (TVdxAttention,
    // TVdxFFN, ...) borrow at dispatch time to stream tensor slices
    // from the mmap'd GGUF into VRAM. Grown on demand via
    // EnsureStagingCapacity; never shrinks. Lifetime is tied to
    // this TVdxCompute instance.
    FStagingHost:     TVdxGpuBuffer;
    FStagingDevice:   TVdxGpuBuffer;
    FStagingCapacity: UInt64;

    // Device info (populated once physical device is selected)
    FDeviceProperties: VkPhysicalDeviceProperties;
    FMemoryProperties: VkPhysicalDeviceMemoryProperties;
    FSelectedGpuIndex: Integer;

    // VRAM accounting — running totals per category
    FVRAMWeightsBytes: UInt64;
    FVRAMCacheBytes:   UInt64;
    FVRAMBuffersBytes: UInt64;

    // Enumeration cache — populated by first EnumerateGpus /
    // Initialize call, reused after that.
    FInstanceReady:         Boolean;
    FGpusEnumerated:        Boolean;
    FGpusCache:             TArray<TVdxGpuInfo>;
    FPhysicalDeviceHandles: TArray<VkPhysicalDevice>;

    // Function pointers — bootstrap
    FvkGetInstanceProcAddr: TvkGetInstanceProcAddr;

    // Function pointers — instance level (Vulkan 1.0)
    FvkCreateInstance:                         TvkCreateInstance;
    FvkDestroyInstance:                        TvkDestroyInstance;
    FvkEnumeratePhysicalDevices:               TvkEnumeratePhysicalDevices;
    FvkGetPhysicalDeviceProperties:            TvkGetPhysicalDeviceProperties;
    FvkGetPhysicalDeviceQueueFamilyProperties: TvkGetPhysicalDeviceQueueFamilyProperties;
    FvkGetPhysicalDeviceMemoryProperties:      TvkGetPhysicalDeviceMemoryProperties;

    // Function pointers — instance level (Vulkan 1.1 chained queries)
    // May be nil on 1.0-only drivers; EnumerateGpus falls back to
    // 1.0 properties in that case.
    FvkGetPhysicalDeviceProperties2: TvkGetPhysicalDeviceProperties2;
    FvkGetPhysicalDeviceFeatures2:   TvkGetPhysicalDeviceFeatures2;

    // Function pointers — device level
    FvkCreateDevice:                TvkCreateDevice;
    FvkDestroyDevice:               TvkDestroyDevice;
    FvkGetDeviceQueue:              TvkGetDeviceQueue;
    FvkCreateBuffer:                TvkCreateBuffer;
    FvkDestroyBuffer:               TvkDestroyBuffer;
    FvkGetBufferMemoryRequirements: TvkGetBufferMemoryRequirements;
    FvkAllocateMemory:              TvkAllocateMemory;
    FvkFreeMemory:                  TvkFreeMemory;
    FvkBindBufferMemory:            TvkBindBufferMemory;
    FvkMapMemory:                   TvkMapMemory;
    FvkUnmapMemory:                 TvkUnmapMemory;
    FvkFlushMappedMemoryRanges:     TvkFlushMappedMemoryRanges;
    FvkCreateShaderModule:          TvkCreateShaderModule;
    FvkDestroyShaderModule:         TvkDestroyShaderModule;
    FvkCreatePipelineLayout:        TvkCreatePipelineLayout;
    FvkDestroyPipelineLayout:       TvkDestroyPipelineLayout;
    FvkCreateComputePipelines:      TvkCreateComputePipelines;
    FvkDestroyPipeline:             TvkDestroyPipeline;
    FvkCreateDescriptorSetLayout:   TvkCreateDescriptorSetLayout;
    FvkDestroyDescriptorSetLayout:  TvkDestroyDescriptorSetLayout;
    FvkCreateDescriptorPool:        TvkCreateDescriptorPool;
    FvkDestroyDescriptorPool:       TvkDestroyDescriptorPool;
    FvkAllocateDescriptorSets:      TvkAllocateDescriptorSets;
    FvkUpdateDescriptorSets:        TvkUpdateDescriptorSets;
    FvkCreateCommandPool:           TvkCreateCommandPool;
    FvkDestroyCommandPool:          TvkDestroyCommandPool;
    FvkAllocateCommandBuffers:      TvkAllocateCommandBuffers;
    FvkBeginCommandBuffer:          TvkBeginCommandBuffer;
    FvkEndCommandBuffer:            TvkEndCommandBuffer;
    FvkCmdBindPipeline:             TvkCmdBindPipeline;
    FvkCmdBindDescriptorSets:       TvkCmdBindDescriptorSets;
    FvkCmdDispatch:                 TvkCmdDispatch;
    FvkQueueSubmit:                 TvkQueueSubmit;
    FvkQueueWaitIdle:               TvkQueueWaitIdle;
    FvkCreateFence:                 TvkCreateFence;
    FvkDestroyFence:                TvkDestroyFence;
    FvkWaitForFences:               TvkWaitForFences;
    FvkResetFences:                 TvkResetFences;
    FvkCmdCopyBuffer:               TvkCmdCopyBuffer;
    FvkCmdPushConstants:            TvkCmdPushConstants;
    FvkCmdPipelineBarrier:          TvkCmdPipelineBarrier;

    // Internal helpers
    function  GetVkProc(const AName: PAnsiChar): Pointer;
    function  CheckVk(const AResult: VkResult; const AContext: string): Boolean;
    function  EnsureInstance(): Boolean;
    function  LoadVulkanLibrary(): Boolean;
    function  LoadGlobalFunctions(): Boolean;
    function  CreateVkInstance(): Boolean;
    procedure LoadInstanceFunctions();
    function  PopulateGpuCache(): Boolean;
    procedure FillGpuInfoFromPhysicalDevice(
      const ADevice: VkPhysicalDevice;
      const AIndex: Integer;
      out AInfo: TVdxGpuInfo);
    function  DecodeApiVersion(const AVersion: UInt32): string;
    function  AutoSelectGpuIndex(out AIndex: Integer): Boolean;
    function  CreateLogicalDevice(): Boolean;
    procedure LoadDeviceFunctions();
    function  CreateCommandResources(): Boolean;
    function  FindMemoryType(const ATypeBits: UInt32;
      const AProperties: VkFlags; out AIndex: UInt32): Boolean;
    procedure InsertBatchBarrier();

  public
    constructor Create(); override;
    destructor  Destroy(); override;

    // Enumerate all Vulkan-capable GPUs on the system. Populates
    // AGpus with one TVdxGpuInfo per physical device (compute-capable
    // or not — caller filters on HasComputeQueue if needed). Safe to
    // call before Initialize; result is cached for reuse. Returns
    // False with FErrors populated if the Vulkan instance can't come
    // up (no driver installed, DLL missing, etc.).
    function EnumerateGpus(out AGpus: TArray<TVdxGpuInfo>): Boolean;

    // Full boot to a chosen GPU. AGpuIndex:
    //   -1 (default) — auto: prefer the first discrete GPU, fall
    //                  back to the first compute-capable device.
    //   >= 0         — explicit slot from EnumerateGpus. Fails with
    //                  VDX_ERROR_VK_GPU_INDEX_INVALID if out of
    //                  range or the selected device has no compute
    //                  queue.
    // Calls EnumerateGpus internally if it hasn't been called yet.
    function Initialize(const AGpuIndex: Integer = -1): Boolean;

    // Buffer primitives — CreateGpuBuffer returns a zeroed record
    // on failure; check HasFatal afterwards. Destroy is safe on a
    // zeroed / already-destroyed record.
    function  CreateGpuBuffer(const ASize: VkDeviceSize;
      const AUsage, AMemProps: VkFlags;
      const ACategory: TVdxVRAMCategory = vcBuffer): TVdxGpuBuffer;
    procedure DestroyGpuBuffer(var ABuffer: TVdxGpuBuffer);
    function  UploadToBuffer(const ABuffer: TVdxGpuBuffer;
      const AData: Pointer; const ASize: VkDeviceSize): Boolean;
    function  DownloadFromBuffer(const ABuffer: TVdxGpuBuffer;
      const AData: Pointer; const ASize: VkDeviceSize): Boolean;
    function  MapBufferPersistent(const ABuffer: TVdxGpuBuffer;
      out AData: Pointer): Boolean;
    procedure UnmapBuffer(const ABuffer: TVdxGpuBuffer);
    function  CopyBuffer(const ASrc, ADst: TVdxGpuBuffer;
      const ASize: VkDeviceSize): Boolean;
    function  CopyBufferRegion(const ASrc: TVdxGpuBuffer;
      const ASrcOffset: VkDeviceSize;
      const ADst: TVdxGpuBuffer;
      const ADstOffset, ASize: VkDeviceSize): Boolean;

    // Streaming-staging pool — one shared host+device buffer pair
    // used by consumers to stream tensor slices from the mmap'd
    // GGUF into VRAM. Consumers call EnsureStagingCapacity(MaxBytes)
    // during their own Init with the largest slice they'll ever
    // upload. The pool grows to max(current, MaxBytes) and never
    // shrinks. First call allocates; later calls that fit inside
    // the current capacity are no-ops. Returns False with FErrors
    // populated (via underlying CreateGpuBuffer) on failure.
    function  EnsureStagingCapacity(const ABytes: UInt64): Boolean;
    // Borrow handles for a streamed upload: fill StagingHost via
    // UploadToBuffer, CopyBuffer(StagingHost -> StagingDevice),
    // BatchBarrier, dispatch against StagingDevice. Both handles
    // are valid only after at least one successful
    // EnsureStagingCapacity call.
    function  GetStagingHost(): TVdxGpuBuffer;
    function  GetStagingDevice(): TVdxGpuBuffer;
    function  GetStagingCapacity(): UInt64;

    // Shader + pipeline
    function  CreateShaderModule(const ACode: Pointer;
      const ACodeSize: NativeUInt): VkShaderModule;
    procedure DestroyShaderModuleHandle(const AModule: VkShaderModule);
    function  CreateComputePipelineSimple(
      const AShaderModule: VkShaderModule;
      const AEntryPoint: PAnsiChar;
      const ADescSetLayout: VkDescriptorSetLayout): TVdxComputePipelineBundle;
    function  CreateComputePipelineWithPush(
      const AShaderModule: VkShaderModule;
      const AEntryPoint: PAnsiChar;
      const ADescSetLayout: VkDescriptorSetLayout;
      const APushSize: UInt32): TVdxComputePipelineBundle;
    function  CreateComputePipelineWithPushAndSpec(
      const AShaderModule: VkShaderModule;
      const AEntryPoint: PAnsiChar;
      const ADescSetLayout: VkDescriptorSetLayout;
      const APushSize, ASpecValue: UInt32): TVdxComputePipelineBundle;
    procedure DestroyComputePipelineBundle(var ABundle: TVdxComputePipelineBundle);

    // Descriptor sets
    function  CreateStorageDescriptorSetLayout(
      const ABindingCount: UInt32): VkDescriptorSetLayout;
    function  CreateDescriptorPoolForStorage(
      const AMaxSets, AMaxDescriptors: UInt32): VkDescriptorPool;
    function  AllocateDescriptorSetForBuffers(
      const APool: VkDescriptorPool;
      const ALayout: VkDescriptorSetLayout;
      const ABuffers: array of TVdxGpuBuffer): VkDescriptorSet;
    procedure UpdateDescriptorSetBuffers(const ADescSet: VkDescriptorSet;
      const ABuffers: array of TVdxGpuBuffer);
    procedure DestroyDescriptorSetLayoutHandle(const ALayout: VkDescriptorSetLayout);
    procedure DestroyDescriptorPoolHandle(const APool: VkDescriptorPool);

    // Dispatch — return False on any VkResult failure (non-batch
    // mode only; batch-mode dispatches never submit, so they return
    // True once the commands are recorded).
    function DispatchCompute(const APipeline: VkPipeline;
      const APipelineLayout: VkPipelineLayout;
      const ADescSet: VkDescriptorSet;
      const AGroupsX: UInt32;
      const AGroupsY: UInt32 = 1;
      const AGroupsZ: UInt32 = 1): Boolean;
    function DispatchComputeWithPush(const APipeline: VkPipeline;
      const APipelineLayout: VkPipelineLayout;
      const ADescSet: VkDescriptorSet;
      const APushData: Pointer; const APushSize: UInt32;
      const AGroupsX: UInt32;
      const AGroupsY: UInt32 = 1;
      const AGroupsZ: UInt32 = 1): Boolean;

    // Batch mode — record multiple dispatches into a single submit+fence.
    // BeginBatch raises if already batching; EndBatch raises if not
    // currently batching. Both are programmer errors.
    procedure BeginBatch();
    procedure EndBatch();
    procedure BatchBarrier();

    // Queries
    function GetDeviceName(): string;
    function GetVRAMSizeMB(): UInt64;
    function GetMaxComputeWorkGroupSize(): UInt32;
    function GetVRAMUsage(): TVdxVRAMUsage;
    function GetSelectedGpuIndex(): Integer;
  end;

implementation

uses
  VindexLLM.Resources;

// ============================================================================
//  TVdxCompute — Construction / Destruction
// ============================================================================

constructor TVdxCompute.Create();
begin
  inherited;
  FLibHandle          := 0;
  FInstance           := nil;
  FPhysicalDevice     := nil;
  FDevice             := nil;
  FComputeQueue       := nil;
  FComputeQueueFamily := 0;
  FCommandPool        := VK_NULL_HANDLE;
  FCommandBuffer      := nil;
  FFence              := VK_NULL_HANDLE;

  FBatchMode              := False;
  FBatchDeferredPoolCount := 0;

  FStagingHost     := Default(TVdxGpuBuffer);
  FStagingDevice   := Default(TVdxGpuBuffer);
  FStagingCapacity := 0;

  FSelectedGpuIndex := -1;

  FVRAMWeightsBytes := 0;
  FVRAMCacheBytes   := 0;
  FVRAMBuffersBytes := 0;

  FInstanceReady  := False;
  FGpusEnumerated := False;
end;

destructor TVdxCompute.Destroy();
begin
  if FDevice <> nil then
  begin
    if Assigned(FvkQueueWaitIdle) then
      FvkQueueWaitIdle(FComputeQueue);

    // Free streaming-staging pool while the device is still alive.
    if FStagingCapacity > 0 then
    begin
      DestroyGpuBuffer(FStagingHost);
      DestroyGpuBuffer(FStagingDevice);
      FStagingCapacity := 0;
    end;

    if FFence <> VK_NULL_HANDLE then
      FvkDestroyFence(FDevice, FFence, nil);

    if FCommandPool <> VK_NULL_HANDLE then
      FvkDestroyCommandPool(FDevice, FCommandPool, nil);

    FvkDestroyDevice(FDevice, nil);
  end;

  if FInstance <> nil then
    FvkDestroyInstance(FInstance, nil);

  if FLibHandle <> 0 then
    FreeLibrary(FLibHandle);

  inherited;
end;

// ============================================================================
//  Internal helpers — library / function loading, CheckVk, proc lookup
// ============================================================================

function TVdxCompute.GetVkProc(const AName: PAnsiChar): Pointer;
begin
  // Try instance-level first (most procs resolve this way), then
  // fall back to GetProcAddress on the DLL for the bootstrap entry
  // points. Nil is returned if neither path finds it.
  if FInstance <> nil then
    Result := FvkGetInstanceProcAddr(FInstance, AName)
  else
    Result := FvkGetInstanceProcAddr(nil, AName);

  if Result = nil then
    Result := GetProcAddress(FLibHandle, AName);
end;

function TVdxCompute.CheckVk(const AResult: VkResult; const AContext: string): Boolean;
begin
  Result := AResult = VK_SUCCESS;
  if not Result then
    FErrors.Add(esFatal, VDX_ERROR_VK_CALL_FAILED,
      RSVkCallFailed, [AContext, Integer(AResult)]);
end;

function TVdxCompute.LoadVulkanLibrary(): Boolean;
begin
  Result := False;

  FLibHandle := LoadLibrary('vulkan-1.dll');
  if FLibHandle = 0 then
  begin
    FErrors.Add(esFatal, VDX_ERROR_VK_LIB_LOAD_FAILED, RSVkLibLoadFailed);
    Exit;
  end;

  FvkGetInstanceProcAddr := TvkGetInstanceProcAddr(
    GetProcAddress(FLibHandle, 'vkGetInstanceProcAddr'));
  if not Assigned(FvkGetInstanceProcAddr) then
  begin
    FErrors.Add(esFatal, VDX_ERROR_VK_PROC_MISSING,
      RSVkProcMissing, ['vkGetInstanceProcAddr']);
    Exit;
  end;

  Result := True;
end;

function TVdxCompute.LoadGlobalFunctions(): Boolean;
begin
  Result := False;

  @FvkCreateInstance := GetVkProc('vkCreateInstance');
  if not Assigned(FvkCreateInstance) then
  begin
    FErrors.Add(esFatal, VDX_ERROR_VK_PROC_MISSING,
      RSVkProcMissing, ['vkCreateInstance']);
    Exit;
  end;

  Result := True;
end;

function TVdxCompute.CreateVkInstance(): Boolean;
var
  LAppInfo: VkApplicationInfo;
  LCreateInfo: VkInstanceCreateInfo;
begin
  Result := False;

  FillChar(LAppInfo, SizeOf(LAppInfo), 0);
  LAppInfo.sType              := VK_STRUCTURE_TYPE_APPLICATION_INFO;
  LAppInfo.pApplicationName   := 'VindexLLM';
  LAppInfo.applicationVersion := 1;
  LAppInfo.pEngineName        := 'VindexLLM';
  LAppInfo.engineVersion      := 1;
  // Request 1.1 so vkGetPhysicalDeviceProperties2 / Features2 are
  // usable for enumeration. On older 1.0 drivers this call will
  // still succeed (instance version is the maximum the driver
  // supports), but the 1.1 entry points will resolve to nil —
  // EnumerateGpus detects and falls back gracefully.
  LAppInfo.apiVersion := VK_API_VERSION_1_1;

  FillChar(LCreateInfo, SizeOf(LCreateInfo), 0);
  LCreateInfo.sType            := VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  LCreateInfo.pApplicationInfo := @LAppInfo;

  if not CheckVk(FvkCreateInstance(LCreateInfo, nil, FInstance),
    'Creating Vulkan instance (vkCreateInstance)') then
    Exit;

  Result := True;
end;

procedure TVdxCompute.LoadInstanceFunctions();
begin
  @FvkDestroyInstance                        := GetVkProc('vkDestroyInstance');
  @FvkEnumeratePhysicalDevices               := GetVkProc('vkEnumeratePhysicalDevices');
  @FvkGetPhysicalDeviceProperties            := GetVkProc('vkGetPhysicalDeviceProperties');
  @FvkGetPhysicalDeviceQueueFamilyProperties := GetVkProc('vkGetPhysicalDeviceQueueFamilyProperties');
  @FvkGetPhysicalDeviceMemoryProperties      := GetVkProc('vkGetPhysicalDeviceMemoryProperties');
  @FvkCreateDevice                           := GetVkProc('vkCreateDevice');

  // Vulkan 1.1 chained-query entry points. Resolve via GetVkProc;
  // may come back nil on 1.0-only drivers. Callers guard with
  // Assigned() before use.
  @FvkGetPhysicalDeviceProperties2 := GetVkProc('vkGetPhysicalDeviceProperties2');
  @FvkGetPhysicalDeviceFeatures2   := GetVkProc('vkGetPhysicalDeviceFeatures2');
end;

function TVdxCompute.EnsureInstance(): Boolean;
begin
  if FInstanceReady then
    Exit(True);

  if not LoadVulkanLibrary() then Exit(False);
  if not LoadGlobalFunctions() then Exit(False);
  if not CreateVkInstance() then Exit(False);
  LoadInstanceFunctions();

  FInstanceReady := True;
  Result := True;
end;

// ============================================================================
//  GPU enumeration
// ============================================================================

function TVdxCompute.DecodeApiVersion(const AVersion: UInt32): string;
begin
  // Vulkan version encoding: bits 22-31 major, 12-21 minor, 0-11 patch
  Result := Format('%d.%d.%d',
    [(AVersion shr 22) and $3FF,
     (AVersion shr 12) and $3FF,
     AVersion and $FFF]);
end;

procedure TVdxCompute.FillGpuInfoFromPhysicalDevice(
  const ADevice: VkPhysicalDevice;
  const AIndex: Integer;
  out AInfo: TVdxGpuInfo);
var
  LProps:     VkPhysicalDeviceProperties;
  LMemProps:  VkPhysicalDeviceMemoryProperties;
  LFamCount:  UInt32;
  LFamilies:  array of VkQueueFamilyProperties;
  LProps2:    VkPhysicalDeviceProperties2;
  LFeatures2: VkPhysicalDeviceFeatures2;
  LIdProps:   VkPhysicalDeviceIDProperties;
  LSubgroup:  VkPhysicalDeviceSubgroupProperties;
  LFp16Int8:  VkPhysicalDeviceShaderFloat16Int8Features;
  LI, LJ:     Integer;
  LTotalMem:  UInt64;
  LLargestDev, LLargestShared: UInt64;
  LHostVis:   Boolean;
  LHasComp, LHasGfx, LHasXfer: Boolean;
begin
  AInfo := Default(TVdxGpuInfo);
  AInfo.Index := AIndex;

  // --- Base properties (Vulkan 1.0) ---
  FvkGetPhysicalDeviceProperties(ADevice, LProps);
  AInfo.Name := string(AnsiString(PAnsiChar(@LProps.deviceName[0])));
  case LProps.deviceType of
    0: AInfo.DeviceType := gdtOther;
    1: AInfo.DeviceType := gdtIntegrated;
    2: AInfo.DeviceType := gdtDiscrete;
    3: AInfo.DeviceType := gdtVirtual;
    4: AInfo.DeviceType := gdtCpu;
  else
    AInfo.DeviceType := gdtOther;
  end;
  AInfo.VendorID      := LProps.vendorID;
  AInfo.DeviceID      := LProps.deviceID;
  AInfo.ApiVersion    := LProps.apiVersion;
  AInfo.ApiVersionStr := DecodeApiVersion(LProps.apiVersion);
  AInfo.DriverVersion := LProps.driverVersion;

  // Compute limits
  AInfo.MaxComputeWorkGroupInvocations     := LProps.limits.maxComputeWorkGroupInvocations;
  AInfo.MaxComputeWorkGroupSize[0]         := LProps.limits.maxComputeWorkGroupSize[0];
  AInfo.MaxComputeWorkGroupSize[1]         := LProps.limits.maxComputeWorkGroupSize[1];
  AInfo.MaxComputeWorkGroupSize[2]         := LProps.limits.maxComputeWorkGroupSize[2];
  AInfo.MaxComputeWorkGroupCount[0]        := LProps.limits.maxComputeWorkGroupCount[0];
  AInfo.MaxComputeWorkGroupCount[1]        := LProps.limits.maxComputeWorkGroupCount[1];
  AInfo.MaxComputeWorkGroupCount[2]        := LProps.limits.maxComputeWorkGroupCount[2];
  AInfo.MaxComputeSharedMemorySize         := LProps.limits.maxComputeSharedMemorySize;
  AInfo.MaxStorageBufferBytes              := LProps.limits.maxStorageBufferRange;
  AInfo.MaxPushConstantsSize               := LProps.limits.maxPushConstantsSize;
  AInfo.MinStorageBufferOffsetAlignment    := LProps.limits.minStorageBufferOffsetAlignment;

  // --- Memory properties ---
  FvkGetPhysicalDeviceMemoryProperties(ADevice, LMemProps);
  LLargestDev    := 0;
  LLargestShared := 0;
  LTotalMem      := 0;
  for LI := 0 to Integer(LMemProps.memoryHeapCount) - 1 do
  begin
    LTotalMem := LTotalMem + LMemProps.memoryHeaps[LI].size;
    if (LMemProps.memoryHeaps[LI].flags and $00000001) <> 0 then
    begin
      // Device-local heap
      if LMemProps.memoryHeaps[LI].size > LLargestDev then
        LLargestDev := LMemProps.memoryHeaps[LI].size;
    end
    else
    begin
      // Host-visible (system) heap — any heap without the
      // device-local flag
      if LMemProps.memoryHeaps[LI].size > LLargestShared then
        LLargestShared := LMemProps.memoryHeaps[LI].size;
    end;
  end;
  // Fallback: if every heap is flagged device-local (common on
  // integrated GPUs and APUs), also surface the largest as shared
  // so callers don't see zero.
  LHostVis := False;
  for LI := 0 to Integer(LMemProps.memoryTypeCount) - 1 do
    if (LMemProps.memoryTypes[LI].propertyFlags and VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) <> 0 then
    begin
      LHostVis := True;
      Break;
    end;
  if (LLargestShared = 0) and LHostVis then
    LLargestShared := LLargestDev;

  AInfo.VRAMMB         := LLargestDev    div (1024 * 1024);
  AInfo.SharedMemoryMB := LLargestShared div (1024 * 1024);
  AInfo.TotalMemoryMB  := LTotalMem      div (1024 * 1024);

  // --- Queue family caps ---
  LFamCount := 0;
  FvkGetPhysicalDeviceQueueFamilyProperties(ADevice, LFamCount, nil);
  if LFamCount > 0 then
  begin
    SetLength(LFamilies, LFamCount);
    FvkGetPhysicalDeviceQueueFamilyProperties(ADevice, LFamCount, @LFamilies[0]);
  end;

  for LJ := 0 to Integer(LFamCount) - 1 do
  begin
    LHasComp := (LFamilies[LJ].queueFlags and VK_QUEUE_COMPUTE_BIT) <> 0;
    LHasGfx  := (LFamilies[LJ].queueFlags and VK_QUEUE_GRAPHICS_BIT) <> 0;
    LHasXfer := (LFamilies[LJ].queueFlags and VK_QUEUE_TRANSFER_BIT) <> 0;

    if LHasComp then
    begin
      AInfo.HasComputeQueue := True;
      if not LHasGfx then
        AInfo.HasDedicatedComputeQueue := True;
    end;
    // Note: compute-capable and graphics-capable queues implicitly
    // support transfer per Vulkan spec, so treat those as having a
    // transfer queue too.
    if LHasXfer or LHasComp or LHasGfx then
      AInfo.HasTransferQueue := True;
  end;

  // --- Chained 1.1 queries (if available) ---
  // Base feature struct defaults — populated below only if the 1.1
  // entry points are available.
  AInfo.SubgroupSize := 0;

  if Assigned(FvkGetPhysicalDeviceProperties2) then
  begin
    LIdProps := Default(VkPhysicalDeviceIDProperties);
    LIdProps.sType := VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

    LSubgroup := Default(VkPhysicalDeviceSubgroupProperties);
    LSubgroup.sType := VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    LSubgroup.pNext := @LIdProps;

    LProps2 := Default(VkPhysicalDeviceProperties2);
    LProps2.sType := VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    LProps2.pNext := @LSubgroup;

    FvkGetPhysicalDeviceProperties2(ADevice, LProps2);

    Move(LIdProps.deviceUUID[0], AInfo.DeviceUUID[0], VK_UUID_SIZE);
    Move(LIdProps.driverUUID[0], AInfo.DriverUUID[0], VK_UUID_SIZE);
    Move(LIdProps.deviceLUID[0], AInfo.DeviceLUID[0], VK_LUID_SIZE);
    AInfo.DeviceLUIDValid := LIdProps.deviceLUIDValid <> 0;

    AInfo.SubgroupSize := LSubgroup.subgroupSize;
  end;

  if Assigned(FvkGetPhysicalDeviceFeatures2) then
  begin
    LFp16Int8 := Default(VkPhysicalDeviceShaderFloat16Int8Features);
    LFp16Int8.sType := VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;

    LFeatures2 := Default(VkPhysicalDeviceFeatures2);
    LFeatures2.sType := VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    LFeatures2.pNext := @LFp16Int8;

    FvkGetPhysicalDeviceFeatures2(ADevice, LFeatures2);

    AInfo.SupportsFp64  := LFeatures2.features.shaderFloat64 <> 0;
    AInfo.SupportsInt64 := LFeatures2.features.shaderInt64   <> 0;
    AInfo.SupportsInt16 := LFeatures2.features.shaderInt16   <> 0;
    AInfo.SupportsFp16  := LFp16Int8.shaderFloat16           <> 0;
    AInfo.SupportsInt8  := LFp16Int8.shaderInt8              <> 0;
  end;
end;

function TVdxCompute.PopulateGpuCache(): Boolean;
var
  LCount: UInt32;
  LDevices: array of VkPhysicalDevice;
  LI: Integer;
begin
  Result := False;

  LCount := 0;
  if not CheckVk(FvkEnumeratePhysicalDevices(FInstance, LCount, nil),
    'Enumerating GPUs (vkEnumeratePhysicalDevices count)') then
    Exit;

  if LCount = 0 then
  begin
    FErrors.Add(esFatal, VDX_ERROR_VK_NO_GPU, RSVkNoGpu);
    Exit;
  end;

  SetLength(LDevices, LCount);
  if not CheckVk(FvkEnumeratePhysicalDevices(FInstance, LCount, @LDevices[0]),
    'Enumerating GPUs (vkEnumeratePhysicalDevices)') then
    Exit;

  SetLength(FPhysicalDeviceHandles, LCount);
  SetLength(FGpusCache, LCount);
  for LI := 0 to Integer(LCount) - 1 do
  begin
    FPhysicalDeviceHandles[LI] := LDevices[LI];
    FillGpuInfoFromPhysicalDevice(LDevices[LI], LI, FGpusCache[LI]);
  end;

  FGpusEnumerated := True;
  Result := True;
end;

function TVdxCompute.EnumerateGpus(out AGpus: TArray<TVdxGpuInfo>): Boolean;
begin
  SetLength(AGpus, 0);
  Result := False;

  if not EnsureInstance() then Exit;

  if not FGpusEnumerated then
    if not PopulateGpuCache() then Exit;

  AGpus := Copy(FGpusCache);
  Result := True;
end;

// ============================================================================
//  Initialization — boot logical device on the chosen GPU
// ============================================================================

function TVdxCompute.AutoSelectGpuIndex(out AIndex: Integer): Boolean;
var
  LI: Integer;
  LFirstCompute: Integer;
begin
  // First pass: prefer a discrete GPU with a compute queue.
  AIndex := -1;
  LFirstCompute := -1;
  for LI := 0 to High(FGpusCache) do
  begin
    if not FGpusCache[LI].HasComputeQueue then
      Continue;
    if LFirstCompute < 0 then
      LFirstCompute := LI;
    if FGpusCache[LI].DeviceType = gdtDiscrete then
    begin
      AIndex := LI;
      Exit(True);
    end;
  end;

  // Second pass: any compute-capable GPU.
  if LFirstCompute >= 0 then
  begin
    AIndex := LFirstCompute;
    Exit(True);
  end;

  // No compute-capable devices at all.
  FErrors.Add(esFatal, VDX_ERROR_VK_NO_COMPUTE_QUEUE, RSVkNoComputeQueue);
  Result := False;
end;

function TVdxCompute.CreateLogicalDevice(): Boolean;
var
  LQueuePriority: Single;
  LQueueCreateInfo: VkDeviceQueueCreateInfo;
  LDeviceCreateInfo: VkDeviceCreateInfo;
begin
  Result := False;

  LQueuePriority := 1.0;

  FillChar(LQueueCreateInfo, SizeOf(LQueueCreateInfo), 0);
  LQueueCreateInfo.sType            := VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  LQueueCreateInfo.queueFamilyIndex := FComputeQueueFamily;
  LQueueCreateInfo.queueCount       := 1;
  LQueueCreateInfo.pQueuePriorities := @LQueuePriority;

  FillChar(LDeviceCreateInfo, SizeOf(LDeviceCreateInfo), 0);
  LDeviceCreateInfo.sType                := VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  LDeviceCreateInfo.queueCreateInfoCount := 1;
  LDeviceCreateInfo.pQueueCreateInfos    := @LQueueCreateInfo;

  if not CheckVk(FvkCreateDevice(FPhysicalDevice, LDeviceCreateInfo, nil, FDevice),
    'Creating logical device (vkCreateDevice)') then
    Exit;

  Result := True;
end;

procedure TVdxCompute.LoadDeviceFunctions();
begin
  @FvkDestroyDevice               := GetVkProc('vkDestroyDevice');
  @FvkGetDeviceQueue              := GetVkProc('vkGetDeviceQueue');
  @FvkCreateBuffer                := GetVkProc('vkCreateBuffer');
  @FvkDestroyBuffer               := GetVkProc('vkDestroyBuffer');
  @FvkGetBufferMemoryRequirements := GetVkProc('vkGetBufferMemoryRequirements');
  @FvkAllocateMemory              := GetVkProc('vkAllocateMemory');
  @FvkFreeMemory                  := GetVkProc('vkFreeMemory');
  @FvkBindBufferMemory            := GetVkProc('vkBindBufferMemory');
  @FvkMapMemory                   := GetVkProc('vkMapMemory');
  @FvkUnmapMemory                 := GetVkProc('vkUnmapMemory');
  @FvkFlushMappedMemoryRanges     := GetVkProc('vkFlushMappedMemoryRanges');
  @FvkCreateShaderModule          := GetVkProc('vkCreateShaderModule');
  @FvkDestroyShaderModule         := GetVkProc('vkDestroyShaderModule');
  @FvkCreatePipelineLayout        := GetVkProc('vkCreatePipelineLayout');
  @FvkDestroyPipelineLayout       := GetVkProc('vkDestroyPipelineLayout');
  @FvkCreateComputePipelines      := GetVkProc('vkCreateComputePipelines');
  @FvkDestroyPipeline             := GetVkProc('vkDestroyPipeline');
  @FvkCreateDescriptorSetLayout   := GetVkProc('vkCreateDescriptorSetLayout');
  @FvkDestroyDescriptorSetLayout  := GetVkProc('vkDestroyDescriptorSetLayout');
  @FvkCreateDescriptorPool        := GetVkProc('vkCreateDescriptorPool');
  @FvkDestroyDescriptorPool       := GetVkProc('vkDestroyDescriptorPool');
  @FvkAllocateDescriptorSets      := GetVkProc('vkAllocateDescriptorSets');
  @FvkUpdateDescriptorSets        := GetVkProc('vkUpdateDescriptorSets');
  @FvkCreateCommandPool           := GetVkProc('vkCreateCommandPool');
  @FvkDestroyCommandPool          := GetVkProc('vkDestroyCommandPool');
  @FvkAllocateCommandBuffers      := GetVkProc('vkAllocateCommandBuffers');
  @FvkBeginCommandBuffer          := GetVkProc('vkBeginCommandBuffer');
  @FvkEndCommandBuffer            := GetVkProc('vkEndCommandBuffer');
  @FvkCmdBindPipeline             := GetVkProc('vkCmdBindPipeline');
  @FvkCmdBindDescriptorSets       := GetVkProc('vkCmdBindDescriptorSets');
  @FvkCmdDispatch                 := GetVkProc('vkCmdDispatch');
  @FvkQueueSubmit                 := GetVkProc('vkQueueSubmit');
  @FvkQueueWaitIdle               := GetVkProc('vkQueueWaitIdle');
  @FvkCreateFence                 := GetVkProc('vkCreateFence');
  @FvkDestroyFence                := GetVkProc('vkDestroyFence');
  @FvkWaitForFences               := GetVkProc('vkWaitForFences');
  @FvkResetFences                 := GetVkProc('vkResetFences');
  @FvkCmdCopyBuffer               := GetVkProc('vkCmdCopyBuffer');
  @FvkCmdPushConstants            := GetVkProc('vkCmdPushConstants');
  @FvkCmdPipelineBarrier          := GetVkProc('vkCmdPipelineBarrier');
end;

function TVdxCompute.CreateCommandResources(): Boolean;
var
  LPoolInfo: VkCommandPoolCreateInfo;
  LAllocInfo: VkCommandBufferAllocateInfo;
  LFenceInfo: VkFenceCreateInfo;
begin
  Result := False;

  // Command pool
  FillChar(LPoolInfo, SizeOf(LPoolInfo), 0);
  LPoolInfo.sType            := VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  LPoolInfo.flags            := VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  LPoolInfo.queueFamilyIndex := FComputeQueueFamily;
  if not CheckVk(FvkCreateCommandPool(FDevice, LPoolInfo, nil, FCommandPool),
    'Creating command pool (vkCreateCommandPool)') then
    Exit;

  // Command buffer
  FillChar(LAllocInfo, SizeOf(LAllocInfo), 0);
  LAllocInfo.sType              := VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  LAllocInfo.commandPool        := FCommandPool;
  LAllocInfo.level              := VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  LAllocInfo.commandBufferCount := 1;
  if not CheckVk(FvkAllocateCommandBuffers(FDevice, LAllocInfo, @FCommandBuffer),
    'Allocating command buffer (vkAllocateCommandBuffers)') then
    Exit;

  // Fence for synchronization
  FillChar(LFenceInfo, SizeOf(LFenceInfo), 0);
  LFenceInfo.sType := VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if not CheckVk(FvkCreateFence(FDevice, LFenceInfo, nil, FFence),
    'Creating fence (vkCreateFence)') then
    Exit;

  Result := True;
end;

function TVdxCompute.FindMemoryType(const ATypeBits: UInt32;
  const AProperties: VkFlags; out AIndex: UInt32): Boolean;
var
  LI: UInt32;
begin
  AIndex := 0;
  for LI := 0 to FMemoryProperties.memoryTypeCount - 1 do
  begin
    if ((ATypeBits and (1 shl LI)) <> 0) and
       ((FMemoryProperties.memoryTypes[LI].propertyFlags and AProperties) = AProperties) then
    begin
      AIndex := LI;
      Exit(True);
    end;
  end;

  FErrors.Add(esFatal, VDX_ERROR_VK_NO_MEMORY_TYPE,
    RSVkNoMemoryType, [ATypeBits, AProperties]);
  Result := False;
end;

function TVdxCompute.Initialize(const AGpuIndex: Integer = -1): Boolean;
var
  LTargetIndex: Integer;
  LFamCount: UInt32;
  LFamilies: array of VkQueueFamilyProperties;
  LJ: Integer;
  LQueueFamily: Integer;
begin
  Result := False;

  Status('Loading Vulkan...');
  if not EnsureInstance() then Exit;

  if not FGpusEnumerated then
    if not PopulateGpuCache() then Exit;

  // --- Select GPU ---
  if AGpuIndex = -1 then
  begin
    Status('Auto-selecting GPU (preferring discrete)...');
    if not AutoSelectGpuIndex(LTargetIndex) then Exit;
  end
  else
  begin
    if (AGpuIndex < 0) or (AGpuIndex >= Length(FGpusCache)) then
    begin
      FErrors.Add(esFatal, VDX_ERROR_VK_GPU_INDEX_INVALID,
        RSVkGpuIndexInvalid, [AGpuIndex, Length(FGpusCache)]);
      Exit;
    end;
    if not FGpusCache[AGpuIndex].HasComputeQueue then
    begin
      FErrors.Add(esFatal, VDX_ERROR_VK_NO_COMPUTE_QUEUE, RSVkNoComputeQueue);
      Exit;
    end;
    LTargetIndex := AGpuIndex;
  end;

  FSelectedGpuIndex := LTargetIndex;
  FPhysicalDevice := FPhysicalDeviceHandles[LTargetIndex];
  FvkGetPhysicalDeviceProperties(FPhysicalDevice, FDeviceProperties);
  FvkGetPhysicalDeviceMemoryProperties(FPhysicalDevice, FMemoryProperties);

  // Find compute queue family on the selected device (the first
  // compute-capable family is fine — we only ever use one).
  LFamCount := 0;
  FvkGetPhysicalDeviceQueueFamilyProperties(FPhysicalDevice, LFamCount, nil);
  SetLength(LFamilies, LFamCount);
  FvkGetPhysicalDeviceQueueFamilyProperties(FPhysicalDevice, LFamCount, @LFamilies[0]);

  LQueueFamily := -1;
  for LJ := 0 to Integer(LFamCount) - 1 do
    if (LFamilies[LJ].queueFlags and VK_QUEUE_COMPUTE_BIT) <> 0 then
    begin
      LQueueFamily := LJ;
      Break;
    end;
  if LQueueFamily < 0 then
  begin
    // Shouldn't happen — HasComputeQueue was already True — but
    // guard anyway so we never crash later.
    FErrors.Add(esFatal, VDX_ERROR_VK_NO_COMPUTE_QUEUE, RSVkNoComputeQueue);
    Exit;
  end;
  FComputeQueueFamily := UInt32(LQueueFamily);

  Status('Selected GPU %d: %s', [LTargetIndex, FGpusCache[LTargetIndex].Name]);

  Status('Creating logical device...');
  if not CreateLogicalDevice() then Exit;
  LoadDeviceFunctions();
  FvkGetDeviceQueue(FDevice, FComputeQueueFamily, 0, FComputeQueue);

  Status('Creating command resources...');
  if not CreateCommandResources() then Exit;

  Status('Vulkan ready: %s (%d MB VRAM)',
    [GetDeviceName(), GetVRAMSizeMB()]);

  Result := True;
end;

// ============================================================================
//  Buffer Operations
// ============================================================================

function TVdxCompute.CreateGpuBuffer(const ASize: VkDeviceSize;
  const AUsage, AMemProps: VkFlags;
  const ACategory: TVdxVRAMCategory): TVdxGpuBuffer;
var
  LBufInfo: VkBufferCreateInfo;
  LMemReqs: VkMemoryRequirements;
  LAllocInfo: VkMemoryAllocateInfo;
  LMemTypeIndex: UInt32;
begin
  Result := Default(TVdxGpuBuffer);
  Result.Size     := ASize;
  Result.Category := ACategory;

  FillChar(LBufInfo, SizeOf(LBufInfo), 0);
  LBufInfo.sType       := VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  LBufInfo.size        := ASize;
  LBufInfo.usage       := AUsage;
  LBufInfo.sharingMode := VK_SHARING_MODE_EXCLUSIVE;
  if not CheckVk(FvkCreateBuffer(FDevice, LBufInfo, nil, Result.Buffer),
    'Creating GPU buffer (vkCreateBuffer)') then
  begin
    Result := Default(TVdxGpuBuffer);
    Exit;
  end;

  FvkGetBufferMemoryRequirements(FDevice, Result.Buffer, LMemReqs);

  if not FindMemoryType(LMemReqs.memoryTypeBits, AMemProps, LMemTypeIndex) then
  begin
    FvkDestroyBuffer(FDevice, Result.Buffer, nil);
    Result := Default(TVdxGpuBuffer);
    Exit;
  end;

  FillChar(LAllocInfo, SizeOf(LAllocInfo), 0);
  LAllocInfo.sType           := VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  LAllocInfo.allocationSize  := LMemReqs.size;
  LAllocInfo.memoryTypeIndex := LMemTypeIndex;
  if not CheckVk(FvkAllocateMemory(FDevice, LAllocInfo, nil, Result.Memory),
    'Allocating GPU memory (vkAllocateMemory)') then
  begin
    FvkDestroyBuffer(FDevice, Result.Buffer, nil);
    Result := Default(TVdxGpuBuffer);
    Exit;
  end;

  if not CheckVk(FvkBindBufferMemory(FDevice, Result.Buffer, Result.Memory, 0),
    'Binding buffer memory (vkBindBufferMemory)') then
  begin
    FvkFreeMemory(FDevice, Result.Memory, nil);
    FvkDestroyBuffer(FDevice, Result.Buffer, nil);
    Result := Default(TVdxGpuBuffer);
    Exit;
  end;

  // Running VRAM accounting — bump the bucket for this category.
  case ACategory of
    vcWeight: FVRAMWeightsBytes := FVRAMWeightsBytes + UInt64(ASize);
    vcCache:  FVRAMCacheBytes   := FVRAMCacheBytes   + UInt64(ASize);
    vcBuffer: FVRAMBuffersBytes := FVRAMBuffersBytes + UInt64(ASize);
  end;
end;

procedure TVdxCompute.DestroyGpuBuffer(var ABuffer: TVdxGpuBuffer);
begin
  if ABuffer.Buffer <> VK_NULL_HANDLE then
    FvkDestroyBuffer(FDevice, ABuffer.Buffer, nil);

  if ABuffer.Memory <> VK_NULL_HANDLE then
    FvkFreeMemory(FDevice, ABuffer.Memory, nil);

  // Decrement the category bucket — Destroy on a zeroed record is
  // harmless because ABuffer.Size is 0 so no underflow occurs.
  case ABuffer.Category of
    vcWeight:
      if FVRAMWeightsBytes >= UInt64(ABuffer.Size) then
        FVRAMWeightsBytes := FVRAMWeightsBytes - UInt64(ABuffer.Size)
      else
        FVRAMWeightsBytes := 0;
    vcCache:
      if FVRAMCacheBytes >= UInt64(ABuffer.Size) then
        FVRAMCacheBytes := FVRAMCacheBytes - UInt64(ABuffer.Size)
      else
        FVRAMCacheBytes := 0;
    vcBuffer:
      if FVRAMBuffersBytes >= UInt64(ABuffer.Size) then
        FVRAMBuffersBytes := FVRAMBuffersBytes - UInt64(ABuffer.Size)
      else
        FVRAMBuffersBytes := 0;
  end;

  FillChar(ABuffer, SizeOf(ABuffer), 0);
end;

function TVdxCompute.UploadToBuffer(const ABuffer: TVdxGpuBuffer;
  const AData: Pointer; const ASize: VkDeviceSize): Boolean;
var
  LMapped: Pointer;
begin
  Result := False;
  if not CheckVk(FvkMapMemory(FDevice, ABuffer.Memory, 0, ASize, 0, LMapped),
    'Mapping buffer for upload (vkMapMemory)') then
    Exit;
  Move(AData^, LMapped^, ASize);
  FvkUnmapMemory(FDevice, ABuffer.Memory);
  Result := True;
end;

function TVdxCompute.DownloadFromBuffer(const ABuffer: TVdxGpuBuffer;
  const AData: Pointer; const ASize: VkDeviceSize): Boolean;
var
  LMapped: Pointer;
begin
  Result := False;
  if not CheckVk(FvkMapMemory(FDevice, ABuffer.Memory, 0, ASize, 0, LMapped),
    'Mapping buffer for download (vkMapMemory)') then
    Exit;
  Move(LMapped^, AData^, ASize);
  FvkUnmapMemory(FDevice, ABuffer.Memory);
  Result := True;
end;

function TVdxCompute.MapBufferPersistent(const ABuffer: TVdxGpuBuffer;
  out AData: Pointer): Boolean;
begin
  AData := nil;
  Result := CheckVk(
    FvkMapMemory(FDevice, ABuffer.Memory, 0, ABuffer.Size, 0, AData),
    'Mapping buffer persistent (vkMapMemory)');
end;

procedure TVdxCompute.UnmapBuffer(const ABuffer: TVdxGpuBuffer);
begin
  FvkUnmapMemory(FDevice, ABuffer.Memory);
end;

function TVdxCompute.CopyBuffer(const ASrc, ADst: TVdxGpuBuffer;
  const ASize: VkDeviceSize): Boolean;
var
  LBeginInfo: VkCommandBufferBeginInfo;
  LCopyRegion: VkBufferCopy;
  LSubmitInfo: VkSubmitInfo;
begin
  Result := False;

  if not FBatchMode then
  begin
    FillChar(LBeginInfo, SizeOf(LBeginInfo), 0);
    LBeginInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    LBeginInfo.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if not CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo),
      'Begin copy command buffer (vkBeginCommandBuffer)') then Exit;
  end;

  FillChar(LCopyRegion, SizeOf(LCopyRegion), 0);
  LCopyRegion.srcOffset := 0;
  LCopyRegion.dstOffset := 0;
  LCopyRegion.size      := ASize;
  FvkCmdCopyBuffer(FCommandBuffer, ASrc.Buffer, ADst.Buffer, 1, LCopyRegion);

  if FBatchMode then
    Exit(True);

  if not CheckVk(FvkEndCommandBuffer(FCommandBuffer),
    'End copy command buffer (vkEndCommandBuffer)') then Exit;

  FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
  LSubmitInfo.sType              := VK_STRUCTURE_TYPE_SUBMIT_INFO;
  LSubmitInfo.commandBufferCount := 1;
  LSubmitInfo.pCommandBuffers    := @FCommandBuffer;

  if not CheckVk(FvkResetFences(FDevice, 1, @FFence),
    'Reset fence (vkResetFences)') then Exit;
  if not CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence),
    'Submit copy (vkQueueSubmit)') then Exit;
  if not CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)),
    'Wait for copy fence (vkWaitForFences)') then Exit;

  Result := True;
end;

function TVdxCompute.CopyBufferRegion(const ASrc: TVdxGpuBuffer;
  const ASrcOffset: VkDeviceSize;
  const ADst: TVdxGpuBuffer;
  const ADstOffset, ASize: VkDeviceSize): Boolean;
var
  LBeginInfo: VkCommandBufferBeginInfo;
  LCopyRegion: VkBufferCopy;
  LSubmitInfo: VkSubmitInfo;
begin
  Result := False;

  if not FBatchMode then
  begin
    FillChar(LBeginInfo, SizeOf(LBeginInfo), 0);
    LBeginInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    LBeginInfo.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if not CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo),
      'Begin copy-region command buffer (vkBeginCommandBuffer)') then Exit;
  end;

  FillChar(LCopyRegion, SizeOf(LCopyRegion), 0);
  LCopyRegion.srcOffset := ASrcOffset;
  LCopyRegion.dstOffset := ADstOffset;
  LCopyRegion.size      := ASize;
  FvkCmdCopyBuffer(FCommandBuffer, ASrc.Buffer, ADst.Buffer, 1, LCopyRegion);

  if FBatchMode then
    Exit(True);

  if not CheckVk(FvkEndCommandBuffer(FCommandBuffer),
    'End copy-region command buffer (vkEndCommandBuffer)') then Exit;

  FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
  LSubmitInfo.sType              := VK_STRUCTURE_TYPE_SUBMIT_INFO;
  LSubmitInfo.commandBufferCount := 1;
  LSubmitInfo.pCommandBuffers    := @FCommandBuffer;

  if not CheckVk(FvkResetFences(FDevice, 1, @FFence),
    'Reset fence (vkResetFences)') then Exit;
  if not CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence),
    'Submit copy-region (vkQueueSubmit)') then Exit;
  if not CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)),
    'Wait for copy-region fence (vkWaitForFences)') then Exit;

  Result := True;
end;

// ============================================================================
//  Streaming staging pool
// ============================================================================

function TVdxCompute.EnsureStagingCapacity(const ABytes: UInt64): Boolean;
var
  LNewHost:   TVdxGpuBuffer;
  LNewDevice: TVdxGpuBuffer;
begin
  Result := False;

  if ABytes = 0 then
  begin
    Result := True;
    Exit;
  end;

  if ABytes <= FStagingCapacity then
  begin
    Result := True;
    Exit;
  end;

  // Allocate the new pair first; only destroy the old pair on
  // success so a failed grow leaves the pool in its previous good
  // state. Underlying CreateGpuBuffer populates FErrors on failure.
  LNewHost := CreateGpuBuffer(
    ABytes,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  if FErrors.HasFatal() then Exit;

  LNewDevice := CreateGpuBuffer(
    ABytes,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    vcBuffer);
  if FErrors.HasFatal() then
  begin
    DestroyGpuBuffer(LNewHost);
    Exit;
  end;

  if FStagingCapacity > 0 then
  begin
    DestroyGpuBuffer(FStagingHost);
    DestroyGpuBuffer(FStagingDevice);
  end;

  FStagingHost     := LNewHost;
  FStagingDevice   := LNewDevice;
  FStagingCapacity := ABytes;
  Result := True;
end;

function TVdxCompute.GetStagingHost(): TVdxGpuBuffer;
begin
  Result := FStagingHost;
end;

function TVdxCompute.GetStagingDevice(): TVdxGpuBuffer;
begin
  Result := FStagingDevice;
end;

function TVdxCompute.GetStagingCapacity(): UInt64;
begin
  Result := FStagingCapacity;
end;

// ============================================================================
//  Shader + Pipeline
// ============================================================================

function TVdxCompute.CreateShaderModule(const ACode: Pointer;
  const ACodeSize: NativeUInt): VkShaderModule;
var
  LCreateInfo: VkShaderModuleCreateInfo;
begin
  Result := VK_NULL_HANDLE;
  FillChar(LCreateInfo, SizeOf(LCreateInfo), 0);
  LCreateInfo.sType    := VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  LCreateInfo.codeSize := ACodeSize;
  LCreateInfo.pCode    := ACode;
  if not CheckVk(FvkCreateShaderModule(FDevice, LCreateInfo, nil, Result),
    'Creating shader module (vkCreateShaderModule)') then
    Result := VK_NULL_HANDLE;
end;

procedure TVdxCompute.DestroyShaderModuleHandle(const AModule: VkShaderModule);
begin
  if AModule <> VK_NULL_HANDLE then
    FvkDestroyShaderModule(FDevice, AModule, nil);
end;

function TVdxCompute.CreateComputePipelineSimple(
  const AShaderModule: VkShaderModule;
  const AEntryPoint: PAnsiChar;
  const ADescSetLayout: VkDescriptorSetLayout): TVdxComputePipelineBundle;
var
  LLayoutInfo: VkPipelineLayoutCreateInfo;
  LStageInfo: VkPipelineShaderStageCreateInfo;
  LPipelineInfo: VkComputePipelineCreateInfo;
begin
  Result := Default(TVdxComputePipelineBundle);

  FillChar(LLayoutInfo, SizeOf(LLayoutInfo), 0);
  LLayoutInfo.sType          := VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  LLayoutInfo.setLayoutCount := 1;
  LLayoutInfo.pSetLayouts    := @ADescSetLayout;
  if not CheckVk(FvkCreatePipelineLayout(FDevice, LLayoutInfo, nil, Result.PipelineLayout),
    'Creating pipeline layout (vkCreatePipelineLayout)') then
  begin
    Result := Default(TVdxComputePipelineBundle);
    Exit;
  end;

  FillChar(LStageInfo, SizeOf(LStageInfo), 0);
  LStageInfo.sType  := VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  LStageInfo.stage  := VK_SHADER_STAGE_COMPUTE_BIT;
  LStageInfo.module := AShaderModule;
  LStageInfo.pName  := AEntryPoint;

  FillChar(LPipelineInfo, SizeOf(LPipelineInfo), 0);
  LPipelineInfo.sType  := VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  LPipelineInfo.stage  := LStageInfo;
  LPipelineInfo.layout := Result.PipelineLayout;
  if not CheckVk(
    FvkCreateComputePipelines(FDevice, VK_NULL_HANDLE, 1, LPipelineInfo, nil, @Result.Pipeline),
    'Creating compute pipeline (vkCreateComputePipelines)') then
  begin
    FvkDestroyPipelineLayout(FDevice, Result.PipelineLayout, nil);
    Result := Default(TVdxComputePipelineBundle);
  end;
end;

function TVdxCompute.CreateComputePipelineWithPush(
  const AShaderModule: VkShaderModule;
  const AEntryPoint: PAnsiChar;
  const ADescSetLayout: VkDescriptorSetLayout;
  const APushSize: UInt32): TVdxComputePipelineBundle;
var
  LPushRange: VkPushConstantRange;
  LLayoutInfo: VkPipelineLayoutCreateInfo;
  LStageInfo: VkPipelineShaderStageCreateInfo;
  LPipelineInfo: VkComputePipelineCreateInfo;
begin
  Result := Default(TVdxComputePipelineBundle);

  FillChar(LPushRange, SizeOf(LPushRange), 0);
  LPushRange.stageFlags := VK_SHADER_STAGE_COMPUTE_BIT;
  LPushRange.offset     := 0;
  LPushRange.size       := APushSize;

  FillChar(LLayoutInfo, SizeOf(LLayoutInfo), 0);
  LLayoutInfo.sType                  := VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  LLayoutInfo.setLayoutCount         := 1;
  LLayoutInfo.pSetLayouts            := @ADescSetLayout;
  LLayoutInfo.pushConstantRangeCount := 1;
  LLayoutInfo.pPushConstantRanges    := @LPushRange;
  if not CheckVk(FvkCreatePipelineLayout(FDevice, LLayoutInfo, nil, Result.PipelineLayout),
    'Creating pipeline layout with push constants (vkCreatePipelineLayout)') then
  begin
    Result := Default(TVdxComputePipelineBundle);
    Exit;
  end;

  FillChar(LStageInfo, SizeOf(LStageInfo), 0);
  LStageInfo.sType  := VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  LStageInfo.stage  := VK_SHADER_STAGE_COMPUTE_BIT;
  LStageInfo.module := AShaderModule;
  LStageInfo.pName  := AEntryPoint;

  FillChar(LPipelineInfo, SizeOf(LPipelineInfo), 0);
  LPipelineInfo.sType  := VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  LPipelineInfo.stage  := LStageInfo;
  LPipelineInfo.layout := Result.PipelineLayout;
  if not CheckVk(
    FvkCreateComputePipelines(FDevice, VK_NULL_HANDLE, 1, LPipelineInfo, nil, @Result.Pipeline),
    'Creating compute pipeline with push (vkCreateComputePipelines)') then
  begin
    FvkDestroyPipelineLayout(FDevice, Result.PipelineLayout, nil);
    Result := Default(TVdxComputePipelineBundle);
  end;
end;

function TVdxCompute.CreateComputePipelineWithPushAndSpec(
  const AShaderModule: VkShaderModule;
  const AEntryPoint: PAnsiChar;
  const ADescSetLayout: VkDescriptorSetLayout;
  const APushSize, ASpecValue: UInt32): TVdxComputePipelineBundle;
var
  LPushRange: VkPushConstantRange;
  LLayoutInfo: VkPipelineLayoutCreateInfo;
  LStageInfo: VkPipelineShaderStageCreateInfo;
  LPipelineInfo: VkComputePipelineCreateInfo;
  LSpecEntry: VkSpecializationMapEntry;
  LSpecInfo: VkSpecializationInfo;
begin
  Result := Default(TVdxComputePipelineBundle);

  // Single UInt32 specialization constant at constant_id = 0
  LSpecEntry := Default(VkSpecializationMapEntry);
  LSpecEntry.constantID := 0;
  LSpecEntry.offset     := 0;
  LSpecEntry.size       := SizeOf(UInt32);

  LSpecInfo := Default(VkSpecializationInfo);
  LSpecInfo.mapEntryCount := 1;
  LSpecInfo.pMapEntries   := @LSpecEntry;
  LSpecInfo.dataSize      := SizeOf(UInt32);
  LSpecInfo.pData         := @ASpecValue;

  LPushRange := Default(VkPushConstantRange);
  LPushRange.stageFlags := VK_SHADER_STAGE_COMPUTE_BIT;
  LPushRange.offset     := 0;
  LPushRange.size       := APushSize;

  LLayoutInfo := Default(VkPipelineLayoutCreateInfo);
  LLayoutInfo.sType                  := VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  LLayoutInfo.setLayoutCount         := 1;
  LLayoutInfo.pSetLayouts            := @ADescSetLayout;
  LLayoutInfo.pushConstantRangeCount := 1;
  LLayoutInfo.pPushConstantRanges    := @LPushRange;
  if not CheckVk(FvkCreatePipelineLayout(FDevice, LLayoutInfo, nil, Result.PipelineLayout),
    'Creating pipeline layout with push+spec (vkCreatePipelineLayout)') then
  begin
    Result := Default(TVdxComputePipelineBundle);
    Exit;
  end;

  LStageInfo := Default(VkPipelineShaderStageCreateInfo);
  LStageInfo.sType               := VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  LStageInfo.stage               := VK_SHADER_STAGE_COMPUTE_BIT;
  LStageInfo.module              := AShaderModule;
  LStageInfo.pName               := AEntryPoint;
  LStageInfo.pSpecializationInfo := @LSpecInfo;

  LPipelineInfo := Default(VkComputePipelineCreateInfo);
  LPipelineInfo.sType  := VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  LPipelineInfo.stage  := LStageInfo;
  LPipelineInfo.layout := Result.PipelineLayout;
  if not CheckVk(
    FvkCreateComputePipelines(FDevice, VK_NULL_HANDLE, 1, LPipelineInfo, nil, @Result.Pipeline),
    'Creating compute pipeline with push+spec (vkCreateComputePipelines)') then
  begin
    FvkDestroyPipelineLayout(FDevice, Result.PipelineLayout, nil);
    Result := Default(TVdxComputePipelineBundle);
  end;
end;

procedure TVdxCompute.DestroyComputePipelineBundle(var ABundle: TVdxComputePipelineBundle);
begin
  if ABundle.Pipeline <> VK_NULL_HANDLE then
    FvkDestroyPipeline(FDevice, ABundle.Pipeline, nil);

  if ABundle.PipelineLayout <> VK_NULL_HANDLE then
    FvkDestroyPipelineLayout(FDevice, ABundle.PipelineLayout, nil);

  FillChar(ABundle, SizeOf(ABundle), 0);
end;

// ============================================================================
//  Descriptor Sets
// ============================================================================

function TVdxCompute.CreateStorageDescriptorSetLayout(
  const ABindingCount: UInt32): VkDescriptorSetLayout;
var
  LBindings: array of VkDescriptorSetLayoutBinding;
  LCreateInfo: VkDescriptorSetLayoutCreateInfo;
  LI: UInt32;
begin
  Result := VK_NULL_HANDLE;
  SetLength(LBindings, ABindingCount);

  for LI := 0 to ABindingCount - 1 do
  begin
    FillChar(LBindings[LI], SizeOf(VkDescriptorSetLayoutBinding), 0);
    LBindings[LI].binding         := LI;
    LBindings[LI].descriptorType  := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    LBindings[LI].descriptorCount := 1;
    LBindings[LI].stageFlags      := VK_SHADER_STAGE_COMPUTE_BIT;
  end;

  FillChar(LCreateInfo, SizeOf(LCreateInfo), 0);
  LCreateInfo.sType        := VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  LCreateInfo.bindingCount := ABindingCount;
  LCreateInfo.pBindings    := @LBindings[0];

  if not CheckVk(FvkCreateDescriptorSetLayout(FDevice, LCreateInfo, nil, Result),
    'Creating descriptor set layout (vkCreateDescriptorSetLayout)') then
    Result := VK_NULL_HANDLE;
end;

function TVdxCompute.CreateDescriptorPoolForStorage(
  const AMaxSets, AMaxDescriptors: UInt32): VkDescriptorPool;
var
  LPoolSize: VkDescriptorPoolSize;
  LCreateInfo: VkDescriptorPoolCreateInfo;
begin
  Result := VK_NULL_HANDLE;

  LPoolSize.descriptorType  := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  LPoolSize.descriptorCount := AMaxDescriptors;

  FillChar(LCreateInfo, SizeOf(LCreateInfo), 0);
  LCreateInfo.sType         := VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  LCreateInfo.maxSets       := AMaxSets;
  LCreateInfo.poolSizeCount := 1;
  LCreateInfo.pPoolSizes    := @LPoolSize;

  if not CheckVk(FvkCreateDescriptorPool(FDevice, LCreateInfo, nil, Result),
    'Creating descriptor pool (vkCreateDescriptorPool)') then
    Result := VK_NULL_HANDLE;
end;

function TVdxCompute.AllocateDescriptorSetForBuffers(
  const APool: VkDescriptorPool;
  const ALayout: VkDescriptorSetLayout;
  const ABuffers: array of TVdxGpuBuffer): VkDescriptorSet;
var
  LAllocInfo: VkDescriptorSetAllocateInfo;
  LBufferInfos: array of VkDescriptorBufferInfo;
  LWrites: array of VkWriteDescriptorSet;
  LI: Integer;
begin
  Result := VK_NULL_HANDLE;

  FillChar(LAllocInfo, SizeOf(LAllocInfo), 0);
  LAllocInfo.sType              := VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  LAllocInfo.descriptorPool     := APool;
  LAllocInfo.descriptorSetCount := 1;
  LAllocInfo.pSetLayouts        := @ALayout;
  if not CheckVk(FvkAllocateDescriptorSets(FDevice, LAllocInfo, @Result),
    'Allocating descriptor set (vkAllocateDescriptorSets)') then
  begin
    Result := VK_NULL_HANDLE;
    Exit;
  end;

  SetLength(LBufferInfos, Length(ABuffers));
  SetLength(LWrites,      Length(ABuffers));

  for LI := 0 to High(ABuffers) do
  begin
    LBufferInfos[LI].buffer := ABuffers[LI].Buffer;
    LBufferInfos[LI].offset := 0;
    LBufferInfos[LI].range  := VK_WHOLE_SIZE;

    FillChar(LWrites[LI], SizeOf(VkWriteDescriptorSet), 0);
    LWrites[LI].sType           := VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    LWrites[LI].dstSet          := Result;
    LWrites[LI].dstBinding      := UInt32(LI);
    LWrites[LI].descriptorCount := 1;
    LWrites[LI].descriptorType  := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    LWrites[LI].pBufferInfo     := @LBufferInfos[LI];
  end;

  FvkUpdateDescriptorSets(FDevice, UInt32(Length(LWrites)), @LWrites[0], 0, nil);
end;

procedure TVdxCompute.UpdateDescriptorSetBuffers(const ADescSet: VkDescriptorSet;
  const ABuffers: array of TVdxGpuBuffer);
var
  LBufferInfos: array of VkDescriptorBufferInfo;
  LWrites: array of VkWriteDescriptorSet;
  LI: Integer;
begin
  SetLength(LBufferInfos, Length(ABuffers));
  SetLength(LWrites,      Length(ABuffers));

  for LI := 0 to High(ABuffers) do
  begin
    LBufferInfos[LI].buffer := ABuffers[LI].Buffer;
    LBufferInfos[LI].offset := 0;
    LBufferInfos[LI].range  := VK_WHOLE_SIZE;

    FillChar(LWrites[LI], SizeOf(VkWriteDescriptorSet), 0);
    LWrites[LI].sType           := VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    LWrites[LI].dstSet          := ADescSet;
    LWrites[LI].dstBinding      := UInt32(LI);
    LWrites[LI].descriptorCount := 1;
    LWrites[LI].descriptorType  := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    LWrites[LI].pBufferInfo     := @LBufferInfos[LI];
  end;

  FvkUpdateDescriptorSets(FDevice, UInt32(Length(LWrites)), @LWrites[0], 0, nil);
end;

procedure TVdxCompute.DestroyDescriptorSetLayoutHandle(const ALayout: VkDescriptorSetLayout);
begin
  if ALayout <> VK_NULL_HANDLE then
    FvkDestroyDescriptorSetLayout(FDevice, ALayout, nil);
end;

procedure TVdxCompute.DestroyDescriptorPoolHandle(const APool: VkDescriptorPool);
begin
  if APool = VK_NULL_HANDLE then Exit;

  if FBatchMode then
  begin
    // Defer destruction until EndBatch — descriptor sets allocated
    // from this pool must remain valid until all batched dispatches
    // have actually executed on the GPU.
    if FBatchDeferredPoolCount >= Length(FBatchDeferredPools) then
      SetLength(FBatchDeferredPools, FBatchDeferredPoolCount + 64);
    FBatchDeferredPools[FBatchDeferredPoolCount] := APool;
    Inc(FBatchDeferredPoolCount);
  end
  else
    FvkDestroyDescriptorPool(FDevice, APool, nil);
end;

// ============================================================================
//  Dispatch
// ============================================================================

function TVdxCompute.DispatchCompute(const APipeline: VkPipeline;
  const APipelineLayout: VkPipelineLayout;
  const ADescSet: VkDescriptorSet;
  const AGroupsX: UInt32;
  const AGroupsY: UInt32;
  const AGroupsZ: UInt32): Boolean;
var
  LBeginInfo: VkCommandBufferBeginInfo;
  LSubmitInfo: VkSubmitInfo;
begin
  Result := False;

  if not FBatchMode then
  begin
    FillChar(LBeginInfo, SizeOf(LBeginInfo), 0);
    LBeginInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    LBeginInfo.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if not CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo),
      'Begin dispatch command buffer (vkBeginCommandBuffer)') then Exit;
  end;

  FvkCmdBindPipeline(FCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, APipeline);
  FvkCmdBindDescriptorSets(FCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
    APipelineLayout, 0, 1, @ADescSet, 0, nil);
  FvkCmdDispatch(FCommandBuffer, AGroupsX, AGroupsY, AGroupsZ);

  if FBatchMode then
    Exit(True);

  if not CheckVk(FvkEndCommandBuffer(FCommandBuffer),
    'End dispatch command buffer (vkEndCommandBuffer)') then Exit;

  FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
  LSubmitInfo.sType              := VK_STRUCTURE_TYPE_SUBMIT_INFO;
  LSubmitInfo.commandBufferCount := 1;
  LSubmitInfo.pCommandBuffers    := @FCommandBuffer;

  if not CheckVk(FvkResetFences(FDevice, 1, @FFence),
    'Reset fence (vkResetFences)') then Exit;
  if not CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence),
    'Submit dispatch (vkQueueSubmit)') then Exit;
  if not CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)),
    'Wait for dispatch fence (vkWaitForFences)') then Exit;

  Result := True;
end;

function TVdxCompute.DispatchComputeWithPush(const APipeline: VkPipeline;
  const APipelineLayout: VkPipelineLayout;
  const ADescSet: VkDescriptorSet;
  const APushData: Pointer; const APushSize: UInt32;
  const AGroupsX: UInt32;
  const AGroupsY: UInt32;
  const AGroupsZ: UInt32): Boolean;
var
  LBeginInfo: VkCommandBufferBeginInfo;
  LSubmitInfo: VkSubmitInfo;
begin
  Result := False;

  if not FBatchMode then
  begin
    FillChar(LBeginInfo, SizeOf(LBeginInfo), 0);
    LBeginInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    LBeginInfo.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if not CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo),
      'Begin push-dispatch command buffer (vkBeginCommandBuffer)') then Exit;
  end;

  FvkCmdBindPipeline(FCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, APipeline);
  FvkCmdBindDescriptorSets(FCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
    APipelineLayout, 0, 1, @ADescSet, 0, nil);
  FvkCmdPushConstants(FCommandBuffer, APipelineLayout,
    VK_SHADER_STAGE_COMPUTE_BIT, 0, APushSize, APushData);
  FvkCmdDispatch(FCommandBuffer, AGroupsX, AGroupsY, AGroupsZ);

  if FBatchMode then
    Exit(True);

  if not CheckVk(FvkEndCommandBuffer(FCommandBuffer),
    'End push-dispatch command buffer (vkEndCommandBuffer)') then Exit;

  FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
  LSubmitInfo.sType              := VK_STRUCTURE_TYPE_SUBMIT_INFO;
  LSubmitInfo.commandBufferCount := 1;
  LSubmitInfo.pCommandBuffers    := @FCommandBuffer;

  if not CheckVk(FvkResetFences(FDevice, 1, @FFence),
    'Reset fence (vkResetFences)') then Exit;
  if not CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence),
    'Submit push-dispatch (vkQueueSubmit)') then Exit;
  if not CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)),
    'Wait for push-dispatch fence (vkWaitForFences)') then Exit;

  Result := True;
end;

// ============================================================================
//  Batch Mode
// ============================================================================

procedure TVdxCompute.InsertBatchBarrier();
var
  LBarrier: VkMemoryBarrier;
begin
  // Full memory barrier covering compute->compute and transfer->compute.
  FillChar(LBarrier, SizeOf(LBarrier), 0);
  LBarrier.sType         := VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  LBarrier.srcAccessMask := VK_ACCESS_SHADER_WRITE_BIT or VK_ACCESS_TRANSFER_WRITE_BIT;
  LBarrier.dstAccessMask := VK_ACCESS_SHADER_READ_BIT or VK_ACCESS_SHADER_WRITE_BIT
    or VK_ACCESS_TRANSFER_READ_BIT or VK_ACCESS_TRANSFER_WRITE_BIT;

  FvkCmdPipelineBarrier(
    FCommandBuffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT or VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT or VK_PIPELINE_STAGE_TRANSFER_BIT,
    0,
    1, @LBarrier,
    0, nil,
    0, nil);
end;

procedure TVdxCompute.BatchBarrier();
begin
  if FBatchMode then
    InsertBatchBarrier();
end;

procedure TVdxCompute.BeginBatch();
var
  LBeginInfo: VkCommandBufferBeginInfo;
begin
  // Programmer error — nested batch entry has no sensible fallback
  // behavior. Raise is justified per TASK-REFACTOR §2.3.
  if FBatchMode then
    raise Exception.Create('TVdxCompute.BeginBatch: already in batch mode');

  FillChar(LBeginInfo, SizeOf(LBeginInfo), 0);
  LBeginInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  LBeginInfo.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if not CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo),
    'Begin batch command buffer (vkBeginCommandBuffer)') then
    Exit;

  FBatchMode := True;
  FBatchDeferredPoolCount := 0;
end;

procedure TVdxCompute.EndBatch();
var
  LSubmitInfo: VkSubmitInfo;
  LI: Integer;
begin
  // Programmer error — EndBatch without prior BeginBatch has no
  // sensible fallback. Raise is justified per TASK-REFACTOR §2.3.
  if not FBatchMode then
    raise Exception.Create('TVdxCompute.EndBatch: not in batch mode');

  if not CheckVk(FvkEndCommandBuffer(FCommandBuffer),
    'End batch command buffer (vkEndCommandBuffer)') then
  begin
    FBatchMode := False;
    Exit;
  end;

  FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
  LSubmitInfo.sType              := VK_STRUCTURE_TYPE_SUBMIT_INFO;
  LSubmitInfo.commandBufferCount := 1;
  LSubmitInfo.pCommandBuffers    := @FCommandBuffer;

  if not CheckVk(FvkResetFences(FDevice, 1, @FFence),
    'Reset batch fence (vkResetFences)') then
  begin
    FBatchMode := False;
    Exit;
  end;
  if not CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence),
    'Submit batch (vkQueueSubmit)') then
  begin
    FBatchMode := False;
    Exit;
  end;
  if not CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)),
    'Wait for batch fence (vkWaitForFences)') then
  begin
    FBatchMode := False;
    Exit;
  end;

  // Destroy deferred descriptor pools now that GPU work is complete.
  for LI := 0 to FBatchDeferredPoolCount - 1 do
    FvkDestroyDescriptorPool(FDevice, FBatchDeferredPools[LI], nil);
  FBatchDeferredPoolCount := 0;

  FBatchMode := False;
end;

// ============================================================================
//  Queries
// ============================================================================

function TVdxCompute.GetDeviceName(): string;
begin
  Result := string(AnsiString(PAnsiChar(@FDeviceProperties.deviceName[0])));
end;

function TVdxCompute.GetVRAMSizeMB(): UInt64;
var
  LI: UInt32;
  LLargest: UInt64;
begin
  // Largest device-local heap in MB.
  LLargest := 0;
  for LI := 0 to FMemoryProperties.memoryHeapCount - 1 do
  begin
    if (FMemoryProperties.memoryHeaps[LI].flags and $00000001) <> 0 then
      if FMemoryProperties.memoryHeaps[LI].size > LLargest then
        LLargest := FMemoryProperties.memoryHeaps[LI].size;
  end;
  Result := LLargest div (1024 * 1024);
end;

function TVdxCompute.GetMaxComputeWorkGroupSize(): UInt32;
begin
  Result := FDeviceProperties.limits.maxComputeWorkGroupInvocations;
end;

function TVdxCompute.GetVRAMUsage(): TVdxVRAMUsage;
begin
  Result.WeightsBytes := FVRAMWeightsBytes;
  Result.CacheBytes   := FVRAMCacheBytes;
  Result.BuffersBytes := FVRAMBuffersBytes;
  Result.TotalBytes   := FVRAMWeightsBytes + FVRAMCacheBytes + FVRAMBuffersBytes;
end;

function TVdxCompute.GetSelectedGpuIndex(): Integer;
begin
  Result := FSelectedGpuIndex;
end;

end.
