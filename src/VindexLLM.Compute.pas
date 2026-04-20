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

type


  { TVdxGpuDeviceType }
  TVdxGpuDeviceType = (
    gdtOther,
    gdtIntegrated,
    gdtDiscrete,
    gdtVirtual,
    gdtCpu
  );

  { TVdxGpuInfo }
  TVdxGpuInfo = record
    // Identity
    Index:      Integer;
    Name:       string;
    DeviceType: TVdxGpuDeviceType;
    VendorID:   UInt32;
    DeviceID:   UInt32;

    // Versions
    ApiVersion:    UInt32;
    ApiVersionStr: string;
    DriverVersion: UInt32;

    // Memory (MB)
    VRAMMB:         UInt64;
    SharedMemoryMB: UInt64;
    TotalMemoryMB:  UInt64;

    // Queue capabilities
    HasComputeQueue: Boolean;

    // Compute limits
    MaxComputeWorkGroupInvocations: UInt32;

    // Subgroup (wavefront / warp)
    SubgroupSize: UInt32;

    // Shader feature flags
    SupportsFp16: Boolean;
    SupportsInt8: Boolean;
    SupportsInt16: Boolean;
    SupportsInt64: Boolean;
    SupportsFp64: Boolean;
  end;

  { TVdxGpuBuffer }
  TVdxGpuBuffer = record
    Buffer: VkBuffer;
    Memory: VkDeviceMemory;
    Size:   VkDeviceSize;
  end;

  { TVdxComputePipelineBundle }
  TVdxComputePipelineBundle = record
    Pipeline:       VkPipeline;
    PipelineLayout: VkPipelineLayout;
  end;

type

  { TVdxCompute }
  TVdxCompute = class(TVdxBaseObject)
  private
    // Library
    FLibHandle: THandle;

    // Core objects
    FInstance:           VkInstance;
    FPhysicalDevice:    VkPhysicalDevice;
    FDevice:            VkDevice;
    FComputeQueue:      VkQueue;
    FComputeQueueFamily: UInt32;
    FCommandPool:       VkCommandPool;
    FCommandBuffer:     VkCommandBuffer;
    FFence:             VkFence;

    // Batch mode — records multiple dispatches into one command buffer submission
    FBatchMode: Boolean;
    FBatchDeferredPools: array of VkDescriptorPool;
    FBatchDeferredPoolCount: Integer;

    // Device info
    FDeviceProperties:  VkPhysicalDeviceProperties;
    FMemoryProperties:  VkPhysicalDeviceMemoryProperties;
    FSelectedGpuIndex:  Integer;

    // GPU enumeration cache
    FGpusCache:             TArray<TVdxGpuInfo>;
    FPhysicalDeviceHandles: TArray<VkPhysicalDevice>;

    // Function pointers — bootstrap
    FvkGetInstanceProcAddr: TvkGetInstanceProcAddr;

    // Function pointers — instance level
    FvkCreateInstance:                         TvkCreateInstance;
    FvkDestroyInstance:                        TvkDestroyInstance;
    FvkEnumeratePhysicalDevices:               TvkEnumeratePhysicalDevices;
    FvkGetPhysicalDeviceProperties:            TvkGetPhysicalDeviceProperties;
    FvkGetPhysicalDeviceQueueFamilyProperties: TvkGetPhysicalDeviceQueueFamilyProperties;
    FvkGetPhysicalDeviceMemoryProperties:      TvkGetPhysicalDeviceMemoryProperties;

    // Function pointers — device level
    FvkCreateDevice:              TvkCreateDevice;
    FvkDestroyDevice:             TvkDestroyDevice;
    FvkGetDeviceQueue:            TvkGetDeviceQueue;
    FvkCreateBuffer:              TvkCreateBuffer;
    FvkDestroyBuffer:             TvkDestroyBuffer;
    FvkGetBufferMemoryRequirements: TvkGetBufferMemoryRequirements;
    FvkAllocateMemory:            TvkAllocateMemory;
    FvkFreeMemory:                TvkFreeMemory;
    FvkBindBufferMemory:          TvkBindBufferMemory;
    FvkMapMemory:                 TvkMapMemory;
    FvkUnmapMemory:               TvkUnmapMemory;
    FvkFlushMappedMemoryRanges:   TvkFlushMappedMemoryRanges;
    FvkCreateShaderModule:        TvkCreateShaderModule;
    FvkDestroyShaderModule:       TvkDestroyShaderModule;
    FvkCreatePipelineLayout:      TvkCreatePipelineLayout;
    FvkDestroyPipelineLayout:     TvkDestroyPipelineLayout;
    FvkCreateComputePipelines:    TvkCreateComputePipelines;
    FvkDestroyPipeline:           TvkDestroyPipeline;
    FvkCreateDescriptorSetLayout: TvkCreateDescriptorSetLayout;
    FvkDestroyDescriptorSetLayout: TvkDestroyDescriptorSetLayout;
    FvkCreateDescriptorPool:      TvkCreateDescriptorPool;
    FvkDestroyDescriptorPool:     TvkDestroyDescriptorPool;
    FvkAllocateDescriptorSets:    TvkAllocateDescriptorSets;
    FvkUpdateDescriptorSets:      TvkUpdateDescriptorSets;
    FvkCreateCommandPool:         TvkCreateCommandPool;
    FvkDestroyCommandPool:        TvkDestroyCommandPool;
    FvkAllocateCommandBuffers:    TvkAllocateCommandBuffers;
    FvkBeginCommandBuffer:        TvkBeginCommandBuffer;
    FvkEndCommandBuffer:          TvkEndCommandBuffer;
    FvkCmdBindPipeline:           TvkCmdBindPipeline;
    FvkCmdBindDescriptorSets:     TvkCmdBindDescriptorSets;
    FvkCmdDispatch:               TvkCmdDispatch;
    FvkQueueSubmit:               TvkQueueSubmit;
    FvkQueueWaitIdle:             TvkQueueWaitIdle;
    FvkCreateFence:               TvkCreateFence;
    FvkDestroyFence:              TvkDestroyFence;
    FvkWaitForFences:             TvkWaitForFences;
    FvkResetFences:               TvkResetFences;
    FvkCmdCopyBuffer:             TvkCmdCopyBuffer;
    FvkCmdPushConstants:          TvkCmdPushConstants;
    FvkCmdPipelineBarrier:        TvkCmdPipelineBarrier;

    // Internal helpers
    procedure LoadVulkanLibrary();
    procedure LoadGlobalFunctions();
    procedure LoadInstanceFunctions();
    procedure LoadDeviceFunctions();
    procedure CreateVkInstance();
    procedure PopulateGpuCache();
    procedure SelectPhysicalDevice(const AGpuIndex: Integer);
    procedure CreateLogicalDevice();
    procedure CreateCommandResources();
    procedure InsertBatchBarrier();
    function  FindMemoryType(const ATypeBits: UInt32; const AProperties: VkFlags): UInt32;
    function  GetVkProc(const AName: PAnsiChar): Pointer;
    function  CheckVk(const AResult: VkResult; const AContext: string): Boolean;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Enumerate all Vulkan-capable GPUs. Safe to call before Init.
    function  EnumerateGpus(out AGpus: TArray<TVdxGpuInfo>): Boolean;

    // Initialize Vulkan on a specific GPU. AGpuIndex:
    //   -1 (default) — auto: prefer discrete, fall back to first compute-capable
    //   >= 0         — explicit index from EnumerateGpus
    procedure Init(const AGpuIndex: Integer = -1);

    // Buffer operations
    function  CreateGpuBuffer(const ASize: VkDeviceSize; const AUsage: VkFlags; const AMemProps: VkFlags): TVdxGpuBuffer;
    procedure DestroyGpuBuffer(var ABuffer: TVdxGpuBuffer);
    procedure UploadToBuffer(const ABuffer: TVdxGpuBuffer; const AData: Pointer; const ASize: VkDeviceSize);
    procedure DownloadFromBuffer(const ABuffer: TVdxGpuBuffer; const AData: Pointer; const ASize: VkDeviceSize);
    function  MapBufferPersistent(const ABuffer: TVdxGpuBuffer): Pointer;
    procedure UnmapBuffer(const ABuffer: TVdxGpuBuffer);

    // Shader + pipeline
    function  CreateShaderModule(const ACode: Pointer; const ACodeSize: NativeUInt): VkShaderModule;
    procedure DestroyShaderModuleHandle(const AModule: VkShaderModule);
    function  CreateComputePipelineSimple(const AShaderModule: VkShaderModule; const AEntryPoint: PAnsiChar; const ADescSetLayout: VkDescriptorSetLayout): TVdxComputePipelineBundle;
    function  CreateComputePipelineWithPush(const AShaderModule: VkShaderModule; const AEntryPoint: PAnsiChar; const ADescSetLayout: VkDescriptorSetLayout; const APushSize: UInt32): TVdxComputePipelineBundle;
    function  CreateComputePipelineWithPushAndSpec(const AShaderModule: VkShaderModule; const AEntryPoint: PAnsiChar; const ADescSetLayout: VkDescriptorSetLayout; const APushSize: UInt32; const ASpecValue: UInt32): TVdxComputePipelineBundle;
    procedure DestroyComputePipelineBundle(var ABundle: TVdxComputePipelineBundle);

    // Descriptor sets
    function  CreateStorageDescriptorSetLayout(const ABindingCount: UInt32): VkDescriptorSetLayout;
    function  CreateDescriptorPoolForStorage(const AMaxSets: UInt32; const AMaxDescriptors: UInt32): VkDescriptorPool;
    function  AllocateDescriptorSetForBuffers(const APool: VkDescriptorPool; const ALayout: VkDescriptorSetLayout; const ABuffers: array of TVdxGpuBuffer): VkDescriptorSet;
    procedure UpdateDescriptorSetBuffers(const ADescSet: VkDescriptorSet; const ABuffers: array of TVdxGpuBuffer);

    // Dispatch
    procedure DispatchCompute(const APipeline: VkPipeline; const APipelineLayout: VkPipelineLayout; const ADescSet: VkDescriptorSet; const AGroupsX: UInt32; const AGroupsY: UInt32 = 1; const AGroupsZ: UInt32 = 1);
    procedure DispatchComputeWithPush(const APipeline: VkPipeline; const APipelineLayout: VkPipelineLayout; const ADescSet: VkDescriptorSet; const APushData: Pointer; const APushSize: UInt32; const AGroupsX: UInt32; const AGroupsY: UInt32 = 1; const AGroupsZ: UInt32 = 1);

    // Buffer-to-buffer copy (for staging uploads to device-local memory)
    procedure CopyBuffer(const ASrc: TVdxGpuBuffer; const ADst: TVdxGpuBuffer; const ASize: VkDeviceSize);
    procedure CopyBufferRegion(const ASrc: TVdxGpuBuffer; const ASrcOffset: VkDeviceSize; const ADst: TVdxGpuBuffer; const ADstOffset: VkDeviceSize; const ASize: VkDeviceSize);

    // Batch mode — record multiple dispatches into one command buffer submission
    // Reduces GPU round-trips from ~250 per token to ~1 per layer
    procedure BeginBatch();
    procedure EndBatch();

    // Explicit memory barrier — caller inserts only where true data
    // dependencies exist (write→read on the same buffer).  No-op outside
    // batch mode because non-batch dispatches are fully synchronised.
    procedure BatchBarrier();

    // Cleanup helpers
    procedure DestroyDescriptorSetLayoutHandle(const ALayout: VkDescriptorSetLayout);
    procedure DestroyDescriptorPoolHandle(const APool: VkDescriptorPool);

    // Queries
    function  GetSelectedGpuIndex(): Integer;
    function  GetDeviceName(): string;
    function  GetVRAMSizeMB(): UInt64;
    function  GetMaxComputeWorkGroupSize(): UInt32;
  end;

implementation

{ TVdxCompute }

constructor TVdxCompute.Create();
begin
  inherited;

  FLibHandle := 0;
  FInstance := nil;
  FPhysicalDevice := nil;
  FDevice := nil;
  FComputeQueue := nil;
  FComputeQueueFamily := 0;
  FCommandPool := VK_NULL_HANDLE;
  FCommandBuffer := nil;
  FFence := VK_NULL_HANDLE;
  FBatchMode := False;
  FBatchDeferredPoolCount := 0;
  FSelectedGpuIndex := -1;
end;

procedure TVdxCompute.Init(const AGpuIndex: Integer);
begin
  Status('Loading Vulkan...');
  LoadVulkanLibrary();
  if FErrors.HasErrors() then Exit;

  LoadGlobalFunctions();
  if FErrors.HasErrors() then Exit;

  Status('Creating Vulkan instance...');
  CreateVkInstance();
  if FErrors.HasErrors() then Exit;

  LoadInstanceFunctions();

  Status('Selecting GPU...');
  SelectPhysicalDevice(AGpuIndex);
  if FErrors.HasErrors() then Exit;

  Status('Creating logical device...');
  CreateLogicalDevice();
  if FErrors.HasErrors() then Exit;

  LoadDeviceFunctions();
  FvkGetDeviceQueue(FDevice, FComputeQueueFamily, 0, FComputeQueue);

  Status('Creating command resources...');
  CreateCommandResources();
  if FErrors.HasErrors() then Exit;

  Status('Vulkan ready: %s (%d MB VRAM)', [GetDeviceName(), GetVRAMSizeMB()]);
end;

destructor TVdxCompute.Destroy();
begin
  if FDevice <> nil then
  begin
    // Wait for GPU to finish before cleanup
    if Assigned(FvkQueueWaitIdle) then
      FvkQueueWaitIdle(FComputeQueue);

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

procedure TVdxCompute.LoadVulkanLibrary();
begin
  try
    FLibHandle := LoadLibrary('vulkan-1.dll');
    if FLibHandle = 0 then
    begin
      FErrors.Add(esError, 'VULKAN_LOAD', 'Failed to load vulkan-1.dll - no Vulkan driver installed');
      Exit;
    end;

    FvkGetInstanceProcAddr := TvkGetInstanceProcAddr(GetProcAddress(FLibHandle, 'vkGetInstanceProcAddr'));
    if not Assigned(FvkGetInstanceProcAddr) then
    begin
      FErrors.Add(esError, 'VULKAN_LOAD', 'vkGetInstanceProcAddr not found in vulkan-1.dll');
      Exit;
    end;
  except
    on E: Exception do
    begin
      FErrors.Add(esError, 'VULKAN_LOAD', 'Exception loading Vulkan library: %s', [E.Message]);
      Exit;
    end;
  end;
end;

function TVdxCompute.GetVkProc(const AName: PAnsiChar): Pointer;
begin
  // Try instance-level first, then fall back to GetProcAddress on the DLL
  if FInstance <> nil then
    Result := FvkGetInstanceProcAddr(FInstance, AName)
  else
    Result := FvkGetInstanceProcAddr(nil, AName);

  if Result = nil then
    Result := GetProcAddress(FLibHandle, AName);
end;

procedure TVdxCompute.LoadGlobalFunctions();
begin
  @FvkCreateInstance := GetVkProc('vkCreateInstance');
  if not Assigned(FvkCreateInstance) then
  begin
    FErrors.Add(esError, 'VULKAN_FUNC', 'Failed to load %s', ['vkCreateInstance']);
    Exit;
  end;
end;

procedure TVdxCompute.LoadInstanceFunctions();
begin
  @FvkDestroyInstance := GetVkProc('vkDestroyInstance');
  @FvkEnumeratePhysicalDevices := GetVkProc('vkEnumeratePhysicalDevices');
  @FvkGetPhysicalDeviceProperties := GetVkProc('vkGetPhysicalDeviceProperties');
  @FvkGetPhysicalDeviceQueueFamilyProperties := GetVkProc('vkGetPhysicalDeviceQueueFamilyProperties');
  @FvkGetPhysicalDeviceMemoryProperties := GetVkProc('vkGetPhysicalDeviceMemoryProperties');
  @FvkCreateDevice := GetVkProc('vkCreateDevice');
end;

procedure TVdxCompute.LoadDeviceFunctions();
begin
  @FvkDestroyDevice := GetVkProc('vkDestroyDevice');
  @FvkGetDeviceQueue := GetVkProc('vkGetDeviceQueue');
  @FvkCreateBuffer := GetVkProc('vkCreateBuffer');
  @FvkDestroyBuffer := GetVkProc('vkDestroyBuffer');
  @FvkGetBufferMemoryRequirements := GetVkProc('vkGetBufferMemoryRequirements');
  @FvkAllocateMemory := GetVkProc('vkAllocateMemory');
  @FvkFreeMemory := GetVkProc('vkFreeMemory');
  @FvkBindBufferMemory := GetVkProc('vkBindBufferMemory');
  @FvkMapMemory := GetVkProc('vkMapMemory');
  @FvkUnmapMemory := GetVkProc('vkUnmapMemory');
  @FvkFlushMappedMemoryRanges := GetVkProc('vkFlushMappedMemoryRanges');
  @FvkCreateShaderModule := GetVkProc('vkCreateShaderModule');
  @FvkDestroyShaderModule := GetVkProc('vkDestroyShaderModule');
  @FvkCreatePipelineLayout := GetVkProc('vkCreatePipelineLayout');
  @FvkDestroyPipelineLayout := GetVkProc('vkDestroyPipelineLayout');
  @FvkCreateComputePipelines := GetVkProc('vkCreateComputePipelines');
  @FvkDestroyPipeline := GetVkProc('vkDestroyPipeline');
  @FvkCreateDescriptorSetLayout := GetVkProc('vkCreateDescriptorSetLayout');
  @FvkDestroyDescriptorSetLayout := GetVkProc('vkDestroyDescriptorSetLayout');
  @FvkCreateDescriptorPool := GetVkProc('vkCreateDescriptorPool');
  @FvkDestroyDescriptorPool := GetVkProc('vkDestroyDescriptorPool');
  @FvkAllocateDescriptorSets := GetVkProc('vkAllocateDescriptorSets');
  @FvkUpdateDescriptorSets := GetVkProc('vkUpdateDescriptorSets');
  @FvkCreateCommandPool := GetVkProc('vkCreateCommandPool');
  @FvkDestroyCommandPool := GetVkProc('vkDestroyCommandPool');
  @FvkAllocateCommandBuffers := GetVkProc('vkAllocateCommandBuffers');
  @FvkBeginCommandBuffer := GetVkProc('vkBeginCommandBuffer');
  @FvkEndCommandBuffer := GetVkProc('vkEndCommandBuffer');
  @FvkCmdBindPipeline := GetVkProc('vkCmdBindPipeline');
  @FvkCmdBindDescriptorSets := GetVkProc('vkCmdBindDescriptorSets');
  @FvkCmdDispatch := GetVkProc('vkCmdDispatch');
  @FvkQueueSubmit := GetVkProc('vkQueueSubmit');
  @FvkQueueWaitIdle := GetVkProc('vkQueueWaitIdle');
  @FvkCreateFence := GetVkProc('vkCreateFence');
  @FvkDestroyFence := GetVkProc('vkDestroyFence');
  @FvkWaitForFences := GetVkProc('vkWaitForFences');
  @FvkResetFences := GetVkProc('vkResetFences');
  @FvkCmdCopyBuffer := GetVkProc('vkCmdCopyBuffer');
  @FvkCmdPushConstants := GetVkProc('vkCmdPushConstants');
  @FvkCmdPipelineBarrier := GetVkProc('vkCmdPipelineBarrier');
end;

function TVdxCompute.CheckVk(const AResult: VkResult; const AContext: string): Boolean;
begin
  Result := (AResult = VK_SUCCESS);
  if not Result then
    FErrors.Add(esError, 'VULKAN', '%s failed (VkResult=%d)', [AContext, Ord(AResult)]);
end;

procedure TVdxCompute.CreateVkInstance();
var
  LAppInfo: VkApplicationInfo;
  LCreateInfo: VkInstanceCreateInfo;
begin
  FillChar(LAppInfo, SizeOf(LAppInfo), 0);
  LAppInfo.sType := VK_STRUCTURE_TYPE_APPLICATION_INFO;
  LAppInfo.pApplicationName := 'VindexLLM';
  LAppInfo.applicationVersion := 1;
  LAppInfo.pEngineName := 'VindexLLM';
  LAppInfo.engineVersion := 1;
  LAppInfo.apiVersion := VK_API_VERSION_1_0;

  FillChar(LCreateInfo, SizeOf(LCreateInfo), 0);
  LCreateInfo.sType := VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  LCreateInfo.pApplicationInfo := @LAppInfo;

  if not CheckVk(FvkCreateInstance(LCreateInfo, nil, FInstance), 'vkCreateInstance') then Exit;
end;

procedure TVdxCompute.PopulateGpuCache();
var
  LDeviceCount: UInt32;
  LProps: VkPhysicalDeviceProperties;
  LMemProps: VkPhysicalDeviceMemoryProperties;
  LFamilyCount: UInt32;
  LFamilies: array of VkQueueFamilyProperties;
  LI: Integer;
  LJ: Integer;
  LInfo: TVdxGpuInfo;
  LHeapIdx: UInt32;
  LDeviceLocalMB: UInt64;
begin
  LDeviceCount := 0;
  if not CheckVk(FvkEnumeratePhysicalDevices(FInstance, LDeviceCount, nil), 'vkEnumeratePhysicalDevices(count)') then Exit;
  if LDeviceCount = 0 then
  begin
    FErrors.Add(esError, 'VULKAN_GPU', 'No Vulkan-capable GPU found');
    Exit;
  end;

  SetLength(FPhysicalDeviceHandles, LDeviceCount);
  if not CheckVk(FvkEnumeratePhysicalDevices(FInstance, LDeviceCount, @FPhysicalDeviceHandles[0]), 'vkEnumeratePhysicalDevices') then Exit;

  SetLength(FGpusCache, LDeviceCount);
  for LI := 0 to Integer(LDeviceCount) - 1 do
  begin
    FillChar(LInfo, SizeOf(LInfo), 0);
    LInfo.Index := LI;

    FvkGetPhysicalDeviceProperties(FPhysicalDeviceHandles[LI], LProps);
    LInfo.Name := string(AnsiString(PAnsiChar(@LProps.deviceName[0])));
    LInfo.DeviceType := TVdxGpuDeviceType(LProps.deviceType);
    LInfo.VendorID := LProps.vendorID;
    LInfo.DeviceID := LProps.deviceID;
    LInfo.ApiVersion := LProps.apiVersion;
    LInfo.ApiVersionStr := Format('%d.%d.%d', [
      (LProps.apiVersion shr 22) and $3FF,
      (LProps.apiVersion shr 12) and $3FF,
      LProps.apiVersion and $FFF]);
    LInfo.DriverVersion := LProps.driverVersion;
    LInfo.MaxComputeWorkGroupInvocations := LProps.limits.maxComputeWorkGroupInvocations;

    // VRAM — sum device-local heap sizes
    FvkGetPhysicalDeviceMemoryProperties(FPhysicalDeviceHandles[LI], LMemProps);
    LDeviceLocalMB := 0;
    for LHeapIdx := 0 to LMemProps.memoryHeapCount - 1 do
    begin
      if (LMemProps.memoryHeaps[LHeapIdx].flags and VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) <> 0 then
        LDeviceLocalMB := LDeviceLocalMB + (LMemProps.memoryHeaps[LHeapIdx].size div (1024 * 1024));
    end;
    LInfo.VRAMMB := LDeviceLocalMB;

    // Compute queue check
    LFamilyCount := 0;
    FvkGetPhysicalDeviceQueueFamilyProperties(FPhysicalDeviceHandles[LI], LFamilyCount, nil);
    SetLength(LFamilies, LFamilyCount);
    FvkGetPhysicalDeviceQueueFamilyProperties(FPhysicalDeviceHandles[LI], LFamilyCount, @LFamilies[0]);
    LInfo.HasComputeQueue := False;
    for LJ := 0 to Integer(LFamilyCount) - 1 do
    begin
      if (LFamilies[LJ].queueFlags and VK_QUEUE_COMPUTE_BIT) <> 0 then
      begin
        LInfo.HasComputeQueue := True;
        Break;
      end;
    end;

    FGpusCache[LI] := LInfo;
  end;
end;

procedure TVdxCompute.SelectPhysicalDevice(const AGpuIndex: Integer);
var
  LTargetIndex: Integer;
  LFamilyCount: UInt32;
  LFamilies: array of VkQueueFamilyProperties;
  LJ: Integer;
  LI: Integer;
begin
  if Length(FGpusCache) = 0 then
  begin
    PopulateGpuCache();
    if FErrors.HasErrors() then Exit;
  end;

  // Select GPU
  if AGpuIndex = -1 then
  begin
    // Auto-select: prefer discrete, fall back to first compute-capable
    LTargetIndex := -1;
    for LI := 0 to High(FGpusCache) do
    begin
      if not FGpusCache[LI].HasComputeQueue then
        Continue;
      if (LTargetIndex = -1) or (FGpusCache[LI].DeviceType = gdtDiscrete) then
        LTargetIndex := LI;
      if FGpusCache[LI].DeviceType = gdtDiscrete then
        Break;
    end;
    if LTargetIndex = -1 then
    begin
      FErrors.Add(esError, 'VULKAN_GPU', 'No GPU with compute queue found');
      Exit;
    end;
  end
  else
  begin
    // Explicit index
    if (AGpuIndex < 0) or (AGpuIndex >= Length(FGpusCache)) then
    begin
      FErrors.Add(esError, 'VULKAN_GPU', 'GPU index %d out of range (0..%d)', [AGpuIndex, Length(FGpusCache) - 1]);
      Exit;
    end;
    if not FGpusCache[AGpuIndex].HasComputeQueue then
    begin
      FErrors.Add(esError, 'VULKAN_GPU', 'GPU %d (%s) has no compute queue', [AGpuIndex, FGpusCache[AGpuIndex].Name]);
      Exit;
    end;
    LTargetIndex := AGpuIndex;
  end;

  FSelectedGpuIndex := LTargetIndex;
  FPhysicalDevice := FPhysicalDeviceHandles[LTargetIndex];
  FvkGetPhysicalDeviceProperties(FPhysicalDevice, FDeviceProperties);
  FvkGetPhysicalDeviceMemoryProperties(FPhysicalDevice, FMemoryProperties);

  // Find compute queue family on selected device
  LFamilyCount := 0;
  FvkGetPhysicalDeviceQueueFamilyProperties(FPhysicalDevice, LFamilyCount, nil);
  SetLength(LFamilies, LFamilyCount);
  FvkGetPhysicalDeviceQueueFamilyProperties(FPhysicalDevice, LFamilyCount, @LFamilies[0]);
  for LJ := 0 to Integer(LFamilyCount) - 1 do
  begin
    if (LFamilies[LJ].queueFlags and VK_QUEUE_COMPUTE_BIT) <> 0 then
    begin
      FComputeQueueFamily := UInt32(LJ);
      Break;
    end;
  end;

  Status('Selected GPU %d: %s (%d MB VRAM)', [LTargetIndex, FGpusCache[LTargetIndex].Name, FGpusCache[LTargetIndex].VRAMMB]);
end;

procedure TVdxCompute.CreateLogicalDevice();
var
  LQueuePriority: Single;
  LQueueCreateInfo: VkDeviceQueueCreateInfo;
  LDeviceCreateInfo: VkDeviceCreateInfo;
begin
  LQueuePriority := 1.0;

  FillChar(LQueueCreateInfo, SizeOf(LQueueCreateInfo), 0);
  LQueueCreateInfo.sType := VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  LQueueCreateInfo.queueFamilyIndex := FComputeQueueFamily;
  LQueueCreateInfo.queueCount := 1;
  LQueueCreateInfo.pQueuePriorities := @LQueuePriority;

  FillChar(LDeviceCreateInfo, SizeOf(LDeviceCreateInfo), 0);
  LDeviceCreateInfo.sType := VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  LDeviceCreateInfo.queueCreateInfoCount := 1;
  LDeviceCreateInfo.pQueueCreateInfos := @LQueueCreateInfo;

  if not CheckVk(FvkCreateDevice(FPhysicalDevice, LDeviceCreateInfo, nil, FDevice), 'vkCreateDevice') then Exit;
  // Note: FvkGetDeviceQueue is loaded in LoadDeviceFunctions, called separately after this
end;

procedure TVdxCompute.CreateCommandResources();
var
  LPoolInfo: VkCommandPoolCreateInfo;
  LAllocInfo: VkCommandBufferAllocateInfo;
  LFenceInfo: VkFenceCreateInfo;
begin
  // Command pool
  FillChar(LPoolInfo, SizeOf(LPoolInfo), 0);
  LPoolInfo.sType := VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  LPoolInfo.flags := VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  LPoolInfo.queueFamilyIndex := FComputeQueueFamily;
  if not CheckVk(FvkCreateCommandPool(FDevice, LPoolInfo, nil, FCommandPool), 'vkCreateCommandPool') then Exit;

  // Command buffer
  FillChar(LAllocInfo, SizeOf(LAllocInfo), 0);
  LAllocInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  LAllocInfo.commandPool := FCommandPool;
  LAllocInfo.level := VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  LAllocInfo.commandBufferCount := 1;
  if not CheckVk(FvkAllocateCommandBuffers(FDevice, LAllocInfo, @FCommandBuffer), 'vkAllocateCommandBuffers') then Exit;

  // Fence for synchronization
  FillChar(LFenceInfo, SizeOf(LFenceInfo), 0);
  LFenceInfo.sType := VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if not CheckVk(FvkCreateFence(FDevice, LFenceInfo, nil, FFence), 'vkCreateFence') then Exit;
end;

function TVdxCompute.FindMemoryType(const ATypeBits: UInt32; const AProperties: VkFlags): UInt32;
var
  LI: UInt32;
begin
  Result := 0;
  for LI := 0 to FMemoryProperties.memoryTypeCount - 1 do
  begin
    if ((ATypeBits and (1 shl LI)) <> 0) and
       ((FMemoryProperties.memoryTypes[LI].propertyFlags and AProperties) = AProperties) then
      Exit(LI);
  end;

  FErrors.Add(esError, 'VULKAN_MEM', 'No suitable memory type found (bits=$%x, props=$%x)', [ATypeBits, AProperties]);
end;

function TVdxCompute.CreateGpuBuffer(const ASize: VkDeviceSize; const AUsage: VkFlags; const AMemProps: VkFlags): TVdxGpuBuffer;
var
  LBufInfo: VkBufferCreateInfo;
  LMemReqs: VkMemoryRequirements;
  LAllocInfo: VkMemoryAllocateInfo;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.Size := ASize;

  // Create buffer
  FillChar(LBufInfo, SizeOf(LBufInfo), 0);
  LBufInfo.sType := VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  LBufInfo.size := ASize;
  LBufInfo.usage := AUsage;
  LBufInfo.sharingMode := VK_SHARING_MODE_EXCLUSIVE;
  if not CheckVk(FvkCreateBuffer(FDevice, LBufInfo, nil, Result.Buffer), 'vkCreateBuffer') then Exit;

  // Get memory requirements
  FvkGetBufferMemoryRequirements(FDevice, Result.Buffer, LMemReqs);

  // Allocate memory
  FillChar(LAllocInfo, SizeOf(LAllocInfo), 0);
  LAllocInfo.sType := VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  LAllocInfo.allocationSize := LMemReqs.size;
  LAllocInfo.memoryTypeIndex := FindMemoryType(LMemReqs.memoryTypeBits, AMemProps);
  if not CheckVk(FvkAllocateMemory(FDevice, LAllocInfo, nil, Result.Memory), 'vkAllocateMemory') then Exit;

  // Bind buffer to memory
  if not CheckVk(FvkBindBufferMemory(FDevice, Result.Buffer, Result.Memory, 0), 'vkBindBufferMemory') then Exit;
end;

procedure TVdxCompute.DestroyGpuBuffer(var ABuffer: TVdxGpuBuffer);
begin
  if ABuffer.Buffer <> VK_NULL_HANDLE then
    FvkDestroyBuffer(FDevice, ABuffer.Buffer, nil);

  if ABuffer.Memory <> VK_NULL_HANDLE then
    FvkFreeMemory(FDevice, ABuffer.Memory, nil);

  FillChar(ABuffer, SizeOf(ABuffer), 0);
end;

procedure TVdxCompute.UploadToBuffer(const ABuffer: TVdxGpuBuffer; const AData: Pointer; const ASize: VkDeviceSize);
var
  LMapped: Pointer;
begin
  if not CheckVk(FvkMapMemory(FDevice, ABuffer.Memory, 0, ASize, 0, LMapped), 'vkMapMemory') then Exit;
  Move(AData^, LMapped^, ASize);
  FvkUnmapMemory(FDevice, ABuffer.Memory);
end;

procedure TVdxCompute.DownloadFromBuffer(const ABuffer: TVdxGpuBuffer; const AData: Pointer; const ASize: VkDeviceSize);
var
  LMapped: Pointer;
begin
  if not CheckVk(FvkMapMemory(FDevice, ABuffer.Memory, 0, ASize, 0, LMapped), 'vkMapMemory') then Exit;
  Move(LMapped^, AData^, ASize);
  FvkUnmapMemory(FDevice, ABuffer.Memory);
end;

function TVdxCompute.MapBufferPersistent(const ABuffer: TVdxGpuBuffer): Pointer;
begin
  if not CheckVk(FvkMapMemory(FDevice, ABuffer.Memory, 0, ABuffer.Size, 0, Result), 'vkMapMemory(persistent)') then Exit;
end;

procedure TVdxCompute.UnmapBuffer(const ABuffer: TVdxGpuBuffer);
begin
  FvkUnmapMemory(FDevice, ABuffer.Memory);
end;

function TVdxCompute.CreateShaderModule(const ACode: Pointer; const ACodeSize: NativeUInt): VkShaderModule;
var
  LCreateInfo: VkShaderModuleCreateInfo;
begin
  FillChar(LCreateInfo, SizeOf(LCreateInfo), 0);
  LCreateInfo.sType := VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  LCreateInfo.codeSize := ACodeSize;
  LCreateInfo.pCode := ACode;
  if not CheckVk(FvkCreateShaderModule(FDevice, LCreateInfo, nil, Result), 'vkCreateShaderModule') then Exit;
end;

procedure TVdxCompute.DestroyShaderModuleHandle(const AModule: VkShaderModule);
begin
  if AModule <> VK_NULL_HANDLE then
    FvkDestroyShaderModule(FDevice, AModule, nil);
end;

function TVdxCompute.CreateComputePipelineSimple(const AShaderModule: VkShaderModule; const AEntryPoint: PAnsiChar; const ADescSetLayout: VkDescriptorSetLayout): TVdxComputePipelineBundle;
var
  LLayoutInfo: VkPipelineLayoutCreateInfo;
  LStageInfo: VkPipelineShaderStageCreateInfo;
  LPipelineInfo: VkComputePipelineCreateInfo;
begin
  FillChar(Result, SizeOf(Result), 0);

  // Pipeline layout
  FillChar(LLayoutInfo, SizeOf(LLayoutInfo), 0);
  LLayoutInfo.sType := VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  LLayoutInfo.setLayoutCount := 1;
  LLayoutInfo.pSetLayouts := @ADescSetLayout;
  if not CheckVk(FvkCreatePipelineLayout(FDevice, LLayoutInfo, nil, Result.PipelineLayout), 'vkCreatePipelineLayout') then Exit;

  // Shader stage
  FillChar(LStageInfo, SizeOf(LStageInfo), 0);
  LStageInfo.sType := VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  LStageInfo.stage := VK_SHADER_STAGE_COMPUTE_BIT;
  LStageInfo.module := AShaderModule;
  LStageInfo.pName := AEntryPoint;

  // Compute pipeline
  FillChar(LPipelineInfo, SizeOf(LPipelineInfo), 0);
  LPipelineInfo.sType := VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  LPipelineInfo.stage := LStageInfo;
  LPipelineInfo.layout := Result.PipelineLayout;
  if not CheckVk(FvkCreateComputePipelines(FDevice, VK_NULL_HANDLE, 1, LPipelineInfo, nil, @Result.Pipeline), 'vkCreateComputePipelines') then Exit;
end;

function TVdxCompute.CreateComputePipelineWithPush(const AShaderModule: VkShaderModule; const AEntryPoint: PAnsiChar; const ADescSetLayout: VkDescriptorSetLayout; const APushSize: UInt32): TVdxComputePipelineBundle;
var
  LPushRange: VkPushConstantRange;
  LLayoutInfo: VkPipelineLayoutCreateInfo;
  LStageInfo: VkPipelineShaderStageCreateInfo;
  LPipelineInfo: VkComputePipelineCreateInfo;
begin
  FillChar(Result, SizeOf(Result), 0);

  // Push constant range for compute stage
  FillChar(LPushRange, SizeOf(LPushRange), 0);
  LPushRange.stageFlags := VK_SHADER_STAGE_COMPUTE_BIT;
  LPushRange.offset := 0;
  LPushRange.size := APushSize;

  // Pipeline layout with push constants
  FillChar(LLayoutInfo, SizeOf(LLayoutInfo), 0);
  LLayoutInfo.sType := VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  LLayoutInfo.setLayoutCount := 1;
  LLayoutInfo.pSetLayouts := @ADescSetLayout;
  LLayoutInfo.pushConstantRangeCount := 1;
  LLayoutInfo.pPushConstantRanges := @LPushRange;
  if not CheckVk(FvkCreatePipelineLayout(FDevice, LLayoutInfo, nil, Result.PipelineLayout), 'vkCreatePipelineLayout') then Exit;

  // Shader stage
  FillChar(LStageInfo, SizeOf(LStageInfo), 0);
  LStageInfo.sType := VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  LStageInfo.stage := VK_SHADER_STAGE_COMPUTE_BIT;
  LStageInfo.module := AShaderModule;
  LStageInfo.pName := AEntryPoint;

  // Compute pipeline
  FillChar(LPipelineInfo, SizeOf(LPipelineInfo), 0);
  LPipelineInfo.sType := VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  LPipelineInfo.stage := LStageInfo;
  LPipelineInfo.layout := Result.PipelineLayout;
  if not CheckVk(FvkCreateComputePipelines(FDevice, VK_NULL_HANDLE, 1, LPipelineInfo, nil, @Result.Pipeline), 'vkCreateComputePipelines') then Exit;
end;

function TVdxCompute.CreateComputePipelineWithPushAndSpec(
  const AShaderModule: VkShaderModule;
  const AEntryPoint: PAnsiChar;
  const ADescSetLayout: VkDescriptorSetLayout;
  const APushSize: UInt32;
  const ASpecValue: UInt32): TVdxComputePipelineBundle;
var
  LPushRange: VkPushConstantRange;
  LLayoutInfo: VkPipelineLayoutCreateInfo;
  LStageInfo: VkPipelineShaderStageCreateInfo;
  LPipelineInfo: VkComputePipelineCreateInfo;
  LSpecEntry: VkSpecializationMapEntry;
  LSpecInfo: VkSpecializationInfo;
begin
  Result := Default(TVdxComputePipelineBundle);

  // Specialization constant: single UInt32 at constant_id = 0
  LSpecEntry := Default(VkSpecializationMapEntry);
  LSpecEntry.constantID := 0;
  LSpecEntry.offset := 0;
  LSpecEntry.size := SizeOf(UInt32);

  LSpecInfo := Default(VkSpecializationInfo);
  LSpecInfo.mapEntryCount := 1;
  LSpecInfo.pMapEntries := @LSpecEntry;
  LSpecInfo.dataSize := SizeOf(UInt32);
  LSpecInfo.pData := @ASpecValue;

  // Push constant range for compute stage
  LPushRange := Default(VkPushConstantRange);
  LPushRange.stageFlags := VK_SHADER_STAGE_COMPUTE_BIT;
  LPushRange.offset := 0;
  LPushRange.size := APushSize;

  // Pipeline layout with push constants
  LLayoutInfo := Default(VkPipelineLayoutCreateInfo);
  LLayoutInfo.sType := VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  LLayoutInfo.setLayoutCount := 1;
  LLayoutInfo.pSetLayouts := @ADescSetLayout;
  LLayoutInfo.pushConstantRangeCount := 1;
  LLayoutInfo.pPushConstantRanges := @LPushRange;
  if not CheckVk(FvkCreatePipelineLayout(FDevice, LLayoutInfo, nil, Result.PipelineLayout), 'vkCreatePipelineLayout') then Exit;

  // Shader stage with specialization constant
  LStageInfo := Default(VkPipelineShaderStageCreateInfo);
  LStageInfo.sType := VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  LStageInfo.stage := VK_SHADER_STAGE_COMPUTE_BIT;
  LStageInfo.module := AShaderModule;
  LStageInfo.pName := AEntryPoint;
  LStageInfo.pSpecializationInfo := @LSpecInfo;

  // Compute pipeline
  LPipelineInfo := Default(VkComputePipelineCreateInfo);
  LPipelineInfo.sType := VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  LPipelineInfo.stage := LStageInfo;
  LPipelineInfo.layout := Result.PipelineLayout;
  if not CheckVk(FvkCreateComputePipelines(FDevice, VK_NULL_HANDLE, 1, LPipelineInfo, nil, @Result.Pipeline), 'vkCreateComputePipelines') then Exit;
end;

procedure TVdxCompute.DestroyComputePipelineBundle(var ABundle: TVdxComputePipelineBundle);
begin
  if ABundle.Pipeline <> VK_NULL_HANDLE then
    FvkDestroyPipeline(FDevice, ABundle.Pipeline, nil);

  if ABundle.PipelineLayout <> VK_NULL_HANDLE then
    FvkDestroyPipelineLayout(FDevice, ABundle.PipelineLayout, nil);

  FillChar(ABundle, SizeOf(ABundle), 0);
end;

function TVdxCompute.CreateStorageDescriptorSetLayout(const ABindingCount: UInt32): VkDescriptorSetLayout;
var
  LBindings: array of VkDescriptorSetLayoutBinding;
  LCreateInfo: VkDescriptorSetLayoutCreateInfo;
  LI: UInt32;
begin
  SetLength(LBindings, ABindingCount);

  for LI := 0 to ABindingCount - 1 do
  begin
    FillChar(LBindings[LI], SizeOf(VkDescriptorSetLayoutBinding), 0);
    LBindings[LI].binding := LI;
    LBindings[LI].descriptorType := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    LBindings[LI].descriptorCount := 1;
    LBindings[LI].stageFlags := VK_SHADER_STAGE_COMPUTE_BIT;
  end;

  FillChar(LCreateInfo, SizeOf(LCreateInfo), 0);
  LCreateInfo.sType := VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  LCreateInfo.bindingCount := ABindingCount;
  LCreateInfo.pBindings := @LBindings[0];

  if not CheckVk(FvkCreateDescriptorSetLayout(FDevice, LCreateInfo, nil, Result), 'vkCreateDescriptorSetLayout') then Exit;
end;

function TVdxCompute.CreateDescriptorPoolForStorage(const AMaxSets: UInt32; const AMaxDescriptors: UInt32): VkDescriptorPool;
var
  LPoolSize: VkDescriptorPoolSize;
  LCreateInfo: VkDescriptorPoolCreateInfo;
begin
  LPoolSize.descriptorType := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  LPoolSize.descriptorCount := AMaxDescriptors;

  FillChar(LCreateInfo, SizeOf(LCreateInfo), 0);
  LCreateInfo.sType := VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  LCreateInfo.maxSets := AMaxSets;
  LCreateInfo.poolSizeCount := 1;
  LCreateInfo.pPoolSizes := @LPoolSize;

  if not CheckVk(FvkCreateDescriptorPool(FDevice, LCreateInfo, nil, Result), 'vkCreateDescriptorPool') then Exit;
end;

function TVdxCompute.AllocateDescriptorSetForBuffers(const APool: VkDescriptorPool; const ALayout: VkDescriptorSetLayout; const ABuffers: array of TVdxGpuBuffer): VkDescriptorSet;
var
  LAllocInfo: VkDescriptorSetAllocateInfo;
  LBufferInfos: array of VkDescriptorBufferInfo;
  LWrites: array of VkWriteDescriptorSet;
  LI: Integer;
begin
  // Allocate the set
  FillChar(LAllocInfo, SizeOf(LAllocInfo), 0);
  LAllocInfo.sType := VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  LAllocInfo.descriptorPool := APool;
  LAllocInfo.descriptorSetCount := 1;
  LAllocInfo.pSetLayouts := @ALayout;
  if not CheckVk(FvkAllocateDescriptorSets(FDevice, LAllocInfo, @Result), 'vkAllocateDescriptorSets') then Exit;

  // Bind buffers to descriptor set
  SetLength(LBufferInfos, Length(ABuffers));
  SetLength(LWrites, Length(ABuffers));

  for LI := 0 to High(ABuffers) do
  begin
    LBufferInfos[LI].buffer := ABuffers[LI].Buffer;
    LBufferInfos[LI].offset := 0;
    LBufferInfos[LI].range := VK_WHOLE_SIZE;

    FillChar(LWrites[LI], SizeOf(VkWriteDescriptorSet), 0);
    LWrites[LI].sType := VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    LWrites[LI].dstSet := Result;
    LWrites[LI].dstBinding := UInt32(LI);
    LWrites[LI].descriptorCount := 1;
    LWrites[LI].descriptorType := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    LWrites[LI].pBufferInfo := @LBufferInfos[LI];
  end;

  FvkUpdateDescriptorSets(FDevice, UInt32(Length(LWrites)), @LWrites[0], 0, nil);
end;

procedure TVdxCompute.UpdateDescriptorSetBuffers(const ADescSet: VkDescriptorSet; const ABuffers: array of TVdxGpuBuffer);
var
  LBufferInfos: array of VkDescriptorBufferInfo;
  LWrites: array of VkWriteDescriptorSet;
  LI: Integer;
begin
  SetLength(LBufferInfos, Length(ABuffers));
  SetLength(LWrites, Length(ABuffers));

  for LI := 0 to High(ABuffers) do
  begin
    LBufferInfos[LI].buffer := ABuffers[LI].Buffer;
    LBufferInfos[LI].offset := 0;
    LBufferInfos[LI].range := VK_WHOLE_SIZE;

    FillChar(LWrites[LI], SizeOf(VkWriteDescriptorSet), 0);
    LWrites[LI].sType := VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    LWrites[LI].dstSet := ADescSet;
    LWrites[LI].dstBinding := UInt32(LI);
    LWrites[LI].descriptorCount := 1;
    LWrites[LI].descriptorType := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    LWrites[LI].pBufferInfo := @LBufferInfos[LI];
  end;

  FvkUpdateDescriptorSets(FDevice, UInt32(Length(LWrites)), @LWrites[0], 0, nil);
end;

procedure TVdxCompute.DispatchCompute(const APipeline: VkPipeline; const APipelineLayout: VkPipelineLayout; const ADescSet: VkDescriptorSet; const AGroupsX: UInt32; const AGroupsY: UInt32; const AGroupsZ: UInt32);
var
  LBeginInfo: VkCommandBufferBeginInfo;
  LSubmitInfo: VkSubmitInfo;
begin
  if not FBatchMode then
  begin
    // Non-batch: full begin → record → end → submit → fence cycle
    FillChar(LBeginInfo, SizeOf(LBeginInfo), 0);
    LBeginInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    LBeginInfo.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if not CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo), 'vkBeginCommandBuffer') then Exit;
  end;

  // Record dispatch commands
  FvkCmdBindPipeline(FCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, APipeline);
  FvkCmdBindDescriptorSets(FCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, APipelineLayout, 0, 1, @ADescSet, 0, nil);
  FvkCmdDispatch(FCommandBuffer, AGroupsX, AGroupsY, AGroupsZ);

  if not FBatchMode then
  begin
    // Non-batch: end + submit + fence
    if not CheckVk(FvkEndCommandBuffer(FCommandBuffer), 'vkEndCommandBuffer') then Exit;
    FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
    LSubmitInfo.sType := VK_STRUCTURE_TYPE_SUBMIT_INFO;
    LSubmitInfo.commandBufferCount := 1;
    LSubmitInfo.pCommandBuffers := @FCommandBuffer;
    if not CheckVk(FvkResetFences(FDevice, 1, @FFence), 'vkResetFences') then Exit;
    if not CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence), 'vkQueueSubmit') then Exit;
    if not CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)), 'vkWaitForFences') then Exit;
  end;
  // Batch mode: caller is responsible for calling BatchBarrier() where needed
end;

procedure TVdxCompute.DispatchComputeWithPush(const APipeline: VkPipeline; const APipelineLayout: VkPipelineLayout; const ADescSet: VkDescriptorSet; const APushData: Pointer; const APushSize: UInt32; const AGroupsX: UInt32; const AGroupsY: UInt32; const AGroupsZ: UInt32);
var
  LBeginInfo: VkCommandBufferBeginInfo;
  LSubmitInfo: VkSubmitInfo;
begin
  if not FBatchMode then
  begin
    FillChar(LBeginInfo, SizeOf(LBeginInfo), 0);
    LBeginInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    LBeginInfo.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if not CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo), 'vkBeginCommandBuffer') then Exit;
  end;

  FvkCmdBindPipeline(FCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, APipeline);
  FvkCmdBindDescriptorSets(FCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, APipelineLayout, 0, 1, @ADescSet, 0, nil);
  FvkCmdPushConstants(FCommandBuffer, APipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, APushSize, APushData);
  FvkCmdDispatch(FCommandBuffer, AGroupsX, AGroupsY, AGroupsZ);

  if not FBatchMode then
  begin
    if not CheckVk(FvkEndCommandBuffer(FCommandBuffer), 'vkEndCommandBuffer') then Exit;
    FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
    LSubmitInfo.sType := VK_STRUCTURE_TYPE_SUBMIT_INFO;
    LSubmitInfo.commandBufferCount := 1;
    LSubmitInfo.pCommandBuffers := @FCommandBuffer;
    if not CheckVk(FvkResetFences(FDevice, 1, @FFence), 'vkResetFences') then Exit;
    if not CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence), 'vkQueueSubmit') then Exit;
    if not CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)), 'vkWaitForFences') then Exit;
  end;
  // Batch mode: caller is responsible for calling BatchBarrier() where needed
end;

procedure TVdxCompute.CopyBuffer(const ASrc: TVdxGpuBuffer; const ADst: TVdxGpuBuffer; const ASize: VkDeviceSize);
var
  LBeginInfo: VkCommandBufferBeginInfo;
  LCopyRegion: VkBufferCopy;
  LSubmitInfo: VkSubmitInfo;
begin
  if not FBatchMode then
  begin
    FillChar(LBeginInfo, SizeOf(LBeginInfo), 0);
    LBeginInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    LBeginInfo.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if not CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo), 'vkBeginCommandBuffer') then Exit;
  end;

  FillChar(LCopyRegion, SizeOf(LCopyRegion), 0);
  LCopyRegion.srcOffset := 0;
  LCopyRegion.dstOffset := 0;
  LCopyRegion.size := ASize;
  FvkCmdCopyBuffer(FCommandBuffer, ASrc.Buffer, ADst.Buffer, 1, LCopyRegion);

  if not FBatchMode then
  begin
    if not CheckVk(FvkEndCommandBuffer(FCommandBuffer), 'vkEndCommandBuffer') then Exit;
    FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
    LSubmitInfo.sType := VK_STRUCTURE_TYPE_SUBMIT_INFO;
    LSubmitInfo.commandBufferCount := 1;
    LSubmitInfo.pCommandBuffers := @FCommandBuffer;
    if not CheckVk(FvkResetFences(FDevice, 1, @FFence), 'vkResetFences') then Exit;
    if not CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence), 'vkQueueSubmit') then Exit;
    if not CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)), 'vkWaitForFences') then Exit;
  end;
  // Batch mode: caller is responsible for calling BatchBarrier() where needed
end;

procedure TVdxCompute.CopyBufferRegion(const ASrc: TVdxGpuBuffer;
  const ASrcOffset: VkDeviceSize; const ADst: TVdxGpuBuffer;
  const ADstOffset: VkDeviceSize; const ASize: VkDeviceSize);
var
  LBeginInfo: VkCommandBufferBeginInfo;
  LCopyRegion: VkBufferCopy;
  LSubmitInfo: VkSubmitInfo;
begin
  if not FBatchMode then
  begin
    FillChar(LBeginInfo, SizeOf(LBeginInfo), 0);
    LBeginInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    LBeginInfo.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if not CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo), 'vkBeginCommandBuffer') then Exit;
  end;

  FillChar(LCopyRegion, SizeOf(LCopyRegion), 0);
  LCopyRegion.srcOffset := ASrcOffset;
  LCopyRegion.dstOffset := ADstOffset;
  LCopyRegion.size := ASize;
  FvkCmdCopyBuffer(FCommandBuffer, ASrc.Buffer, ADst.Buffer, 1, LCopyRegion);

  if not FBatchMode then
  begin
    if not CheckVk(FvkEndCommandBuffer(FCommandBuffer), 'vkEndCommandBuffer') then Exit;
    FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
    LSubmitInfo.sType := VK_STRUCTURE_TYPE_SUBMIT_INFO;
    LSubmitInfo.commandBufferCount := 1;
    LSubmitInfo.pCommandBuffers := @FCommandBuffer;
    if not CheckVk(FvkResetFences(FDevice, 1, @FFence), 'vkResetFences') then Exit;
    if not CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence), 'vkQueueSubmit') then Exit;
    if not CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)), 'vkWaitForFences') then Exit;
  end;
  // Batch mode: caller is responsible for calling BatchBarrier() where needed
end;

procedure TVdxCompute.InsertBatchBarrier();
var
  LBarrier: VkMemoryBarrier;
begin
  // Full memory barrier covering compute→compute and transfer→compute
  FillChar(LBarrier, SizeOf(LBarrier), 0);
  LBarrier.sType := VK_STRUCTURE_TYPE_MEMORY_BARRIER;
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
  if FBatchMode then
  begin
    FErrors.Add(esError, 'VULKAN_BATCH', 'BeginBatch called while already in batch mode');
    Exit;
  end;

  FillChar(LBeginInfo, SizeOf(LBeginInfo), 0);
  LBeginInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  LBeginInfo.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if not CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo), 'vkBeginCommandBuffer(batch)') then Exit;

  FBatchMode := True;
  FBatchDeferredPoolCount := 0;
end;

procedure TVdxCompute.EndBatch();
var
  LSubmitInfo: VkSubmitInfo;
  LI: Integer;
begin
  if not FBatchMode then
  begin
    FErrors.Add(esError, 'VULKAN_BATCH', 'EndBatch called without BeginBatch');
    Exit;
  end;

  // End command buffer recording
  if not CheckVk(FvkEndCommandBuffer(FCommandBuffer), 'vkEndCommandBuffer(batch)') then Exit;

  // Submit once and wait once
  FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
  LSubmitInfo.sType := VK_STRUCTURE_TYPE_SUBMIT_INFO;
  LSubmitInfo.commandBufferCount := 1;
  LSubmitInfo.pCommandBuffers := @FCommandBuffer;

  if not CheckVk(FvkResetFences(FDevice, 1, @FFence), 'vkResetFences(batch)') then Exit;
  if not CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence), 'vkQueueSubmit(batch)') then Exit;
  if not CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)), 'vkWaitForFences(batch)') then Exit;

  // Destroy deferred descriptor pools now that GPU work is complete
  for LI := 0 to FBatchDeferredPoolCount - 1 do
    FvkDestroyDescriptorPool(FDevice, FBatchDeferredPools[LI], nil);
  FBatchDeferredPoolCount := 0;

  FBatchMode := False;
end;

procedure TVdxCompute.DestroyDescriptorSetLayoutHandle(const ALayout: VkDescriptorSetLayout);
begin
  if ALayout <> VK_NULL_HANDLE then
    FvkDestroyDescriptorSetLayout(FDevice, ALayout, nil);
end;

procedure TVdxCompute.DestroyDescriptorPoolHandle(const APool: VkDescriptorPool);
begin
  if APool = VK_NULL_HANDLE then
    Exit;

  if FBatchMode then
  begin
    // Defer destruction until EndBatch — descriptor sets must stay valid
    if FBatchDeferredPoolCount >= Length(FBatchDeferredPools) then
      SetLength(FBatchDeferredPools, FBatchDeferredPoolCount + 64);
    FBatchDeferredPools[FBatchDeferredPoolCount] := APool;
    Inc(FBatchDeferredPoolCount);
  end
  else
    FvkDestroyDescriptorPool(FDevice, APool, nil);
end;

function TVdxCompute.EnumerateGpus(out AGpus: TArray<TVdxGpuInfo>): Boolean;
begin
  Result := False;

  // Boot enough Vulkan to enumerate if not done yet
  if FLibHandle = 0 then
  begin
    LoadVulkanLibrary();
    if FErrors.HasErrors() then Exit;
    LoadGlobalFunctions();
    if FErrors.HasErrors() then Exit;
  end;
  if FInstance = nil then
  begin
    CreateVkInstance();
    if FErrors.HasErrors() then Exit;
    LoadInstanceFunctions();
  end;

  if Length(FGpusCache) = 0 then
  begin
    PopulateGpuCache();
    if FErrors.HasErrors() then Exit;
  end;

  AGpus := FGpusCache;
  Result := True;
end;

function TVdxCompute.GetSelectedGpuIndex(): Integer;
begin
  Result := FSelectedGpuIndex;
end;

function TVdxCompute.GetDeviceName(): string;
begin
  Result := string(AnsiString(PAnsiChar(@FDeviceProperties.deviceName[0])));
end;

function TVdxCompute.GetVRAMSizeMB(): UInt64;
var
  LI: UInt32;
  LLargest: UInt64;
begin
  // Find the largest device-local heap
  LLargest := 0;
  for LI := 0 to FMemoryProperties.memoryHeapCount - 1 do
  begin
    if (FMemoryProperties.memoryHeaps[LI].flags and $00000001) <> 0 then // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
    begin
      if FMemoryProperties.memoryHeaps[LI].size > LLargest then
        LLargest := FMemoryProperties.memoryHeaps[LI].size;
    end;
  end;

  Result := LLargest div (1024 * 1024);
end;

function TVdxCompute.GetMaxComputeWorkGroupSize(): UInt32;
begin
  Result := FDeviceProperties.limits.maxComputeWorkGroupInvocations;
end;

end.
