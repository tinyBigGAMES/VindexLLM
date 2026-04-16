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
  TVdxGpuBuffer = record
    Buffer: VkBuffer;
    Memory: VkDeviceMemory;
    Size:   VkDeviceSize;
  end;

// ============================================================================
//  TVdxComputePipelineBundle — Groups pipeline + layout for easy cleanup
// ============================================================================

  TVdxComputePipelineBundle = record
    Pipeline:       VkPipeline;
    PipelineLayout: VkPipelineLayout;
  end;

// ============================================================================
//  TVdxVulkanCompute — Vulkan compute device manager
// ============================================================================

type
  TVdxVulkanCompute = class(TVdxStatusObject)
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
    procedure SelectPhysicalDevice();
    procedure CreateLogicalDevice();
    procedure CreateCommandResources();
    procedure InsertBatchBarrier();
    function  FindMemoryType(const ATypeBits: UInt32; const AProperties: VkFlags): UInt32;
    function  GetVkProc(const AName: PAnsiChar): Pointer;
    procedure CheckVk(const AResult: VkResult; const AContext: string);

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Call after setting status callback to initialize Vulkan
    procedure Init();

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
    function  GetDeviceName(): string;
    function  GetVRAMSizeMB(): UInt64;
    function  GetMaxComputeWorkGroupSize(): UInt32;
  end;

implementation

// ============================================================================
//  TVdxVulkanCompute — Construction / Destruction
// ============================================================================

constructor TVdxVulkanCompute.Create();
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
end;

procedure TVdxVulkanCompute.Init();
begin
  Status('Loading Vulkan...');
  LoadVulkanLibrary();
  LoadGlobalFunctions();

  Status('Creating Vulkan instance...');
  CreateVkInstance();
  LoadInstanceFunctions();

  Status('Selecting GPU...');
  SelectPhysicalDevice();

  Status('Creating logical device...');
  CreateLogicalDevice();
  LoadDeviceFunctions();
  FvkGetDeviceQueue(FDevice, FComputeQueueFamily, 0, FComputeQueue);

  Status('Creating command resources...');
  CreateCommandResources();

  Status('Vulkan ready: %s (%d MB VRAM)', [GetDeviceName(), GetVRAMSizeMB()]);
end;

destructor TVdxVulkanCompute.Destroy();
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

// ============================================================================
//  Library + Function Loading
// ============================================================================

procedure TVdxVulkanCompute.LoadVulkanLibrary();
begin
  FLibHandle := LoadLibrary('vulkan-1.dll');
  if FLibHandle = 0 then
    raise Exception.Create('VulkanCompute: Failed to load vulkan-1.dll — no Vulkan driver installed');

  FvkGetInstanceProcAddr := TvkGetInstanceProcAddr(GetProcAddress(FLibHandle, 'vkGetInstanceProcAddr'));
  if not Assigned(FvkGetInstanceProcAddr) then
    raise Exception.Create('VulkanCompute: vkGetInstanceProcAddr not found in vulkan-1.dll');
end;

function TVdxVulkanCompute.GetVkProc(const AName: PAnsiChar): Pointer;
begin
  // Try instance-level first, then fall back to GetProcAddress on the DLL
  if FInstance <> nil then
    Result := FvkGetInstanceProcAddr(FInstance, AName)
  else
    Result := FvkGetInstanceProcAddr(nil, AName);

  if Result = nil then
    Result := GetProcAddress(FLibHandle, AName);
end;

procedure TVdxVulkanCompute.LoadGlobalFunctions();
begin
  @FvkCreateInstance := GetVkProc('vkCreateInstance');
  TVdxUtils.FailIf(not Assigned(FvkCreateInstance), 'VulkanCompute: Failed to load %s', ['vkCreateInstance']);
end;

procedure TVdxVulkanCompute.LoadInstanceFunctions();
begin
  @FvkDestroyInstance := GetVkProc('vkDestroyInstance');
  @FvkEnumeratePhysicalDevices := GetVkProc('vkEnumeratePhysicalDevices');
  @FvkGetPhysicalDeviceProperties := GetVkProc('vkGetPhysicalDeviceProperties');
  @FvkGetPhysicalDeviceQueueFamilyProperties := GetVkProc('vkGetPhysicalDeviceQueueFamilyProperties');
  @FvkGetPhysicalDeviceMemoryProperties := GetVkProc('vkGetPhysicalDeviceMemoryProperties');
  @FvkCreateDevice := GetVkProc('vkCreateDevice');
end;

procedure TVdxVulkanCompute.LoadDeviceFunctions();
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

// ============================================================================
//  Vulkan Initialization
// ============================================================================

procedure TVdxVulkanCompute.CheckVk(const AResult: VkResult; const AContext: string);
begin
  if AResult <> VK_SUCCESS then
    raise Exception.CreateFmt('VulkanCompute: %s failed (VkResult=%d)', [AContext, AResult]);
end;

procedure TVdxVulkanCompute.CreateVkInstance();
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

  CheckVk(FvkCreateInstance(LCreateInfo, nil, FInstance), 'vkCreateInstance');
end;

procedure TVdxVulkanCompute.SelectPhysicalDevice();
var
  LDeviceCount: UInt32;
  LDevices: array of VkPhysicalDevice;
  LProps: VkPhysicalDeviceProperties;
  LFamilyCount: UInt32;
  LFamilies: array of VkQueueFamilyProperties;
  LI: Integer;
  LJ: Integer;
  LFound: Boolean;
begin
  LDeviceCount := 0;
  CheckVk(FvkEnumeratePhysicalDevices(FInstance, LDeviceCount, nil), 'vkEnumeratePhysicalDevices(count)');
  if LDeviceCount = 0 then
    raise Exception.Create('VulkanCompute: No Vulkan-capable GPU found');

  SetLength(LDevices, LDeviceCount);
  CheckVk(FvkEnumeratePhysicalDevices(FInstance, LDeviceCount, @LDevices[0]), 'vkEnumeratePhysicalDevices');

  // Prefer discrete GPU, fall back to first with compute queue
  FPhysicalDevice := nil;
  LFound := False;

  for LI := 0 to LDeviceCount - 1 do
  begin
    FvkGetPhysicalDeviceProperties(LDevices[LI], LProps);

    // Find compute queue family
    LFamilyCount := 0;
    FvkGetPhysicalDeviceQueueFamilyProperties(LDevices[LI], LFamilyCount, nil);
    SetLength(LFamilies, LFamilyCount);
    FvkGetPhysicalDeviceQueueFamilyProperties(LDevices[LI], LFamilyCount, @LFamilies[0]);

    for LJ := 0 to LFamilyCount - 1 do
    begin
      if (LFamilies[LJ].queueFlags and VK_QUEUE_COMPUTE_BIT) <> 0 then
      begin
        // Found a compute-capable device
        if (not LFound) or (LProps.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) then
        begin
          FPhysicalDevice := LDevices[LI];
          FComputeQueueFamily := UInt32(LJ);
          FDeviceProperties := LProps;
          LFound := True;
        end;

        Break;
      end;
    end;
  end;

  if not LFound then
    raise Exception.Create('VulkanCompute: No GPU with compute queue found');

  // Cache memory properties
  FvkGetPhysicalDeviceMemoryProperties(FPhysicalDevice, FMemoryProperties);

  Status('Selected GPU: %s (type=%d)', [string(AnsiString(PAnsiChar(@FDeviceProperties.deviceName[0]))), FDeviceProperties.deviceType]);
end;

procedure TVdxVulkanCompute.CreateLogicalDevice();
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

  CheckVk(FvkCreateDevice(FPhysicalDevice, LDeviceCreateInfo, nil, FDevice), 'vkCreateDevice');
  // Note: FvkGetDeviceQueue is loaded in LoadDeviceFunctions, called separately after this
end;

procedure TVdxVulkanCompute.CreateCommandResources();
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
  CheckVk(FvkCreateCommandPool(FDevice, LPoolInfo, nil, FCommandPool), 'vkCreateCommandPool');

  // Command buffer
  FillChar(LAllocInfo, SizeOf(LAllocInfo), 0);
  LAllocInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  LAllocInfo.commandPool := FCommandPool;
  LAllocInfo.level := VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  LAllocInfo.commandBufferCount := 1;
  CheckVk(FvkAllocateCommandBuffers(FDevice, LAllocInfo, @FCommandBuffer), 'vkAllocateCommandBuffers');

  // Fence for synchronization
  FillChar(LFenceInfo, SizeOf(LFenceInfo), 0);
  LFenceInfo.sType := VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  CheckVk(FvkCreateFence(FDevice, LFenceInfo, nil, FFence), 'vkCreateFence');
end;

function TVdxVulkanCompute.FindMemoryType(const ATypeBits: UInt32; const AProperties: VkFlags): UInt32;
var
  LI: UInt32;
begin
  for LI := 0 to FMemoryProperties.memoryTypeCount - 1 do
  begin
    if ((ATypeBits and (1 shl LI)) <> 0) and
       ((FMemoryProperties.memoryTypes[LI].propertyFlags and AProperties) = AProperties) then
      Exit(LI);
  end;

  raise Exception.CreateFmt('VulkanCompute: No suitable memory type found (bits=$%x, props=$%x)', [ATypeBits, AProperties]);
end;

// ============================================================================
//  Buffer Operations
// ============================================================================

function TVdxVulkanCompute.CreateGpuBuffer(const ASize: VkDeviceSize; const AUsage: VkFlags; const AMemProps: VkFlags): TVdxGpuBuffer;
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
  CheckVk(FvkCreateBuffer(FDevice, LBufInfo, nil, Result.Buffer), 'vkCreateBuffer');

  // Get memory requirements
  FvkGetBufferMemoryRequirements(FDevice, Result.Buffer, LMemReqs);

  // Allocate memory
  FillChar(LAllocInfo, SizeOf(LAllocInfo), 0);
  LAllocInfo.sType := VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  LAllocInfo.allocationSize := LMemReqs.size;
  LAllocInfo.memoryTypeIndex := FindMemoryType(LMemReqs.memoryTypeBits, AMemProps);
  CheckVk(FvkAllocateMemory(FDevice, LAllocInfo, nil, Result.Memory), 'vkAllocateMemory');

  // Bind buffer to memory
  CheckVk(FvkBindBufferMemory(FDevice, Result.Buffer, Result.Memory, 0), 'vkBindBufferMemory');
end;

procedure TVdxVulkanCompute.DestroyGpuBuffer(var ABuffer: TVdxGpuBuffer);
begin
  if ABuffer.Buffer <> VK_NULL_HANDLE then
    FvkDestroyBuffer(FDevice, ABuffer.Buffer, nil);

  if ABuffer.Memory <> VK_NULL_HANDLE then
    FvkFreeMemory(FDevice, ABuffer.Memory, nil);

  FillChar(ABuffer, SizeOf(ABuffer), 0);
end;

procedure TVdxVulkanCompute.UploadToBuffer(const ABuffer: TVdxGpuBuffer; const AData: Pointer; const ASize: VkDeviceSize);
var
  LMapped: Pointer;
begin
  CheckVk(FvkMapMemory(FDevice, ABuffer.Memory, 0, ASize, 0, LMapped), 'vkMapMemory');
  Move(AData^, LMapped^, ASize);
  FvkUnmapMemory(FDevice, ABuffer.Memory);
end;

procedure TVdxVulkanCompute.DownloadFromBuffer(const ABuffer: TVdxGpuBuffer; const AData: Pointer; const ASize: VkDeviceSize);
var
  LMapped: Pointer;
begin
  CheckVk(FvkMapMemory(FDevice, ABuffer.Memory, 0, ASize, 0, LMapped), 'vkMapMemory');
  Move(LMapped^, AData^, ASize);
  FvkUnmapMemory(FDevice, ABuffer.Memory);
end;

function TVdxVulkanCompute.MapBufferPersistent(const ABuffer: TVdxGpuBuffer): Pointer;
begin
  CheckVk(FvkMapMemory(FDevice, ABuffer.Memory, 0, ABuffer.Size, 0, Result), 'vkMapMemory(persistent)');
end;

procedure TVdxVulkanCompute.UnmapBuffer(const ABuffer: TVdxGpuBuffer);
begin
  FvkUnmapMemory(FDevice, ABuffer.Memory);
end;

// ============================================================================
//  Shader + Pipeline
// ============================================================================

function TVdxVulkanCompute.CreateShaderModule(const ACode: Pointer; const ACodeSize: NativeUInt): VkShaderModule;
var
  LCreateInfo: VkShaderModuleCreateInfo;
begin
  FillChar(LCreateInfo, SizeOf(LCreateInfo), 0);
  LCreateInfo.sType := VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  LCreateInfo.codeSize := ACodeSize;
  LCreateInfo.pCode := ACode;
  CheckVk(FvkCreateShaderModule(FDevice, LCreateInfo, nil, Result), 'vkCreateShaderModule');
end;

procedure TVdxVulkanCompute.DestroyShaderModuleHandle(const AModule: VkShaderModule);
begin
  if AModule <> VK_NULL_HANDLE then
    FvkDestroyShaderModule(FDevice, AModule, nil);
end;

function TVdxVulkanCompute.CreateComputePipelineSimple(const AShaderModule: VkShaderModule; const AEntryPoint: PAnsiChar; const ADescSetLayout: VkDescriptorSetLayout): TVdxComputePipelineBundle;
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
  CheckVk(FvkCreatePipelineLayout(FDevice, LLayoutInfo, nil, Result.PipelineLayout), 'vkCreatePipelineLayout');

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
  CheckVk(FvkCreateComputePipelines(FDevice, VK_NULL_HANDLE, 1, LPipelineInfo, nil, @Result.Pipeline), 'vkCreateComputePipelines');
end;

function TVdxVulkanCompute.CreateComputePipelineWithPush(const AShaderModule: VkShaderModule; const AEntryPoint: PAnsiChar; const ADescSetLayout: VkDescriptorSetLayout; const APushSize: UInt32): TVdxComputePipelineBundle;
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
  CheckVk(FvkCreatePipelineLayout(FDevice, LLayoutInfo, nil, Result.PipelineLayout), 'vkCreatePipelineLayout');

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
  CheckVk(FvkCreateComputePipelines(FDevice, VK_NULL_HANDLE, 1, LPipelineInfo, nil, @Result.Pipeline), 'vkCreateComputePipelines');
end;

function TVdxVulkanCompute.CreateComputePipelineWithPushAndSpec(
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
  CheckVk(FvkCreatePipelineLayout(FDevice, LLayoutInfo, nil, Result.PipelineLayout), 'vkCreatePipelineLayout');

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
  CheckVk(FvkCreateComputePipelines(FDevice, VK_NULL_HANDLE, 1, LPipelineInfo, nil, @Result.Pipeline), 'vkCreateComputePipelines');
end;

procedure TVdxVulkanCompute.DestroyComputePipelineBundle(var ABundle: TVdxComputePipelineBundle);
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

function TVdxVulkanCompute.CreateStorageDescriptorSetLayout(const ABindingCount: UInt32): VkDescriptorSetLayout;
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

  CheckVk(FvkCreateDescriptorSetLayout(FDevice, LCreateInfo, nil, Result), 'vkCreateDescriptorSetLayout');
end;

function TVdxVulkanCompute.CreateDescriptorPoolForStorage(const AMaxSets: UInt32; const AMaxDescriptors: UInt32): VkDescriptorPool;
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

  CheckVk(FvkCreateDescriptorPool(FDevice, LCreateInfo, nil, Result), 'vkCreateDescriptorPool');
end;

function TVdxVulkanCompute.AllocateDescriptorSetForBuffers(const APool: VkDescriptorPool; const ALayout: VkDescriptorSetLayout; const ABuffers: array of TVdxGpuBuffer): VkDescriptorSet;
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
  CheckVk(FvkAllocateDescriptorSets(FDevice, LAllocInfo, @Result), 'vkAllocateDescriptorSets');

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

// ============================================================================
//  UpdateDescriptorSetBuffers — Rebind buffers to an existing descriptor set
//  Cheap: no allocation, just patches buffer pointers in the driver
// ============================================================================

procedure TVdxVulkanCompute.UpdateDescriptorSetBuffers(const ADescSet: VkDescriptorSet; const ABuffers: array of TVdxGpuBuffer);
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

// ============================================================================
//  Dispatch
// ============================================================================

procedure TVdxVulkanCompute.DispatchCompute(const APipeline: VkPipeline; const APipelineLayout: VkPipelineLayout; const ADescSet: VkDescriptorSet; const AGroupsX: UInt32; const AGroupsY: UInt32; const AGroupsZ: UInt32);
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
    CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo), 'vkBeginCommandBuffer');
  end;

  // Record dispatch commands
  FvkCmdBindPipeline(FCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, APipeline);
  FvkCmdBindDescriptorSets(FCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, APipelineLayout, 0, 1, @ADescSet, 0, nil);
  FvkCmdDispatch(FCommandBuffer, AGroupsX, AGroupsY, AGroupsZ);

  if not FBatchMode then
  begin
    // Non-batch: end + submit + fence
    CheckVk(FvkEndCommandBuffer(FCommandBuffer), 'vkEndCommandBuffer');
    FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
    LSubmitInfo.sType := VK_STRUCTURE_TYPE_SUBMIT_INFO;
    LSubmitInfo.commandBufferCount := 1;
    LSubmitInfo.pCommandBuffers := @FCommandBuffer;
    CheckVk(FvkResetFences(FDevice, 1, @FFence), 'vkResetFences');
    CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence), 'vkQueueSubmit');
    CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)), 'vkWaitForFences');
  end;
  // Batch mode: caller is responsible for calling BatchBarrier() where needed
end;

procedure TVdxVulkanCompute.DispatchComputeWithPush(const APipeline: VkPipeline; const APipelineLayout: VkPipelineLayout; const ADescSet: VkDescriptorSet; const APushData: Pointer; const APushSize: UInt32; const AGroupsX: UInt32; const AGroupsY: UInt32; const AGroupsZ: UInt32);
var
  LBeginInfo: VkCommandBufferBeginInfo;
  LSubmitInfo: VkSubmitInfo;
begin
  if not FBatchMode then
  begin
    FillChar(LBeginInfo, SizeOf(LBeginInfo), 0);
    LBeginInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    LBeginInfo.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo), 'vkBeginCommandBuffer');
  end;

  FvkCmdBindPipeline(FCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, APipeline);
  FvkCmdBindDescriptorSets(FCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, APipelineLayout, 0, 1, @ADescSet, 0, nil);
  FvkCmdPushConstants(FCommandBuffer, APipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, APushSize, APushData);
  FvkCmdDispatch(FCommandBuffer, AGroupsX, AGroupsY, AGroupsZ);

  if not FBatchMode then
  begin
    CheckVk(FvkEndCommandBuffer(FCommandBuffer), 'vkEndCommandBuffer');
    FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
    LSubmitInfo.sType := VK_STRUCTURE_TYPE_SUBMIT_INFO;
    LSubmitInfo.commandBufferCount := 1;
    LSubmitInfo.pCommandBuffers := @FCommandBuffer;
    CheckVk(FvkResetFences(FDevice, 1, @FFence), 'vkResetFences');
    CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence), 'vkQueueSubmit');
    CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)), 'vkWaitForFences');
  end;
  // Batch mode: caller is responsible for calling BatchBarrier() where needed
end;

// ============================================================================
//  Buffer Copy (staging → device-local)
// ============================================================================

procedure TVdxVulkanCompute.CopyBuffer(const ASrc: TVdxGpuBuffer; const ADst: TVdxGpuBuffer; const ASize: VkDeviceSize);
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
    CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo), 'vkBeginCommandBuffer');
  end;

  FillChar(LCopyRegion, SizeOf(LCopyRegion), 0);
  LCopyRegion.srcOffset := 0;
  LCopyRegion.dstOffset := 0;
  LCopyRegion.size := ASize;
  FvkCmdCopyBuffer(FCommandBuffer, ASrc.Buffer, ADst.Buffer, 1, LCopyRegion);

  if not FBatchMode then
  begin
    CheckVk(FvkEndCommandBuffer(FCommandBuffer), 'vkEndCommandBuffer');
    FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
    LSubmitInfo.sType := VK_STRUCTURE_TYPE_SUBMIT_INFO;
    LSubmitInfo.commandBufferCount := 1;
    LSubmitInfo.pCommandBuffers := @FCommandBuffer;
    CheckVk(FvkResetFences(FDevice, 1, @FFence), 'vkResetFences');
    CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence), 'vkQueueSubmit');
    CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)), 'vkWaitForFences');
  end;
  // Batch mode: caller is responsible for calling BatchBarrier() where needed
end;

procedure TVdxVulkanCompute.CopyBufferRegion(const ASrc: TVdxGpuBuffer;
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
    CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo), 'vkBeginCommandBuffer');
  end;

  FillChar(LCopyRegion, SizeOf(LCopyRegion), 0);
  LCopyRegion.srcOffset := ASrcOffset;
  LCopyRegion.dstOffset := ADstOffset;
  LCopyRegion.size := ASize;
  FvkCmdCopyBuffer(FCommandBuffer, ASrc.Buffer, ADst.Buffer, 1, LCopyRegion);

  if not FBatchMode then
  begin
    CheckVk(FvkEndCommandBuffer(FCommandBuffer), 'vkEndCommandBuffer');
    FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
    LSubmitInfo.sType := VK_STRUCTURE_TYPE_SUBMIT_INFO;
    LSubmitInfo.commandBufferCount := 1;
    LSubmitInfo.pCommandBuffers := @FCommandBuffer;
    CheckVk(FvkResetFences(FDevice, 1, @FFence), 'vkResetFences');
    CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence), 'vkQueueSubmit');
    CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)), 'vkWaitForFences');
  end;
  // Batch mode: caller is responsible for calling BatchBarrier() where needed
end;

// ============================================================================
//  Batch Mode — Record multiple dispatches, one submit+fence at the end
// ============================================================================

procedure TVdxVulkanCompute.InsertBatchBarrier();
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

procedure TVdxVulkanCompute.BatchBarrier();
begin
  if FBatchMode then
    InsertBatchBarrier();
end;

procedure TVdxVulkanCompute.BeginBatch();
var
  LBeginInfo: VkCommandBufferBeginInfo;
begin
  TVdxUtils.FailIf(FBatchMode, 'VulkanCompute: BeginBatch called while already in batch mode', []);

  FillChar(LBeginInfo, SizeOf(LBeginInfo), 0);
  LBeginInfo.sType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  LBeginInfo.flags := VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  CheckVk(FvkBeginCommandBuffer(FCommandBuffer, LBeginInfo), 'vkBeginCommandBuffer(batch)');

  FBatchMode := True;
  FBatchDeferredPoolCount := 0;
end;

procedure TVdxVulkanCompute.EndBatch();
var
  LSubmitInfo: VkSubmitInfo;
  LI: Integer;
begin
  TVdxUtils.FailIf(not FBatchMode, 'VulkanCompute: EndBatch called without BeginBatch', []);

  // End command buffer recording
  CheckVk(FvkEndCommandBuffer(FCommandBuffer), 'vkEndCommandBuffer(batch)');

  // Submit once and wait once
  FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
  LSubmitInfo.sType := VK_STRUCTURE_TYPE_SUBMIT_INFO;
  LSubmitInfo.commandBufferCount := 1;
  LSubmitInfo.pCommandBuffers := @FCommandBuffer;

  CheckVk(FvkResetFences(FDevice, 1, @FFence), 'vkResetFences(batch)');
  CheckVk(FvkQueueSubmit(FComputeQueue, 1, LSubmitInfo, FFence), 'vkQueueSubmit(batch)');
  CheckVk(FvkWaitForFences(FDevice, 1, @FFence, VK_TRUE, UInt64($FFFFFFFFFFFFFFFF)), 'vkWaitForFences(batch)');

  // Destroy deferred descriptor pools now that GPU work is complete
  for LI := 0 to FBatchDeferredPoolCount - 1 do
    FvkDestroyDescriptorPool(FDevice, FBatchDeferredPools[LI], nil);
  FBatchDeferredPoolCount := 0;

  FBatchMode := False;
end;

// ============================================================================
//  Cleanup Helpers
// ============================================================================

procedure TVdxVulkanCompute.DestroyDescriptorSetLayoutHandle(const ALayout: VkDescriptorSetLayout);
begin
  if ALayout <> VK_NULL_HANDLE then
    FvkDestroyDescriptorSetLayout(FDevice, ALayout, nil);
end;

procedure TVdxVulkanCompute.DestroyDescriptorPoolHandle(const APool: VkDescriptorPool);
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

// ============================================================================
//  Queries
// ============================================================================

function TVdxVulkanCompute.GetDeviceName(): string;
begin
  Result := string(AnsiString(PAnsiChar(@FDeviceProperties.deviceName[0])));
end;

function TVdxVulkanCompute.GetVRAMSizeMB(): UInt64;
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

function TVdxVulkanCompute.GetMaxComputeWorkGroupSize(): UInt32;
begin
  Result := FDeviceProperties.limits.maxComputeWorkGroupInvocations;
end;

end.
