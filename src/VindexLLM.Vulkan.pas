{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Vulkan;

{$I VindexLLM.Defines.inc}

interface

// ============================================================================
//  Vulkan Constants
// ============================================================================

const
  VK_NULL_HANDLE                       = 0;
  VK_TRUE                              = 1;
  VK_FALSE                             = 0;
  VK_WHOLE_SIZE                        = UInt64($FFFFFFFFFFFFFFFF);
  VK_MAX_PHYSICAL_DEVICE_NAME_SIZE     = 256;
  VK_UUID_SIZE                         = 16;
  VK_MAX_MEMORY_TYPES                  = 32;
  VK_MAX_MEMORY_HEAPS                  = 16;

  // VkResult
  VK_SUCCESS                           = 0;
  VK_NOT_READY                         = 1;
  VK_TIMEOUT                           = 2;
  VK_ERROR_OUT_OF_HOST_MEMORY          = -1;
  VK_ERROR_OUT_OF_DEVICE_MEMORY        = -2;
  VK_ERROR_INITIALIZATION_FAILED       = -3;
  VK_ERROR_DEVICE_LOST                 = -4;
  VK_ERROR_MEMORY_MAP_FAILED           = -5;
  VK_ERROR_LAYER_NOT_PRESENT           = -6;
  VK_ERROR_EXTENSION_NOT_PRESENT       = -7;
  VK_ERROR_FEATURE_NOT_PRESENT         = -8;
  VK_ERROR_TOO_MANY_OBJECTS            = -10;

  // VkStructureType (only the values we use)
  VK_STRUCTURE_TYPE_APPLICATION_INFO              = 0;
  VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO          = 1;
  VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO      = 2;
  VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO            = 3;
  VK_STRUCTURE_TYPE_SUBMIT_INFO                   = 4;
  VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO          = 5;
  VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE           = 6;
  VK_STRUCTURE_TYPE_FENCE_CREATE_INFO             = 8;
  VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO            = 12;
  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO     = 15;
  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO  = 29;
  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO   = 30;
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
  VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO   = 33;
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO  = 34;
  VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET          = 35;
  VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO       = 39;
  VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO   = 40;
  VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO       = 42;
  VK_STRUCTURE_TYPE_MEMORY_BARRIER                 = 46;

  // VkBufferUsageFlagBits
  VK_BUFFER_USAGE_TRANSFER_SRC_BIT    = $00000001;
  VK_BUFFER_USAGE_TRANSFER_DST_BIT    = $00000002;
  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT  = $00000010;
  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT  = $00000020;

  // VkMemoryPropertyFlagBits
  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT  = $00000001;
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT  = $00000002;
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = $00000004;
  VK_MEMORY_PROPERTY_HOST_CACHED_BIT   = $00000008;

  // VkQueueFlagBits
  VK_QUEUE_GRAPHICS_BIT  = $00000001;
  VK_QUEUE_COMPUTE_BIT   = $00000002;
  VK_QUEUE_TRANSFER_BIT  = $00000004;

  // VkDescriptorType
  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;

  // VkShaderStageFlagBits
  VK_SHADER_STAGE_COMPUTE_BIT = $00000020;

  // VkPipelineBindPoint
  VK_PIPELINE_BIND_POINT_COMPUTE = 1;

  // VkCommandBufferLevel
  VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;

  // VkPipelineStageFlagBits
  VK_PIPELINE_STAGE_TRANSFER_BIT         = $00001000;
  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT   = $00000020;

  // VkAccessFlagBits
  VK_ACCESS_SHADER_READ_BIT              = $00000020;
  VK_ACCESS_SHADER_WRITE_BIT             = $00000040;
  VK_ACCESS_TRANSFER_READ_BIT            = $00000800;
  VK_ACCESS_TRANSFER_WRITE_BIT           = $00001000;

  // VkCommandBufferUsageFlagBits
  VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = $00000001;

  // VkFenceCreateFlagBits
  VK_FENCE_CREATE_SIGNALED_BIT = $00000001;

  // VkCommandPoolCreateFlagBits
  VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = $00000002;

  // VkPhysicalDeviceType
  VK_PHYSICAL_DEVICE_TYPE_OTHER          = 0;
  VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1;
  VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   = 2;
  VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    = 3;
  VK_PHYSICAL_DEVICE_TYPE_CPU            = 4;

  // VkSharingMode
  VK_SHARING_MODE_EXCLUSIVE  = 0;
  VK_SHARING_MODE_CONCURRENT = 1;

  // VkDescriptorPoolCreateFlagBits
  VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT = $00000001;

  // Vulkan API version: 1.0.0
  VK_API_VERSION_1_0 = (1 shl 22) or (0 shl 12) or 0;

// ============================================================================
//  Vulkan Base Types
// ============================================================================

type
  VkResult            = Int32;
  VkStructureType     = UInt32;
  VkFlags             = UInt32;
  VkBool32            = UInt32;
  VkDeviceSize        = UInt64;
  VkSampleCountFlags  = VkFlags;

  // Dispatchable handles (pointers)
  VkInstance          = Pointer;
  VkPhysicalDevice    = Pointer;
  VkDevice            = Pointer;
  VkQueue             = Pointer;
  VkCommandBuffer     = Pointer;

  // Non-dispatchable handles (UInt64)
  VkBuffer            = UInt64;
  VkDeviceMemory      = UInt64;
  VkShaderModule      = UInt64;
  VkPipelineLayout    = UInt64;
  VkPipeline          = UInt64;
  VkCommandPool       = UInt64;
  VkDescriptorSetLayout = UInt64;
  VkDescriptorPool    = UInt64;
  VkDescriptorSet     = UInt64;
  VkFence             = UInt64;
  VkSemaphore         = UInt64;
  VkSampler           = UInt64;
  VkBufferView        = UInt64;

  PVkBuffer           = ^VkBuffer;
  PVkCommandBuffer    = ^VkCommandBuffer;
  PVkDescriptorSet    = ^VkDescriptorSet;
  PVkPhysicalDevice   = ^VkPhysicalDevice;
  PVkPipeline         = ^VkPipeline;
  PVkFence            = ^VkFence;

// ============================================================================
//  Vulkan Records (matching C struct layouts exactly)
// ============================================================================

type
  VkExtent3D = record
    width:  UInt32;
    height: UInt32;
    depth:  UInt32;
  end;

  VkApplicationInfo = record
    sType:              VkStructureType;
    pNext:              Pointer;
    pApplicationName:   PAnsiChar;
    applicationVersion: UInt32;
    pEngineName:        PAnsiChar;
    engineVersion:      UInt32;
    apiVersion:         UInt32;
  end;

  VkInstanceCreateInfo = record
    sType:                   VkStructureType;
    pNext:                   Pointer;
    flags:                   VkFlags;
    pApplicationInfo:        ^VkApplicationInfo;
    enabledLayerCount:       UInt32;
    ppEnabledLayerNames:     PPAnsiChar;
    enabledExtensionCount:   UInt32;
    ppEnabledExtensionNames: PPAnsiChar;
  end;

  VkPhysicalDeviceLimits = record
    maxImageDimension1D:                             UInt32;
    maxImageDimension2D:                             UInt32;
    maxImageDimension3D:                             UInt32;
    maxImageDimensionCube:                           UInt32;
    maxImageArrayLayers:                             UInt32;
    maxTexelBufferElements:                          UInt32;
    maxUniformBufferRange:                           UInt32;
    maxStorageBufferRange:                           UInt32;
    maxPushConstantsSize:                            UInt32;
    maxMemoryAllocationCount:                        UInt32;
    maxSamplerAllocationCount:                       UInt32;
    bufferImageGranularity:                          VkDeviceSize;
    sparseAddressSpaceSize:                          VkDeviceSize;
    maxBoundDescriptorSets:                          UInt32;
    maxPerStageDescriptorSamplers:                   UInt32;
    maxPerStageDescriptorUniformBuffers:             UInt32;
    maxPerStageDescriptorStorageBuffers:             UInt32;
    maxPerStageDescriptorSampledImages:              UInt32;
    maxPerStageDescriptorStorageImages:              UInt32;
    maxPerStageDescriptorInputAttachments:           UInt32;
    maxPerStageResources:                            UInt32;
    maxDescriptorSetSamplers:                        UInt32;
    maxDescriptorSetUniformBuffers:                  UInt32;
    maxDescriptorSetUniformBuffersDynamic:           UInt32;
    maxDescriptorSetStorageBuffers:                  UInt32;
    maxDescriptorSetStorageBuffersDynamic:           UInt32;
    maxDescriptorSetSampledImages:                   UInt32;
    maxDescriptorSetStorageImages:                   UInt32;
    maxDescriptorSetInputAttachments:                UInt32;
    maxVertexInputAttributes:                        UInt32;
    maxVertexInputBindings:                          UInt32;
    maxVertexInputAttributeOffset:                   UInt32;
    maxVertexInputBindingStride:                     UInt32;
    maxVertexOutputComponents:                       UInt32;
    maxTessellationGenerationLevel:                  UInt32;
    maxTessellationPatchSize:                        UInt32;
    maxTessellationControlPerVertexInputComponents:  UInt32;
    maxTessellationControlPerVertexOutputComponents: UInt32;
    maxTessellationControlPerPatchOutputComponents:  UInt32;
    maxTessellationControlTotalOutputComponents:     UInt32;
    maxTessellationEvaluationInputComponents:        UInt32;
    maxTessellationEvaluationOutputComponents:       UInt32;
    maxGeometryShaderInvocations:                    UInt32;
    maxGeometryInputComponents:                      UInt32;
    maxGeometryOutputComponents:                     UInt32;
    maxGeometryOutputVertices:                       UInt32;
    maxGeometryTotalOutputComponents:                UInt32;
    maxFragmentInputComponents:                      UInt32;
    maxFragmentOutputAttachments:                    UInt32;
    maxFragmentDualSrcAttachments:                   UInt32;
    maxFragmentCombinedOutputResources:              UInt32;
    maxComputeSharedMemorySize:                      UInt32;
    maxComputeWorkGroupCount:                        array[0..2] of UInt32;
    maxComputeWorkGroupInvocations:                  UInt32;
    maxComputeWorkGroupSize:                         array[0..2] of UInt32;
    subPixelPrecisionBits:                           UInt32;
    subTexelPrecisionBits:                           UInt32;
    mipmapPrecisionBits:                             UInt32;
    maxDrawIndexedIndexValue:                        UInt32;
    maxDrawIndirectCount:                            UInt32;
    maxSamplerLodBias:                               Single;
    maxSamplerAnisotropy:                            Single;
    maxViewports:                                    UInt32;
    maxViewportDimensions:                           array[0..1] of UInt32;
    viewportBoundsRange:                             array[0..1] of Single;
    viewportSubPixelBits:                            UInt32;
    minMemoryMapAlignment:                           NativeUInt;
    minTexelBufferOffsetAlignment:                   VkDeviceSize;
    minUniformBufferOffsetAlignment:                 VkDeviceSize;
    minStorageBufferOffsetAlignment:                 VkDeviceSize;
    minTexelOffset:                                  Int32;
    maxTexelOffset:                                  UInt32;
    minTexelGatherOffset:                            Int32;
    maxTexelGatherOffset:                            UInt32;
    minInterpolationOffset:                          Single;
    maxInterpolationOffset:                          Single;
    subPixelInterpolationOffsetBits:                 UInt32;
    maxFramebufferWidth:                             UInt32;
    maxFramebufferHeight:                            UInt32;
    maxFramebufferLayers:                            UInt32;
    framebufferColorSampleCounts:                    VkSampleCountFlags;
    framebufferDepthSampleCounts:                    VkSampleCountFlags;
    framebufferStencilSampleCounts:                  VkSampleCountFlags;
    framebufferNoAttachmentsSampleCounts:            VkSampleCountFlags;
    maxColorAttachments:                             UInt32;
    sampledImageColorSampleCounts:                   VkSampleCountFlags;
    sampledImageIntegerSampleCounts:                 VkSampleCountFlags;
    sampledImageDepthSampleCounts:                   VkSampleCountFlags;
    sampledImageStencilSampleCounts:                 VkSampleCountFlags;
    storageImageSampleCounts:                        VkSampleCountFlags;
    maxSampleMaskWords:                              UInt32;
    timestampComputeAndGraphics:                     VkBool32;
    timestampPeriod:                                 Single;
    maxClipDistances:                                UInt32;
    maxCullDistances:                                UInt32;
    maxCombinedClipAndCullDistances:                 UInt32;
    discreteQueuePriorities:                         UInt32;
    pointSizeRange:                                  array[0..1] of Single;
    lineWidthRange:                                  array[0..1] of Single;
    pointSizeGranularity:                            Single;
    lineWidthGranularity:                            Single;
    strictLines:                                     VkBool32;
    standardSampleLocations:                         VkBool32;
    optimalBufferCopyOffsetAlignment:                VkDeviceSize;
    optimalBufferCopyRowPitchAlignment:              VkDeviceSize;
    nonCoherentAtomSize:                             VkDeviceSize;
  end;

  VkPhysicalDeviceSparseProperties = record
    residencyStandard2DBlockShape:            VkBool32;
    residencyStandard2DMultisampleBlockShape: VkBool32;
    residencyStandard3DBlockShape:            VkBool32;
    residencyAlignedMipSize:                  VkBool32;
    residencyNonResidentStrict:               VkBool32;
  end;

  VkPhysicalDeviceProperties = record
    apiVersion:       UInt32;
    driverVersion:    UInt32;
    vendorID:         UInt32;
    deviceID:         UInt32;
    deviceType:       UInt32;
    deviceName:       array[0..VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1] of AnsiChar;
    pipelineCacheUUID: array[0..VK_UUID_SIZE - 1] of UInt8;
    limits:           VkPhysicalDeviceLimits;
    sparseProperties: VkPhysicalDeviceSparseProperties;
  end;

  VkQueueFamilyProperties = record
    queueFlags:                  VkFlags;
    queueCount:                  UInt32;
    timestampValidBits:          UInt32;
    minImageTransferGranularity: VkExtent3D;
  end;
  PVkQueueFamilyProperties = ^VkQueueFamilyProperties;

  VkMemoryType = record
    propertyFlags: VkFlags;
    heapIndex:     UInt32;
  end;

  VkMemoryHeap = record
    size:  VkDeviceSize;
    flags: VkFlags;
  end;

  VkPhysicalDeviceMemoryProperties = record
    memoryTypeCount: UInt32;
    memoryTypes:     array[0..VK_MAX_MEMORY_TYPES - 1] of VkMemoryType;
    memoryHeapCount: UInt32;
    memoryHeaps:     array[0..VK_MAX_MEMORY_HEAPS - 1] of VkMemoryHeap;
  end;

  VkDeviceQueueCreateInfo = record
    sType:            VkStructureType;
    pNext:            Pointer;
    flags:            VkFlags;
    queueFamilyIndex: UInt32;
    queueCount:       UInt32;
    pQueuePriorities: PSingle;
  end;

  VkDeviceCreateInfo = record
    sType:                   VkStructureType;
    pNext:                   Pointer;
    flags:                   VkFlags;
    queueCreateInfoCount:    UInt32;
    pQueueCreateInfos:       ^VkDeviceQueueCreateInfo;
    enabledLayerCount:       UInt32;
    ppEnabledLayerNames:     PPAnsiChar;
    enabledExtensionCount:   UInt32;
    ppEnabledExtensionNames: PPAnsiChar;
    pEnabledFeatures:        Pointer;
  end;

  VkBufferCreateInfo = record
    sType:                 VkStructureType;
    pNext:                 Pointer;
    flags:                 VkFlags;
    size:                  VkDeviceSize;
    usage:                 VkFlags;
    sharingMode:           UInt32;
    queueFamilyIndexCount: UInt32;
    pQueueFamilyIndices:   PUInt32;
  end;

  VkMemoryRequirements = record
    size:           VkDeviceSize;
    alignment:      VkDeviceSize;
    memoryTypeBits: UInt32;
  end;

  VkMemoryAllocateInfo = record
    sType:           VkStructureType;
    pNext:           Pointer;
    allocationSize:  VkDeviceSize;
    memoryTypeIndex: UInt32;
  end;

  VkMappedMemoryRange = record
    sType:  VkStructureType;
    pNext:  Pointer;
    memory: VkDeviceMemory;
    offset: VkDeviceSize;
    size:   VkDeviceSize;
  end;

  VkShaderModuleCreateInfo = record
    sType:    VkStructureType;
    pNext:    Pointer;
    flags:    VkFlags;
    codeSize: NativeUInt;
    pCode:    PUInt32;
  end;

  VkSpecializationMapEntry = record
    constantID: UInt32;
    offset:     UInt32;
    size:       NativeUInt;
  end;

  VkSpecializationInfo = record
    mapEntryCount: UInt32;
    pMapEntries:   ^VkSpecializationMapEntry;
    dataSize:      NativeUInt;
    pData:         Pointer;
  end;

  VkPipelineShaderStageCreateInfo = record
    sType:               VkStructureType;
    pNext:               Pointer;
    flags:               VkFlags;
    stage:               VkFlags;
    module:              VkShaderModule;
    pName:               PAnsiChar;
    pSpecializationInfo: ^VkSpecializationInfo;
  end;

  VkPushConstantRange = record
    stageFlags: VkFlags;
    offset:     UInt32;
    size:       UInt32;
  end;

  VkPipelineLayoutCreateInfo = record
    sType:                  VkStructureType;
    pNext:                  Pointer;
    flags:                  VkFlags;
    setLayoutCount:         UInt32;
    pSetLayouts:            ^VkDescriptorSetLayout;
    pushConstantRangeCount: UInt32;
    pPushConstantRanges:    ^VkPushConstantRange;
  end;

  VkComputePipelineCreateInfo = record
    sType:              VkStructureType;
    pNext:              Pointer;
    flags:              VkFlags;
    stage:              VkPipelineShaderStageCreateInfo;
    layout:             VkPipelineLayout;
    basePipelineHandle: VkPipeline;
    basePipelineIndex:  Int32;
  end;

  VkDescriptorSetLayoutBinding = record
    binding:            UInt32;
    descriptorType:     UInt32;
    descriptorCount:    UInt32;
    stageFlags:         VkFlags;
    pImmutableSamplers: ^VkSampler;
  end;
  PVkDescriptorSetLayoutBinding = ^VkDescriptorSetLayoutBinding;

  VkDescriptorSetLayoutCreateInfo = record
    sType:        VkStructureType;
    pNext:        Pointer;
    flags:        VkFlags;
    bindingCount: UInt32;
    pBindings:    PVkDescriptorSetLayoutBinding;
  end;

  VkDescriptorPoolSize = record
    descriptorType:  UInt32;
    descriptorCount: UInt32;
  end;
  PVkDescriptorPoolSize = ^VkDescriptorPoolSize;

  VkDescriptorPoolCreateInfo = record
    sType:         VkStructureType;
    pNext:         Pointer;
    flags:         VkFlags;
    maxSets:       UInt32;
    poolSizeCount: UInt32;
    pPoolSizes:    PVkDescriptorPoolSize;
  end;

  VkDescriptorSetAllocateInfo = record
    sType:              VkStructureType;
    pNext:              Pointer;
    descriptorPool:     VkDescriptorPool;
    descriptorSetCount: UInt32;
    pSetLayouts:        ^VkDescriptorSetLayout;
  end;

  VkDescriptorBufferInfo = record
    buffer: VkBuffer;
    offset: VkDeviceSize;
    range:  VkDeviceSize;
  end;
  PVkDescriptorBufferInfo = ^VkDescriptorBufferInfo;

  VkDescriptorImageInfo = record
    sampler:     VkSampler;
    imageView:   UInt64;
    imageLayout: UInt32;
  end;
  PVkDescriptorImageInfo = ^VkDescriptorImageInfo;

  VkWriteDescriptorSet = record
    sType:            VkStructureType;
    pNext:            Pointer;
    dstSet:           VkDescriptorSet;
    dstBinding:       UInt32;
    dstArrayElement:  UInt32;
    descriptorCount:  UInt32;
    descriptorType:   UInt32;
    pImageInfo:       PVkDescriptorImageInfo;
    pBufferInfo:      PVkDescriptorBufferInfo;
    pTexelBufferView: ^VkBufferView;
  end;
  PVkWriteDescriptorSet = ^VkWriteDescriptorSet;

  VkCommandPoolCreateInfo = record
    sType:            VkStructureType;
    pNext:            Pointer;
    flags:            VkFlags;
    queueFamilyIndex: UInt32;
  end;

  VkCommandBufferAllocateInfo = record
    sType:              VkStructureType;
    pNext:              Pointer;
    commandPool:        VkCommandPool;
    level:              UInt32;
    commandBufferCount: UInt32;
  end;

  VkCommandBufferBeginInfo = record
    sType:            VkStructureType;
    pNext:            Pointer;
    flags:            VkFlags;
    pInheritanceInfo: Pointer;
  end;

  VkSubmitInfo = record
    sType:                VkStructureType;
    pNext:                Pointer;
    waitSemaphoreCount:   UInt32;
    pWaitSemaphores:      ^VkSemaphore;
    pWaitDstStageMask:    PUInt32;
    commandBufferCount:   UInt32;
    pCommandBuffers:      PVkCommandBuffer;
    signalSemaphoreCount: UInt32;
    pSignalSemaphores:    ^VkSemaphore;
  end;

  VkFenceCreateInfo = record
    sType: VkStructureType;
    pNext: Pointer;
    flags: VkFlags;
  end;

  VkBufferCopy = record
    srcOffset: VkDeviceSize;
    dstOffset: VkDeviceSize;
    size:      VkDeviceSize;
  end;

  VkMemoryBarrier = record
    sType:         VkStructureType;
    pNext:         Pointer;
    srcAccessMask: VkFlags;
    dstAccessMask: VkFlags;
  end;

// ============================================================================
//  Vulkan Function Pointer Types
// ============================================================================

type
  TvkGetInstanceProcAddr = function(AInstance: VkInstance; const AName: PAnsiChar): Pointer; stdcall;

  TvkCreateInstance = function(const ACreateInfo: VkInstanceCreateInfo; const AAllocator: Pointer; out AInstance: VkInstance): VkResult; stdcall;
  TvkDestroyInstance = procedure(AInstance: VkInstance; const AAllocator: Pointer); stdcall;

  TvkEnumeratePhysicalDevices = function(AInstance: VkInstance; var ACount: UInt32; ADevices: PVkPhysicalDevice): VkResult; stdcall;
  TvkGetPhysicalDeviceProperties = procedure(ADevice: VkPhysicalDevice; out AProperties: VkPhysicalDeviceProperties); stdcall;
  TvkGetPhysicalDeviceQueueFamilyProperties = procedure(ADevice: VkPhysicalDevice; var ACount: UInt32; AProperties: PVkQueueFamilyProperties); stdcall;
  TvkGetPhysicalDeviceMemoryProperties = procedure(ADevice: VkPhysicalDevice; out AProperties: VkPhysicalDeviceMemoryProperties); stdcall;

  TvkCreateDevice = function(APhysicalDevice: VkPhysicalDevice; const ACreateInfo: VkDeviceCreateInfo; const AAllocator: Pointer; out ADevice: VkDevice): VkResult; stdcall;
  TvkDestroyDevice = procedure(ADevice: VkDevice; const AAllocator: Pointer); stdcall;
  TvkGetDeviceQueue = procedure(ADevice: VkDevice; AQueueFamilyIndex: UInt32; AQueueIndex: UInt32; out AQueue: VkQueue); stdcall;

  TvkCreateBuffer = function(ADevice: VkDevice; const ACreateInfo: VkBufferCreateInfo; const AAllocator: Pointer; out ABuffer: VkBuffer): VkResult; stdcall;
  TvkDestroyBuffer = procedure(ADevice: VkDevice; ABuffer: VkBuffer; const AAllocator: Pointer); stdcall;
  TvkGetBufferMemoryRequirements = procedure(ADevice: VkDevice; ABuffer: VkBuffer; out ARequirements: VkMemoryRequirements); stdcall;

  TvkAllocateMemory = function(ADevice: VkDevice; const AAllocInfo: VkMemoryAllocateInfo; const AAllocator: Pointer; out AMemory: VkDeviceMemory): VkResult; stdcall;
  TvkFreeMemory = procedure(ADevice: VkDevice; AMemory: VkDeviceMemory; const AAllocator: Pointer); stdcall;
  TvkBindBufferMemory = function(ADevice: VkDevice; ABuffer: VkBuffer; AMemory: VkDeviceMemory; AOffset: VkDeviceSize): VkResult; stdcall;
  TvkMapMemory = function(ADevice: VkDevice; AMemory: VkDeviceMemory; AOffset: VkDeviceSize; ASize: VkDeviceSize; AFlags: VkFlags; out AData: Pointer): VkResult; stdcall;
  TvkUnmapMemory = procedure(ADevice: VkDevice; AMemory: VkDeviceMemory); stdcall;
  TvkFlushMappedMemoryRanges = function(ADevice: VkDevice; ARangeCount: UInt32; const ARanges: VkMappedMemoryRange): VkResult; stdcall;

  TvkCreateShaderModule = function(ADevice: VkDevice; const ACreateInfo: VkShaderModuleCreateInfo; const AAllocator: Pointer; out AModule: VkShaderModule): VkResult; stdcall;
  TvkDestroyShaderModule = procedure(ADevice: VkDevice; AModule: VkShaderModule; const AAllocator: Pointer); stdcall;

  TvkCreatePipelineLayout = function(ADevice: VkDevice; const ACreateInfo: VkPipelineLayoutCreateInfo; const AAllocator: Pointer; out ALayout: VkPipelineLayout): VkResult; stdcall;
  TvkDestroyPipelineLayout = procedure(ADevice: VkDevice; ALayout: VkPipelineLayout; const AAllocator: Pointer); stdcall;
  TvkCreateComputePipelines = function(ADevice: VkDevice; APipelineCache: UInt64; ACount: UInt32; const ACreateInfos: VkComputePipelineCreateInfo; const AAllocator: Pointer; APipelines: PVkPipeline): VkResult; stdcall;
  TvkDestroyPipeline = procedure(ADevice: VkDevice; APipeline: VkPipeline; const AAllocator: Pointer); stdcall;

  TvkCreateDescriptorSetLayout = function(ADevice: VkDevice; const ACreateInfo: VkDescriptorSetLayoutCreateInfo; const AAllocator: Pointer; out ALayout: VkDescriptorSetLayout): VkResult; stdcall;
  TvkDestroyDescriptorSetLayout = procedure(ADevice: VkDevice; ALayout: VkDescriptorSetLayout; const AAllocator: Pointer); stdcall;
  TvkCreateDescriptorPool = function(ADevice: VkDevice; const ACreateInfo: VkDescriptorPoolCreateInfo; const AAllocator: Pointer; out APool: VkDescriptorPool): VkResult; stdcall;
  TvkDestroyDescriptorPool = procedure(ADevice: VkDevice; APool: VkDescriptorPool; const AAllocator: Pointer); stdcall;
  TvkAllocateDescriptorSets = function(ADevice: VkDevice; const AAllocInfo: VkDescriptorSetAllocateInfo; ASets: PVkDescriptorSet): VkResult; stdcall;
  TvkUpdateDescriptorSets = procedure(ADevice: VkDevice; AWriteCount: UInt32; const AWrites: PVkWriteDescriptorSet; ACopyCount: UInt32; const ACopies: Pointer); stdcall;

  TvkCreateCommandPool = function(ADevice: VkDevice; const ACreateInfo: VkCommandPoolCreateInfo; const AAllocator: Pointer; out APool: VkCommandPool): VkResult; stdcall;
  TvkDestroyCommandPool = procedure(ADevice: VkDevice; APool: VkCommandPool; const AAllocator: Pointer); stdcall;
  TvkAllocateCommandBuffers = function(ADevice: VkDevice; const AAllocInfo: VkCommandBufferAllocateInfo; ABuffers: PVkCommandBuffer): VkResult; stdcall;

  TvkBeginCommandBuffer = function(ACmdBuf: VkCommandBuffer; const ABeginInfo: VkCommandBufferBeginInfo): VkResult; stdcall;
  TvkEndCommandBuffer = function(ACmdBuf: VkCommandBuffer): VkResult; stdcall;
  TvkCmdBindPipeline = procedure(ACmdBuf: VkCommandBuffer; ABindPoint: UInt32; APipeline: VkPipeline); stdcall;
  TvkCmdBindDescriptorSets = procedure(ACmdBuf: VkCommandBuffer; ABindPoint: UInt32; ALayout: VkPipelineLayout; AFirstSet: UInt32; ASetCount: UInt32; ASets: PVkDescriptorSet; ADynOffsetCount: UInt32; ADynOffsets: PUInt32); stdcall;
  TvkCmdDispatch = procedure(ACmdBuf: VkCommandBuffer; AGroupCountX: UInt32; AGroupCountY: UInt32; AGroupCountZ: UInt32); stdcall;

  TvkQueueSubmit = function(AQueue: VkQueue; ASubmitCount: UInt32; const ASubmits: VkSubmitInfo; AFence: VkFence): VkResult; stdcall;
  TvkQueueWaitIdle = function(AQueue: VkQueue): VkResult; stdcall;

  TvkCreateFence = function(ADevice: VkDevice; const ACreateInfo: VkFenceCreateInfo; const AAllocator: Pointer; out AFence: VkFence): VkResult; stdcall;
  TvkDestroyFence = procedure(ADevice: VkDevice; AFence: VkFence; const AAllocator: Pointer); stdcall;
  TvkWaitForFences = function(ADevice: VkDevice; ACount: UInt32; const AFences: PVkFence; AWaitAll: VkBool32; ATimeout: UInt64): VkResult; stdcall;
  TvkResetFences = function(ADevice: VkDevice; ACount: UInt32; const AFences: PVkFence): VkResult; stdcall;
  TvkCmdCopyBuffer = procedure(ACmdBuf: VkCommandBuffer; ASrcBuffer: VkBuffer; ADstBuffer: VkBuffer; ARegionCount: UInt32; const ARegions: VkBufferCopy); stdcall;
  TvkCmdPushConstants = procedure(ACmdBuf: VkCommandBuffer; ALayout: VkPipelineLayout; AStageFlags: VkFlags; AOffset: UInt32; ASize: UInt32; const AValues: Pointer); stdcall;
  TvkCmdPipelineBarrier = procedure(ACmdBuf: VkCommandBuffer; ASrcStageMask: VkFlags; ADstStageMask: VkFlags; ADependencyFlags: VkFlags; AMemoryBarrierCount: UInt32; const AMemoryBarriers: Pointer; ABufferMemoryBarrierCount: UInt32; const ABufferMemoryBarriers: Pointer; AImageMemoryBarrierCount: UInt32; const AImageMemoryBarriers: Pointer); stdcall;

implementation

end.
