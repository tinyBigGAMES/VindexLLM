{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.TurboQuant;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  VindexLLM.Utils,
  VindexLLM.Resources,
  VindexLLM.Vulkan,
  VindexLLM.Compute,
  VindexLLM.Shaders;

const

  //--------------------------------------------------------------------------
  // Error Codes
  //--------------------------------------------------------------------------
  VDX_ERROR_TQ_COMPUTE_NIL   = 'TQ01';
  VDX_ERROR_TQ_ALREADY_INIT  = 'TQ02';
  VDX_ERROR_TQ_NOT_INIT      = 'TQ03';
  VDX_ERROR_TQ_INIT_EXCEPTION = 'TQ04';

  //--------------------------------------------------------------------------
  // TQ3 block layout — matches shader side byte-for-byte.
  // Block of 32 F32 values -> 16 bytes (4 x uint32, naturally aligned).
  //--------------------------------------------------------------------------
  CTQ3BlockSize   = 32;
  CTQ3PackedBytes = 16;
  CTQ3PackedWords = 4;

  // Lloyd–Max 3-bit centroids (8 levels, optimal for N(0,1)).
  CTQ3Centroids: array[0..7] of Single = (
    -2.1573, -1.3336, -0.7434, -0.2428,
    +0.2428, +0.7434, +1.3336, +2.1573
  );

  // Decision boundaries (midpoints between adjacent centroids).
  CTQ3Boundaries: array[0..6] of Single = (
    -1.7455, -1.0385, -0.4931, 0.0,
    +0.4931, +1.0385, +1.7455
  );

  // WHT normalization factor: 1/sqrt(32).
  CTQ3WHTNorm: Single = 0.17677669529663688;

  // Outermost centroid value — used as the scale divisor.
  CTQ3CentroidMax: Single = 2.1573;

  // Fixed sign-flip pattern (must match shaders exactly).
  CTQ3Signs: array[0..31] of Integer = (
    +1,-1,+1,+1,-1,+1,-1,-1, +1,+1,-1,+1,+1,-1,+1,-1,
    -1,+1,+1,-1,+1,-1,+1,+1, -1,-1,+1,-1,+1,+1,-1,+1
  );

type

  //==========================================================================
  // Push-constant records — match the SPIR-V shader layouts byte-for-byte.
  // Do not reorder fields or change widths; any divergence breaks binding.
  //==========================================================================

  { TVdxTQ3Push }
  // Generic TQ3 quantize/dequantize on arbitrary F32<->TQ3 buffers.
  TVdxTQ3Push = record
    NumBlocks: UInt32;
  end;

  { TVdxTQ3KVQuantPush }
  // Single-position KV cache quantize (decode path).
  TVdxTQ3KVQuantPush = record
    BlocksPerHead: UInt32;
    MaxSeq:        UInt32;
    Position:      UInt32;
    NumHeads:      UInt32;
  end;

  { TVdxTQ3KVDequantPush }
  // Range KV cache dequantize (V-cache read, continuation-prefill fill-in).
  TVdxTQ3KVDequantPush = record
    BlocksPerHead: UInt32;
    MaxSeq:        UInt32;
    SeqLen:        UInt32;
    NumHeads:      UInt32;
  end;

  { TVdxKVStoreBatchTQ3Push }
  // Fused batch KV store + TQ3 quantize (prefill path, writes both the
  // F32 decode buffer and the per-layer TQ3 cache in one dispatch).
  TVdxKVStoreBatchTQ3Push = record
    HeadDim:   UInt32;
    MaxSeq:    UInt32;
    NumHeads:  UInt32;
    NumTokens: UInt32;
    StartPos:  UInt32;
  end;

  { TVdxAttnScoresMHTQ3Push }
  // Fused multi-head attention scores over a TQ3-compressed K cache —
  // WHT on the fly, dots against packed centroids, no K dequant.
  TVdxAttnScoresMHTQ3Push = record
    HeadDim:       UInt32;
    SeqLen:        UInt32;
    MaxSeq:        UInt32;
    Scale:         Single;
    NumQHeads:     UInt32;
    GqaRatio:      UInt32;
    BlocksPerHead: UInt32;
  end;

  { TVdxTQ3Block }
  // In-memory layout of one TQ3 block (32 F32 values packed into 16 bytes).
  // Matches the shader-side block struct exactly.
  TVdxTQ3Block = record
    QS0:   UInt32;  // qs bytes [0..3]: elements  0..15, low 2 index bits
    QS1:   UInt32;  // qs bytes [4..7]: elements 16..31, low 2 index bits
    QR:    UInt32;  // qr bytes [0..3]: elements  0..31, high 1 index bit
    Gamma: UInt32;  // FP16 scale packed into low 16 bits
  end;
  PVdxTQ3Block = ^TVdxTQ3Block;

  { TVdxTurboQuant }
  // Owns every TQ3 shader, pipeline, descriptor layout/pool/set in the
  // project. Attention holds one of these as a field, shares its error
  // buffer via SetErrors at construction, and delegates all TQ3 dispatch
  // work here. Generic Quantize/Dequantize are for arbitrary F32<->TQ3
  // buffers (tests, future consumers). KVStoreBatchTQ3 / AttnScoresMHTQ3
  // / TQ3KVQuantAt / TQ3KVDequantRange are the KV-cache-aware fused
  // kernels used by Attention — they must be dispatched inside an
  // active batch opened by the caller (TVdxCompute.BeginBatch); this
  // unit issues no BeginBatch / EndBatch / BatchBarrier calls itself,
  // so callers keep full control of their submission ordering.
  TVdxTurboQuant = class(TVdxBaseObject)
  private
    FCompute:     TVdxCompute;
    FInitialized: Boolean;

    // Shader modules
    FQuantShader:           VkShaderModule;
    FDequantShader:         VkShaderModule;
    FTQ3KVQuantShader:      VkShaderModule;
    FTQ3KVDequantShader:    VkShaderModule;
    FKVStoreBatchTQ3Shader: VkShaderModule;
    FAttnScoresMHTQ3Shader: VkShaderModule;

    // Pipeline bundles
    FQuantBundle:           TVdxComputePipelineBundle;
    FDequantBundle:         TVdxComputePipelineBundle;
    FTQ3KVQuantBundle:      TVdxComputePipelineBundle;
    FTQ3KVDequantBundle:    TVdxComputePipelineBundle;
    FKVStoreBatchTQ3Bundle: TVdxComputePipelineBundle;
    FAttnScoresMHTQ3Bundle: TVdxComputePipelineBundle;

    // Descriptor set layouts (two shapes: 2-binding and 3-binding).
    FDescLayout2: VkDescriptorSetLayout;  // Quant, Dequant, TQ3KVQuant, TQ3KVDequant
    FDescLayout3: VkDescriptorSetLayout;  // KVStoreBatchTQ3, AttnScoresMHTQ3

    // Single consolidated descriptor pool + pre-allocated sets.
    FDescPool:                VkDescriptorPool;
    FQuantDescSet:            VkDescriptorSet;
    FDequantDescSet:          VkDescriptorSet;
    FTQ3KVQuantDescSet:       VkDescriptorSet;
    FTQ3KVDequantDescSet:     VkDescriptorSet;
    FKVStoreBatchTQ3DescSet:  VkDescriptorSet;
    FAttnScoresMHTQ3DescSet:  VkDescriptorSet;

    function LoadShader(const AName: string): VkShaderModule;
  public
    constructor Create(); override;
    destructor  Destroy(); override;

    // Build every shader / pipeline / desc layout / desc pool / desc set
    // this unit owns. Returns False with FErrors populated on any
    // failure; partial-init state is rolled back via Cleanup (safe to
    // call on an uninitialized or partially-initialized instance).
    function Init(const ACompute: TVdxCompute): Boolean;

    // Release every GPU resource. Safe on an uninitialized, partially-
    // initialized, or already-cleaned-up instance. Destroy calls this.
    procedure Cleanup();

    property Initialized: Boolean read FInitialized;

    //----------------------------------------------------------------------
    // Generic TQ3 primitive — arbitrary F32 <-> TQ3 buffers. Not tied to
    // any particular layout. Used by the Phase 9 test and any future
    // non-KV consumer (value cache, activations, weights). ANumBlocks
    // counts TQ3 blocks, not elements (1 block = 32 F32 input values =
    // 16 output bytes). Must run inside the caller's active batch.
    //----------------------------------------------------------------------
    procedure Quantize(const AInput: TVdxGpuBuffer;
      const AOutput: TVdxGpuBuffer;
      const ANumBlocks: Integer);
    procedure Dequantize(const AInput: TVdxGpuBuffer;
      const AOutput: TVdxGpuBuffer;
      const ANumBlocks: Integer);

    //----------------------------------------------------------------------
    // Fused batch KV store + TQ3 quantize. Reads ANumTokens projected
    // K (or V) vectors from AProj, writes both AWriteF32 (F32 decode
    // buffer, used by prefill attention reads in the same batch) and
    // ACacheTQ3 (compressed per-layer cache, used by decode-phase reads
    // in later batches) in a single dispatch.
    //
    // Caller wraps the attention step in its own BeginBatch / EndBatch
    // and places BatchBarriers appropriately.
    //----------------------------------------------------------------------
    procedure KVStoreBatchTQ3(const AProj: TVdxGpuBuffer;
      const AWriteF32: TVdxGpuBuffer;
      const ACacheTQ3: TVdxGpuBuffer;
      const AHeadDim: UInt32;
      const AMaxSeq: UInt32;
      const ANumHeads: UInt32;
      const ANumTokens: UInt32;
      const AStartPos: UInt32);

    //----------------------------------------------------------------------
    // Single-position decode-path quantize: read one position from the
    // F32 decode buffer, write it into the TQ3 cache at APosition.
    //----------------------------------------------------------------------
    procedure TQ3KVQuantAt(const ADecodeF32: TVdxGpuBuffer;
      const ACacheTQ3: TVdxGpuBuffer;
      const ABlocksPerHead: UInt32;
      const AMaxSeq: UInt32;
      const APosition: UInt32;
      const ANumHeads: UInt32);

    //----------------------------------------------------------------------
    // Range dequantize: read positions [0, ASeqLen) from the TQ3 cache
    // into the F32 decode buffer. Used for V-cache reads every decode
    // step and for K/V continuation-prefill fill-in when AStartPos > 0.
    //----------------------------------------------------------------------
    procedure TQ3KVDequantRange(const ACacheTQ3: TVdxGpuBuffer;
      const ADecodeF32: TVdxGpuBuffer;
      const ABlocksPerHead: UInt32;
      const AMaxSeq: UInt32;
      const ASeqLen: UInt32;
      const ANumHeads: UInt32);

    //----------------------------------------------------------------------
    // Fused multi-head attention scores over a TQ3-compressed K cache.
    // Reads Q + compressed K, applies WHT on the fly, dots against
    // packed centroids — the K dequant pass is eliminated.
    //----------------------------------------------------------------------
    procedure AttnScoresMHTQ3(const AQ: TVdxGpuBuffer;
      const AKCacheTQ3: TVdxGpuBuffer;
      const AScores: TVdxGpuBuffer;
      const AHeadDim: UInt32;
      const ASeqLen: UInt32;
      const AMaxSeq: UInt32;
      const AScale: Single;
      const ANumQHeads: UInt32;
      const AGqaRatio: UInt32;
      const ABlocksPerHead: UInt32);

    //----------------------------------------------------------------------
    // CPU reference implementations. Used by the test program to
    // validate GPU bit-equivalence and round-trip tolerance. Not part
    // of the inference fast path.
    //----------------------------------------------------------------------
    class procedure QuantizeBlockCPU(const AInput: PSingle;
      var AOutput: TVdxTQ3Block); static;
    class procedure DequantizeBlockCPU(const AInput: TVdxTQ3Block;
      const AOutput: PSingle); static;
    class function ComputeMSE(const AA: PSingle;
      const AB: PSingle;
      const ACount: Integer): Double; static;
  end;

implementation

//==============================================================================
//  FP16 <-> FP32 scalar conversion — unit-private helpers, used only by the
//  CPU reference Quantize/Dequantize class methods. The GPU path handles
//  FP16 natively inside the shader; these are for tests.
//==============================================================================

function VdxSingleToHalf(const AValue: Single): UInt32;
var
  LBits: UInt32;
  LSign: UInt32;
  LExp:  Integer;
  LMant: UInt32;
begin
  LBits := PUInt32(@AValue)^;
  LSign := (LBits shr 16) and $8000;
  LExp  := Integer((LBits shr 23) and $FF) - 127 + 15;
  LMant := (LBits and $7FFFFF) shr 13;

  if LExp <= 0 then
    Result := LSign
  else if LExp >= 31 then
    Result := LSign or $7C00
  else
    Result := LSign or (UInt32(LExp) shl 10) or LMant;
end;

function VdxHalfToSingle(const AHalf: UInt32): Single;
var
  LSign: UInt32;
  LExp:  UInt32;
  LMant: UInt32;
  LBits: UInt32;
begin
  LSign := (AHalf and $8000) shl 16;
  LExp  := (AHalf shr 10) and $1F;
  LMant := AHalf and $3FF;

  if LExp = 0 then
    LBits := LSign
  else if LExp = 31 then
    LBits := LSign or $7F800000 or (LMant shl 13)
  else
    LBits := LSign or ((LExp + 127 - 15) shl 23) or (LMant shl 13);

  Result := PSingle(@LBits)^;
end;


//==============================================================================
//  TVdxTurboQuant — Construction / Destruction
//==============================================================================

constructor TVdxTurboQuant.Create();
begin
  inherited Create();

  FCompute     := nil;
  FInitialized := False;

  FQuantShader           := VK_NULL_HANDLE;
  FDequantShader         := VK_NULL_HANDLE;
  FTQ3KVQuantShader      := VK_NULL_HANDLE;
  FTQ3KVDequantShader    := VK_NULL_HANDLE;
  FKVStoreBatchTQ3Shader := VK_NULL_HANDLE;
  FAttnScoresMHTQ3Shader := VK_NULL_HANDLE;

  FQuantBundle           := Default(TVdxComputePipelineBundle);
  FDequantBundle         := Default(TVdxComputePipelineBundle);
  FTQ3KVQuantBundle      := Default(TVdxComputePipelineBundle);
  FTQ3KVDequantBundle    := Default(TVdxComputePipelineBundle);
  FKVStoreBatchTQ3Bundle := Default(TVdxComputePipelineBundle);
  FAttnScoresMHTQ3Bundle := Default(TVdxComputePipelineBundle);

  FDescLayout2 := VK_NULL_HANDLE;
  FDescLayout3 := VK_NULL_HANDLE;

  FDescPool               := VK_NULL_HANDLE;
  FQuantDescSet           := VK_NULL_HANDLE;
  FDequantDescSet         := VK_NULL_HANDLE;
  FTQ3KVQuantDescSet      := VK_NULL_HANDLE;
  FTQ3KVDequantDescSet    := VK_NULL_HANDLE;
  FKVStoreBatchTQ3DescSet := VK_NULL_HANDLE;
  FAttnScoresMHTQ3DescSet := VK_NULL_HANDLE;
end;

destructor TVdxTurboQuant.Destroy();
begin
  Cleanup();
  inherited Destroy();
end;

function TVdxTurboQuant.LoadShader(const AName: string): VkShaderModule;
var
  LSpv: TBytes;
begin
  Result := VK_NULL_HANDLE;
  LSpv := VdxLoadShader(AName);
  if Length(LSpv) = 0 then Exit;
  Result := FCompute.CreateShaderModule(@LSpv[0], NativeUInt(Length(LSpv)));
end;

function TVdxTurboQuant.Init(const ACompute: TVdxCompute): Boolean;
var
  LDummyBuf: TVdxGpuBuffer;
begin
  Result := False;

  if FInitialized then
  begin
    FErrors.Add(esError, VDX_ERROR_TQ_ALREADY_INIT, RSTQAlreadyInit);
    Exit;
  end;

  if ACompute = nil then
  begin
    FErrors.Add(esFatal, VDX_ERROR_TQ_COMPUTE_NIL, RSTQComputeNil);
    Exit;
  end;

  FCompute := ACompute;

  try
    // --- Shader modules ---
    FQuantShader := LoadShader('TQ3_QUANTIZE');
    if FErrors.HasFatal() then Exit;

    FDequantShader := LoadShader('TQ3_DEQUANTIZE');
    if FErrors.HasFatal() then Exit;

    FTQ3KVQuantShader := LoadShader('TQ3_KV_QUANTIZE');
    if FErrors.HasFatal() then Exit;

    FTQ3KVDequantShader := LoadShader('TQ3_KV_DEQUANTIZE');
    if FErrors.HasFatal() then Exit;

    FKVStoreBatchTQ3Shader := LoadShader('KV_CACHE_STORE_BATCH_TQ3');
    if FErrors.HasFatal() then Exit;

    FAttnScoresMHTQ3Shader := LoadShader('ATTN_SCORES_MH_TQ3');
    if FErrors.HasFatal() then Exit;

    // --- Descriptor set layouts ---
    FDescLayout2 := FCompute.CreateStorageDescriptorSetLayout(2);
    if FErrors.HasFatal() then Exit;

    FDescLayout3 := FCompute.CreateStorageDescriptorSetLayout(3);
    if FErrors.HasFatal() then Exit;

    // --- Pipeline bundles ---
    FQuantBundle := FCompute.CreateComputePipelineWithPush(
      FQuantShader, 'main', FDescLayout2, SizeOf(TVdxTQ3Push));
    if FErrors.HasFatal() then Exit;

    FDequantBundle := FCompute.CreateComputePipelineWithPush(
      FDequantShader, 'main', FDescLayout2, SizeOf(TVdxTQ3Push));
    if FErrors.HasFatal() then Exit;

    FTQ3KVQuantBundle := FCompute.CreateComputePipelineWithPush(
      FTQ3KVQuantShader, 'main', FDescLayout2,
      SizeOf(TVdxTQ3KVQuantPush));
    if FErrors.HasFatal() then Exit;

    FTQ3KVDequantBundle := FCompute.CreateComputePipelineWithPush(
      FTQ3KVDequantShader, 'main', FDescLayout2,
      SizeOf(TVdxTQ3KVDequantPush));
    if FErrors.HasFatal() then Exit;

    FKVStoreBatchTQ3Bundle := FCompute.CreateComputePipelineWithPush(
      FKVStoreBatchTQ3Shader, 'main', FDescLayout3,
      SizeOf(TVdxKVStoreBatchTQ3Push));
    if FErrors.HasFatal() then Exit;

    FAttnScoresMHTQ3Bundle := FCompute.CreateComputePipelineWithPush(
      FAttnScoresMHTQ3Shader, 'main', FDescLayout3,
      SizeOf(TVdxAttnScoresMHTQ3Push));
    if FErrors.HasFatal() then Exit;

    // --- Single consolidated descriptor pool ---
    // 6 sets total: 4 on FDescLayout2 (2 bindings each = 8 descriptors)
    //             + 2 on FDescLayout3 (3 bindings each = 6 descriptors)
    // = 14 storage-buffer descriptors across 6 sets.
    FDescPool := FCompute.CreateDescriptorPoolForStorage(6, 14);
    if FErrors.HasFatal() then Exit;

    LDummyBuf := Default(TVdxGpuBuffer);

    FQuantDescSet := FCompute.AllocateDescriptorSetForBuffers(
      FDescPool, FDescLayout2, [LDummyBuf, LDummyBuf]);
    if FErrors.HasFatal() then Exit;

    FDequantDescSet := FCompute.AllocateDescriptorSetForBuffers(
      FDescPool, FDescLayout2, [LDummyBuf, LDummyBuf]);
    if FErrors.HasFatal() then Exit;

    FTQ3KVQuantDescSet := FCompute.AllocateDescriptorSetForBuffers(
      FDescPool, FDescLayout2, [LDummyBuf, LDummyBuf]);
    if FErrors.HasFatal() then Exit;

    FTQ3KVDequantDescSet := FCompute.AllocateDescriptorSetForBuffers(
      FDescPool, FDescLayout2, [LDummyBuf, LDummyBuf]);
    if FErrors.HasFatal() then Exit;

    FKVStoreBatchTQ3DescSet := FCompute.AllocateDescriptorSetForBuffers(
      FDescPool, FDescLayout3, [LDummyBuf, LDummyBuf, LDummyBuf]);
    if FErrors.HasFatal() then Exit;

    FAttnScoresMHTQ3DescSet := FCompute.AllocateDescriptorSetForBuffers(
      FDescPool, FDescLayout3, [LDummyBuf, LDummyBuf, LDummyBuf]);
    if FErrors.HasFatal() then Exit;

    FInitialized := True;
    Result       := True;
  except
    on E: Exception do
    begin
      FErrors.Add(esFatal, VDX_ERROR_TQ_INIT_EXCEPTION,
        Format(RSTQInitException, [E.Message]));
      Result := False;
    end;
  end;
end;

procedure TVdxTurboQuant.Cleanup();
begin
  if FCompute = nil then
  begin
    FInitialized := False;
    Exit;
  end;

  // Pool is freed first — it owns all desc sets allocated from it.
  if FDescPool <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FDescPool);
    FDescPool := VK_NULL_HANDLE;
  end;
  FQuantDescSet           := VK_NULL_HANDLE;
  FDequantDescSet         := VK_NULL_HANDLE;
  FTQ3KVQuantDescSet      := VK_NULL_HANDLE;
  FTQ3KVDequantDescSet    := VK_NULL_HANDLE;
  FKVStoreBatchTQ3DescSet := VK_NULL_HANDLE;
  FAttnScoresMHTQ3DescSet := VK_NULL_HANDLE;

  // Pipelines.
  FCompute.DestroyComputePipelineBundle(FQuantBundle);
  FCompute.DestroyComputePipelineBundle(FDequantBundle);
  FCompute.DestroyComputePipelineBundle(FTQ3KVQuantBundle);
  FCompute.DestroyComputePipelineBundle(FTQ3KVDequantBundle);
  FCompute.DestroyComputePipelineBundle(FKVStoreBatchTQ3Bundle);
  FCompute.DestroyComputePipelineBundle(FAttnScoresMHTQ3Bundle);

  // Layouts.
  if FDescLayout2 <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorSetLayoutHandle(FDescLayout2);
    FDescLayout2 := VK_NULL_HANDLE;
  end;
  if FDescLayout3 <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorSetLayoutHandle(FDescLayout3);
    FDescLayout3 := VK_NULL_HANDLE;
  end;

  // Shader modules.
  if FQuantShader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FQuantShader);
    FQuantShader := VK_NULL_HANDLE;
  end;
  if FDequantShader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FDequantShader);
    FDequantShader := VK_NULL_HANDLE;
  end;
  if FTQ3KVQuantShader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FTQ3KVQuantShader);
    FTQ3KVQuantShader := VK_NULL_HANDLE;
  end;
  if FTQ3KVDequantShader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FTQ3KVDequantShader);
    FTQ3KVDequantShader := VK_NULL_HANDLE;
  end;
  if FKVStoreBatchTQ3Shader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FKVStoreBatchTQ3Shader);
    FKVStoreBatchTQ3Shader := VK_NULL_HANDLE;
  end;
  if FAttnScoresMHTQ3Shader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FAttnScoresMHTQ3Shader);
    FAttnScoresMHTQ3Shader := VK_NULL_HANDLE;
  end;

  FInitialized := False;
  FCompute     := nil;
end;


//==============================================================================
//  GPU dispatches — each method records exactly one dispatch into the
//  caller's active batch. Callers own BeginBatch/EndBatch and place
//  BatchBarriers between steps.
//==============================================================================

procedure TVdxTurboQuant.Quantize(const AInput: TVdxGpuBuffer;
  const AOutput: TVdxGpuBuffer;
  const ANumBlocks: Integer);
var
  LPush: TVdxTQ3Push;
begin
  LPush.NumBlocks := UInt32(ANumBlocks);

  FCompute.UpdateDescriptorSetBuffers(FQuantDescSet, [AInput, AOutput]);
  FCompute.DispatchComputeWithPush(
    FQuantBundle.Pipeline, FQuantBundle.PipelineLayout,
    FQuantDescSet, @LPush, SizeOf(LPush), UInt32(ANumBlocks));
end;

procedure TVdxTurboQuant.Dequantize(const AInput: TVdxGpuBuffer;
  const AOutput: TVdxGpuBuffer;
  const ANumBlocks: Integer);
var
  LPush: TVdxTQ3Push;
begin
  LPush.NumBlocks := UInt32(ANumBlocks);

  FCompute.UpdateDescriptorSetBuffers(FDequantDescSet, [AInput, AOutput]);
  FCompute.DispatchComputeWithPush(
    FDequantBundle.Pipeline, FDequantBundle.PipelineLayout,
    FDequantDescSet, @LPush, SizeOf(LPush), UInt32(ANumBlocks));
end;

procedure TVdxTurboQuant.KVStoreBatchTQ3(const AProj: TVdxGpuBuffer;
  const AWriteF32: TVdxGpuBuffer;
  const ACacheTQ3: TVdxGpuBuffer;
  const AHeadDim: UInt32;
  const AMaxSeq: UInt32;
  const ANumHeads: UInt32;
  const ANumTokens: UInt32;
  const AStartPos: UInt32);
var
  LPush: TVdxKVStoreBatchTQ3Push;
begin
  LPush.HeadDim   := AHeadDim;
  LPush.MaxSeq    := AMaxSeq;
  LPush.NumHeads  := ANumHeads;
  LPush.NumTokens := ANumTokens;
  LPush.StartPos  := AStartPos;

  FCompute.UpdateDescriptorSetBuffers(FKVStoreBatchTQ3DescSet,
    [AProj, AWriteF32, ACacheTQ3]);
  FCompute.DispatchComputeWithPush(
    FKVStoreBatchTQ3Bundle.Pipeline,
    FKVStoreBatchTQ3Bundle.PipelineLayout,
    FKVStoreBatchTQ3DescSet, @LPush, SizeOf(LPush),
    (AHeadDim div CTQ3BlockSize) * ANumHeads, ANumTokens);
end;

procedure TVdxTurboQuant.TQ3KVQuantAt(const ADecodeF32: TVdxGpuBuffer;
  const ACacheTQ3: TVdxGpuBuffer;
  const ABlocksPerHead: UInt32;
  const AMaxSeq: UInt32;
  const APosition: UInt32;
  const ANumHeads: UInt32);
var
  LPush: TVdxTQ3KVQuantPush;
begin
  LPush.BlocksPerHead := ABlocksPerHead;
  LPush.MaxSeq        := AMaxSeq;
  LPush.Position      := APosition;
  LPush.NumHeads      := ANumHeads;

  FCompute.UpdateDescriptorSetBuffers(FTQ3KVQuantDescSet,
    [ADecodeF32, ACacheTQ3]);
  FCompute.DispatchComputeWithPush(
    FTQ3KVQuantBundle.Pipeline, FTQ3KVQuantBundle.PipelineLayout,
    FTQ3KVQuantDescSet, @LPush, SizeOf(LPush),
    ABlocksPerHead * ANumHeads);
end;

procedure TVdxTurboQuant.TQ3KVDequantRange(const ACacheTQ3: TVdxGpuBuffer;
  const ADecodeF32: TVdxGpuBuffer;
  const ABlocksPerHead: UInt32;
  const AMaxSeq: UInt32;
  const ASeqLen: UInt32;
  const ANumHeads: UInt32);
var
  LPush: TVdxTQ3KVDequantPush;
begin
  LPush.BlocksPerHead := ABlocksPerHead;
  LPush.MaxSeq        := AMaxSeq;
  LPush.SeqLen        := ASeqLen;
  LPush.NumHeads      := ANumHeads;

  FCompute.UpdateDescriptorSetBuffers(FTQ3KVDequantDescSet,
    [ACacheTQ3, ADecodeF32]);
  FCompute.DispatchComputeWithPush(
    FTQ3KVDequantBundle.Pipeline, FTQ3KVDequantBundle.PipelineLayout,
    FTQ3KVDequantDescSet, @LPush, SizeOf(LPush),
    ABlocksPerHead * ANumHeads * ASeqLen);
end;

procedure TVdxTurboQuant.AttnScoresMHTQ3(const AQ: TVdxGpuBuffer;
  const AKCacheTQ3: TVdxGpuBuffer;
  const AScores: TVdxGpuBuffer;
  const AHeadDim: UInt32;
  const ASeqLen: UInt32;
  const AMaxSeq: UInt32;
  const AScale: Single;
  const ANumQHeads: UInt32;
  const AGqaRatio: UInt32;
  const ABlocksPerHead: UInt32);
var
  LPush: TVdxAttnScoresMHTQ3Push;
begin
  LPush.HeadDim       := AHeadDim;
  LPush.SeqLen        := ASeqLen;
  LPush.MaxSeq        := AMaxSeq;
  LPush.Scale         := AScale;
  LPush.NumQHeads     := ANumQHeads;
  LPush.GqaRatio      := AGqaRatio;
  LPush.BlocksPerHead := ABlocksPerHead;

  FCompute.UpdateDescriptorSetBuffers(FAttnScoresMHTQ3DescSet,
    [AQ, AKCacheTQ3, AScores]);
  FCompute.DispatchComputeWithPush(
    FAttnScoresMHTQ3Bundle.Pipeline,
    FAttnScoresMHTQ3Bundle.PipelineLayout,
    FAttnScoresMHTQ3DescSet, @LPush, SizeOf(LPush),
    (ASeqLen + 255) div 256, ANumQHeads);
end;


//==============================================================================
//  CPU reference implementations — used by tests to validate GPU results.
//  Not on the inference fast path; accuracy takes priority over speed.
//==============================================================================

class procedure TVdxTurboQuant.QuantizeBlockCPU(const AInput: PSingle;
  var AOutput: TVdxTQ3Block);
var
  LTemp:     array[0..31] of Single;
  LI:        Integer;
  LStep:     Integer;
  LJ:        Integer;
  LA:        Single;
  LB:        Single;
  LAmax:     Single;
  LGamma:    Single;
  LInvGamma: Single;
  LScaled:   Single;
  LIdx:      Integer;
  LQSWord:   Integer;
  LShift:    Integer;
begin
  // 1. Copy input and apply sign flips.
  for LI := 0 to 31 do
    LTemp[LI] := PSingle(PByte(AInput) + LI * SizeOf(Single))^ * CTQ3Signs[LI];

  // 2. WHT butterfly (5 stages: step = 1, 2, 4, 8, 16).
  LStep := 1;
  while LStep <= 16 do
  begin
    LI := 0;
    while LI < 32 do
    begin
      for LJ := LI to LI + LStep - 1 do
      begin
        LA := LTemp[LJ];
        LB := LTemp[LJ + LStep];
        LTemp[LJ]         := LA + LB;
        LTemp[LJ + LStep] := LA - LB;
      end;
      Inc(LI, LStep * 2);
    end;
    LStep := LStep * 2;
  end;

  // 3. Normalize by 1/sqrt(32).
  for LI := 0 to 31 do
    LTemp[LI] := LTemp[LI] * CTQ3WHTNorm;

  // 4. Find amax and compute the scale.
  LAmax := 0.0;
  for LI := 0 to 31 do
    if Abs(LTemp[LI]) > LAmax then
      LAmax := Abs(LTemp[LI]);

  LGamma := LAmax / CTQ3CentroidMax;
  if LGamma > 0.0 then
    LInvGamma := 1.0 / LGamma
  else
    LInvGamma := 0.0;

  // 5. Quantize each value and pack into the output block.
  AOutput.QS0 := 0;
  AOutput.QS1 := 0;
  AOutput.QR  := 0;

  for LI := 0 to 31 do
  begin
    LScaled := LTemp[LI] * LInvGamma;

    // Nearest centroid index (0..7) via boundary lookup.
    if      LScaled < CTQ3Boundaries[0] then LIdx := 0
    else if LScaled < CTQ3Boundaries[1] then LIdx := 1
    else if LScaled < CTQ3Boundaries[2] then LIdx := 2
    else if LScaled < CTQ3Boundaries[3] then LIdx := 3
    else if LScaled < CTQ3Boundaries[4] then LIdx := 4
    else if LScaled < CTQ3Boundaries[5] then LIdx := 5
    else if LScaled < CTQ3Boundaries[6] then LIdx := 6
    else                                     LIdx := 7;

    // Pack low 2 bits into the qs word.
    LQSWord := LI div 16;
    LShift  := ((LI div 4) mod 4) * 8 + (LI mod 4) * 2;
    if LQSWord = 0 then
      AOutput.QS0 := AOutput.QS0 or (UInt32(LIdx and 3) shl LShift)
    else
      AOutput.QS1 := AOutput.QS1 or (UInt32(LIdx and 3) shl LShift);

    // Pack high 1 bit into the qr word.
    AOutput.QR := AOutput.QR or (UInt32((LIdx shr 2) and 1) shl LI);
  end;

  // 6. Gamma stored as FP16 in the low 16 bits.
  AOutput.Gamma := VdxSingleToHalf(LGamma);
end;

class procedure TVdxTurboQuant.DequantizeBlockCPU(const AInput: TVdxTQ3Block;
  const AOutput: PSingle);
var
  LTemp:   array[0..31] of Single;
  LI:      Integer;
  LStep:   Integer;
  LJ:      Integer;
  LA:      Single;
  LB:      Single;
  LLow2:   UInt32;
  LHigh1:  UInt32;
  LIdx:    UInt32;
  LGamma:  Single;
  LQSWord: UInt32;
  LShift:  Integer;
begin
  // 1. Unpack gamma from FP16.
  LGamma := VdxHalfToSingle(AInput.Gamma);

  // 2. Unpack indices and lookup centroids.
  for LI := 0 to 31 do
  begin
    if LI < 16 then
      LQSWord := AInput.QS0
    else
      LQSWord := AInput.QS1;

    LShift := ((LI div 4) mod 4) * 8 + (LI mod 4) * 2;
    LLow2  := (LQSWord shr LShift) and 3;
    LHigh1 := (AInput.QR shr LI) and 1;
    LIdx   := LLow2 or (LHigh1 shl 2);

    LTemp[LI] := CTQ3Centroids[LIdx] * LGamma;
  end;

  // 3. Inverse WHT butterfly (same code as forward — WHT is self-inverse).
  LStep := 1;
  while LStep <= 16 do
  begin
    LI := 0;
    while LI < 32 do
    begin
      for LJ := LI to LI + LStep - 1 do
      begin
        LA := LTemp[LJ];
        LB := LTemp[LJ + LStep];
        LTemp[LJ]         := LA + LB;
        LTemp[LJ + LStep] := LA - LB;
      end;
      Inc(LI, LStep * 2);
    end;
    LStep := LStep * 2;
  end;

  // 4. Normalize by 1/sqrt(32) and undo sign flips.
  for LI := 0 to 31 do
    PSingle(PByte(AOutput) + LI * SizeOf(Single))^ :=
      LTemp[LI] * CTQ3WHTNorm * CTQ3Signs[LI];
end;

class function TVdxTurboQuant.ComputeMSE(const AA: PSingle;
  const AB: PSingle;
  const ACount: Integer): Double;
var
  LI:   Integer;
  LDiff: Double;
  LSum:  Double;
begin
  LSum := 0.0;
  for LI := 0 to ACount - 1 do
  begin
    LDiff := Double(PSingle(PByte(AA) + LI * SizeOf(Single))^) -
             Double(PSingle(PByte(AB) + LI * SizeOf(Single))^);
    LSum  := LSum + LDiff * LDiff;
  end;
  if ACount > 0 then
    Result := LSum / ACount
  else
    Result := 0.0;
end;

end.
