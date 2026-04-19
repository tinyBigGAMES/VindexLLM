{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.TurboQuant;

interface

uses
  VindexLLM.TestCase,
  VindexLLM.Compute,
  VindexLLM.TurboQuant;

type

  { TTurboQuantTest }
  TTurboQuantTest = class(TVdxTestCase)
  private
    procedure SecCreateDestroy();
    procedure SecInitNilCompute();
    procedure SecInitSuccess();
    procedure SecCPURoundTrip();
    procedure SecGPURoundTrip();
    procedure SecGPUvsCPU();
  protected
    procedure Run(); override;
  public
    constructor Create(); override;
  end;

implementation

uses
  System.SysUtils,
  System.Math,
  VindexLLM.Utils,
  VindexLLM.Vulkan;

const
  // Number of TQ3 blocks the GPU sections exercise — enough to cross
  // a workgroup boundary on most implementations while keeping the
  // test fast.
  CTestNumBlocks = 64;
  CTestCount     = CTestNumBlocks * CTQ3BlockSize;  // 2048 F32 values

  // Byte size of the compressed TQ3 output for CTestNumBlocks blocks.
  CTestCompressedBytes = CTestNumBlocks * CTQ3PackedBytes;  // 1024 bytes

// ---------------------------------------------------------------------------
// NextLCGFloat — deterministic [0,1) pseudo-random stream driven by an
// LCG in Int64 space so {$Q+} debug builds don't trip on UInt32 overflow
// in the multiply. The seed is updated in place. Same constants as a
// standard Numerical Recipes LCG, but we do the wrap explicitly via
// mask instead of relying on UInt32 natural overflow — per the no-
// compiler-directive-suppression rule, the math has to be correct on
// the type, not patched by {$Q-}.
// ---------------------------------------------------------------------------
function NextLCGFloat(var ASeed: UInt32): Single;
var
  LScratch: Int64;
begin
  LScratch := (Int64(ASeed) * 1664525 + 1013904223) and $FFFFFFFF;
  ASeed    := UInt32(LScratch);
  Result   := Single(ASeed) / 4294967296.0;
end;

{ TTurboQuantTest }

constructor TTurboQuantTest.Create();
begin
  inherited;
  Title := 'Test_TurboQuant';
end;

procedure TTurboQuantTest.Run();
begin
  SecCreateDestroy();
  SecInitNilCompute();
  SecInitSuccess();
  SecCPURoundTrip();
  SecGPURoundTrip();
  SecGPUvsCPU();
end;

// ---------------------------------------------------------------------------
// SecCreateDestroy — instantiate TVdxTurboQuant, verify default
// uninitialized state, destroy cleanly. No TVdxCompute dependency.
// ---------------------------------------------------------------------------
procedure TTurboQuantTest.SecCreateDestroy();
var
  LTQ: TVdxTurboQuant;
begin
  Section('Create + Destroy (no compute)');

  LTQ := TVdxTurboQuant.Create();
  try
    Check(LTQ <> nil, 'Create returned a non-nil instance');
    Check(not LTQ.Initialized, 'Initialized is False before Init');
    FlushErrors(LTQ.GetErrors());
  finally
    LTQ.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecInitNilCompute — calling Init with ACompute=nil must fail fast,
// populate FErrors with the nil-compute code, and leave the instance
// unusable. No Vulkan work is attempted.
// ---------------------------------------------------------------------------
procedure TTurboQuantTest.SecInitNilCompute();
var
  LTQ: TVdxTurboQuant;
begin
  Section('Init(nil) -> error');

  LTQ := TVdxTurboQuant.Create();
  try
    Check(not LTQ.Init(nil), 'Init(nil) returns False');
    Check(LTQ.GetErrors().HasFatal(),
      'Init(nil) populated a fatal error');
    Check(not LTQ.Initialized, 'Initialized is False after Init(nil)');
    FlushErrors(LTQ.GetErrors());
  finally
    LTQ.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecInitSuccess — full boot on a real Vulkan GPU. Verifies every
// shader / pipeline / desc layout / pool / set inside TurboQuant was
// constructed without error, and Cleanup tears it down cleanly so a
// subsequent Init on a fresh instance would succeed too.
// ---------------------------------------------------------------------------
procedure TTurboQuantTest.SecInitSuccess();
var
  LCompute: TVdxCompute;
  LTQ:      TVdxTurboQuant;
begin
  Section('Init on real GPU');

  LCompute := TVdxCompute.Create();
  try
    Check(LCompute.Initialize(-1),
      'LCompute.Initialize(-1)');
    FlushErrors(LCompute.GetErrors());

    LTQ := TVdxTurboQuant.Create();
    try
      LTQ.SetErrors(LCompute.GetErrors());
      Check(LTQ.Init(LCompute), 'TurboQuant.Init(LCompute)');
      Check(LTQ.Initialized, 'Initialized is True after Init');
      FlushErrors(LTQ.GetErrors());
    finally
      LTQ.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecCPURoundTrip — Quantize a known F32 block on the CPU, dequantize
// it back, and check reconstruction error stays below the TQ3 tolerance
// budget. This is the reference path that SecGPUvsCPU compares against
// — if it's wrong, everything downstream is meaningless.
//
// Gaussian-ish input built from a cheap deterministic LCG; no RTL
// random dependency so the test is reproducible across runs and
// machines.
// ---------------------------------------------------------------------------
procedure TTurboQuantTest.SecCPURoundTrip();
var
  LInput:  array[0..CTQ3BlockSize - 1] of Single;
  LOutput: array[0..CTQ3BlockSize - 1] of Single;
  LBlock:  TVdxTQ3Block;
  LI:      Integer;
  LSeed:   UInt32;
  LU:      Single;
  LMSE:    Double;
begin
  Section('CPU Quantize -> Dequantize round trip');

  // Deterministic pseudo-random input in roughly [-2, +2] — matches
  // the range TQ3 is actually tuned for (post-WHT it maps to the
  // centroid dynamic range).
  LSeed := $C0FFEE42;
  for LI := 0 to CTQ3BlockSize - 1 do
  begin
    LU := NextLCGFloat(LSeed);          // [0, 1)
    LInput[LI] := (LU * 4.0) - 2.0;     // [-2, +2)
  end;

  TVdxTurboQuant.QuantizeBlockCPU(@LInput[0], LBlock);
  TVdxTurboQuant.DequantizeBlockCPU(LBlock, @LOutput[0]);

  LMSE := TVdxTurboQuant.ComputeMSE(@LInput[0], @LOutput[0],
    CTQ3BlockSize);

  Check(LMSE < 0.05,
    Format('CPU round-trip MSE %.6f < 0.05', [LMSE]));
end;

// ---------------------------------------------------------------------------
// SecGPURoundTrip — F32 input buffer -> GPU Quantize -> GPU Dequantize
// -> F32 output buffer. Compares reconstructed output to original
// input via MSE. Proves the GPU pipeline produces results within the
// same tolerance budget as the CPU reference.
// ---------------------------------------------------------------------------
procedure TTurboQuantTest.SecGPURoundTrip();
var
  LCompute:   TVdxCompute;
  LTQ:        TVdxTurboQuant;
  LInput:     array of Single;
  LOutput:    array of Single;
  LInputBuf:  TVdxGpuBuffer;
  LTQ3Buf:    TVdxGpuBuffer;
  LOutputBuf: TVdxGpuBuffer;
  LInputBytes:  UInt64;
  LTQ3Bytes:    UInt64;
  LI:         Integer;
  LSeed:      UInt32;
  LU:         Single;
  LMSE:       Double;
begin
  Section('GPU Quantize -> Dequantize round trip');

  SetLength(LInput, CTestCount);
  SetLength(LOutput, CTestCount);

  LSeed := $BADC0DE1;
  for LI := 0 to CTestCount - 1 do
  begin
    LU := NextLCGFloat(LSeed);
    LInput[LI] := (LU * 4.0) - 2.0;
  end;

  LInputBytes := UInt64(CTestCount) * SizeOf(Single);
  LTQ3Bytes   := UInt64(CTestCompressedBytes);

  LCompute := TVdxCompute.Create();
  try
    Check(LCompute.Initialize(-1), 'LCompute.Initialize(-1)');
    FlushErrors(LCompute.GetErrors());

    LTQ := TVdxTurboQuant.Create();
    try
      LTQ.SetErrors(LCompute.GetErrors());
      Check(LTQ.Init(LCompute), 'TurboQuant.Init(LCompute)');
      FlushErrors(LTQ.GetErrors());

      LInputBuf := LCompute.CreateGpuBuffer(LInputBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        vcBuffer);
      LTQ3Buf := LCompute.CreateGpuBuffer(LTQ3Bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vcBuffer);
      LOutputBuf := LCompute.CreateGpuBuffer(LInputBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        vcBuffer);
      try
        Check(LCompute.UploadToBuffer(LInputBuf, @LInput[0], LInputBytes),
          'Upload F32 input');
        FlushErrors(LCompute.GetErrors());

        LCompute.BeginBatch();
        try
          LTQ.Quantize(LInputBuf, LTQ3Buf, CTestNumBlocks);
          LCompute.BatchBarrier();
          LTQ.Dequantize(LTQ3Buf, LOutputBuf, CTestNumBlocks);
        finally
          LCompute.EndBatch();
        end;
        FlushErrors(LCompute.GetErrors());

        Check(LCompute.DownloadFromBuffer(
          LOutputBuf, @LOutput[0], LInputBytes),
          'Download F32 output');
        FlushErrors(LCompute.GetErrors());

        LMSE := TVdxTurboQuant.ComputeMSE(
          @LInput[0], @LOutput[0], CTestCount);
        Check(LMSE < 0.05,
          Format('GPU round-trip MSE %.6f < 0.05', [LMSE]));
      finally
        LCompute.DestroyGpuBuffer(LInputBuf);
        LCompute.DestroyGpuBuffer(LTQ3Buf);
        LCompute.DestroyGpuBuffer(LOutputBuf);
      end;
    finally
      LTQ.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecGPUvsCPU — run the same F32 input through GPU Quantize and the
// CPU QuantizeBlockCPU reference, then dequantize BOTH sets of
// compressed blocks via the CPU reference and compare reconstructions.
//
// Bit-identical on the compressed bytes is not the right bar — float
// precision drift in the WHT stage and boundary-case tie-breaking in
// the centroid lookup (elements landing exactly on a decision
// boundary go different ways between the shader and Pascal code)
// produce small index differences that don't affect reconstruction
// quality. The real contract is "GPU quantization is functionally
// equivalent to CPU quantization" — measured by reconstructed-value
// MSE, which must be tight (far tighter than the lossy-compression
// tolerance of Sections 4 and 5).
// ---------------------------------------------------------------------------
procedure TTurboQuantTest.SecGPUvsCPU();
var
  LCompute:    TVdxCompute;
  LTQ:         TVdxTurboQuant;
  LInput:      array of Single;
  LGpuBlocks:  array of TVdxTQ3Block;
  LCpuBlocks:  array of TVdxTQ3Block;
  LGpuRecon:   array of Single;
  LCpuRecon:   array of Single;
  LInputBuf:   TVdxGpuBuffer;
  LTQ3Buf:     TVdxGpuBuffer;
  LInputBytes: UInt64;
  LTQ3Bytes:   UInt64;
  LI:          Integer;
  LSeed:       UInt32;
  LU:          Single;
  LCrossMSE:   Double;
begin
  Section('GPU Quantize vs CPU QuantizeBlockCPU — reconstruction equivalence');

  SetLength(LInput, CTestCount);
  SetLength(LGpuBlocks, CTestNumBlocks);
  SetLength(LCpuBlocks, CTestNumBlocks);
  SetLength(LGpuRecon, CTestCount);
  SetLength(LCpuRecon, CTestCount);

  LSeed := $FEEDFACE;
  for LI := 0 to CTestCount - 1 do
  begin
    LU := NextLCGFloat(LSeed);
    LInput[LI] := (LU * 4.0) - 2.0;
  end;

  // CPU reference path — one block at a time.
  for LI := 0 to CTestNumBlocks - 1 do
    TVdxTurboQuant.QuantizeBlockCPU(
      @LInput[LI * CTQ3BlockSize], LCpuBlocks[LI]);

  LInputBytes := UInt64(CTestCount) * SizeOf(Single);
  LTQ3Bytes   := UInt64(CTestCompressedBytes);

  LCompute := TVdxCompute.Create();
  try
    Check(LCompute.Initialize(-1), 'LCompute.Initialize(-1)');
    FlushErrors(LCompute.GetErrors());

    LTQ := TVdxTurboQuant.Create();
    try
      LTQ.SetErrors(LCompute.GetErrors());
      Check(LTQ.Init(LCompute), 'TurboQuant.Init(LCompute)');
      FlushErrors(LTQ.GetErrors());

      LInputBuf := LCompute.CreateGpuBuffer(LInputBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        vcBuffer);
      LTQ3Buf := LCompute.CreateGpuBuffer(LTQ3Bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        vcBuffer);
      try
        Check(LCompute.UploadToBuffer(LInputBuf, @LInput[0], LInputBytes),
          'Upload F32 input');
        FlushErrors(LCompute.GetErrors());

        LCompute.BeginBatch();
        try
          LTQ.Quantize(LInputBuf, LTQ3Buf, CTestNumBlocks);
        finally
          LCompute.EndBatch();
        end;
        FlushErrors(LCompute.GetErrors());

        Check(LCompute.DownloadFromBuffer(
          LTQ3Buf, @LGpuBlocks[0], LTQ3Bytes),
          'Download compressed TQ3 output');
        FlushErrors(LCompute.GetErrors());

        // Reconstruct both sets via the CPU dequant — keeps the
        // comparison anchored to one well-understood arithmetic path.
        for LI := 0 to CTestNumBlocks - 1 do
        begin
          TVdxTurboQuant.DequantizeBlockCPU(
            LGpuBlocks[LI], @LGpuRecon[LI * CTQ3BlockSize]);
          TVdxTurboQuant.DequantizeBlockCPU(
            LCpuBlocks[LI], @LCpuRecon[LI * CTQ3BlockSize]);
        end;

        LCrossMSE := TVdxTurboQuant.ComputeMSE(
          @LGpuRecon[0], @LCpuRecon[0], CTestCount);

        // Tolerance: well below the lossy-round-trip budget. If the
        // paths are functionally equivalent (same centroid picks for
        // all non-boundary values, occasional ±1 centroid index on
        // exact boundaries), cross-MSE should be << 0.005.
        Check(LCrossMSE < 0.005,
          Format('GPU vs CPU reconstruction MSE %.6f < 0.005',
            [LCrossMSE]));
      finally
        LCompute.DestroyGpuBuffer(LInputBuf);
        LCompute.DestroyGpuBuffer(LTQ3Buf);
      end;
    finally
      LTQ.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

end.
