{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.LayerNorm;

interface

uses
  VindexLLM.TestCase,
  VindexLLM.Compute,
  VindexLLM.GGUFReader,
  VindexLLM.LayerNorm;

type

  { TLayerNormTest }
  TLayerNormTest = class(TVdxTestCase)
  private
    FCompute: TVdxCompute;
    FLayerNorm: TVdxLayerNorm;

    procedure SecInit();
    procedure SecInitNilCompute();
    procedure SecApplyUninitialized();
    procedure SecApplyBasic();
    procedure SecApplyCopy();
    procedure SecApplyBatch();
    procedure SecApplyCopyBatch();
    procedure SecUploadNormWeights();
  protected
    procedure Run(); override;
  public
    constructor Create(); override;
    destructor  Destroy(); override;
  end;

implementation

uses
  System.SysUtils,
  System.Math,
  VindexLLM.Utils,
  VindexLLM.Vulkan;

const
  // Same model path used by UTest.GGUFReader and UTest.Tokenizer.
  CModelPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';

  CHiddenDim:   UInt32 = 256;
  CBatchTokens: UInt32 = 4;
  CEps:         Single = 1e-6;
  CTol:         Single = 1e-3;

// CPU reference for RMSNorm, matching rmsnorm.comp exactly:
//   rms    = sqrt(mean(x^2) + eps)
//   out[i] = (x[i] / rms) * w[i]
procedure RMSNormRef(const AIn: array of Single;
  const AWeight: array of Single;
  var AOut: array of Single;
  const AEps: Single);
var
  LI: Integer;
  LSumSq: Double;
  LInvRms: Single;
begin
  LSumSq := 0.0;
  for LI := 0 to High(AIn) do
    LSumSq := LSumSq + Double(AIn[LI]) * Double(AIn[LI]);
  LInvRms := 1.0 / Sqrt(LSumSq / Length(AIn) + AEps);
  for LI := 0 to High(AIn) do
    AOut[LI] := AIn[LI] * LInvRms * AWeight[LI];
end;

// Element-wise compare within tolerance. Sets AMaxErr to the largest
// abs diff seen and returns False on first mismatch above ATol.
function CompareArrays(const AActual, AExpected: array of Single;
  const ATol: Single; out AMaxErr: Single): Boolean;
var
  LI: Integer;
  LDiff: Single;
begin
  Result  := True;
  AMaxErr := 0;
  for LI := 0 to High(AActual) do
  begin
    LDiff := Abs(AActual[LI] - AExpected[LI]);
    if LDiff > AMaxErr then
      AMaxErr := LDiff;
    if LDiff > ATol then
    begin
      Result := False;
      Exit;
    end;
  end;
end;

{ TLayerNormTest }

constructor TLayerNormTest.Create();
begin
  inherited;
  Title := 'Test_LayerNorm';
  FCompute := nil;
  FLayerNorm := nil;
end;

destructor TLayerNormTest.Destroy();
begin
  FLayerNorm.Free();
  FCompute.Free();
  inherited;
end;

procedure TLayerNormTest.Run();
begin
  SecInit();
  SecInitNilCompute();
  SecApplyUninitialized();
  SecApplyBasic();
  SecApplyCopy();
  SecApplyBatch();
  SecApplyCopyBatch();
  SecUploadNormWeights();
end;

// ---------------------------------------------------------------------------
// SecInit — create shared FCompute and FLayerNorm. Wire error buffers
// via SetErrors so every Check reads FCompute.GetErrors(). This
// mirrors how TVdxModel (Phase 12) will own the wiring once it lands.
// ---------------------------------------------------------------------------
procedure TLayerNormTest.SecInit();
begin
  Section('Init (TVdxCompute auto + TVdxLayerNorm shared errors)');

  FCompute := TVdxCompute.Create();
  Check(FCompute.Initialize(-1),
    'FCompute.Initialize(-1) returns True');
  FlushErrors(FCompute.GetErrors());

  FLayerNorm := TVdxLayerNorm.Create();
  FLayerNorm.SetErrors(FCompute.GetErrors());

  Check(FLayerNorm.Init(FCompute, CEps),
    'FLayerNorm.Init returns True');
  Check(FLayerNorm.Initialized, 'LayerNorm.Initialized is True');
  Check(Abs(FLayerNorm.Epsilon - CEps) < 1e-9,
    'LayerNorm.Epsilon matches configured value');
  FlushErrors(FCompute.GetErrors());
end;

// ---------------------------------------------------------------------------
// SecInitNilCompute — fresh local instance, Init(nil) must fail with
// VDX_ERROR_LN_COMPUTE_NIL logged and Initialized still False.
// ---------------------------------------------------------------------------
procedure TLayerNormTest.SecInitNilCompute();
var
  LLocal: TVdxLayerNorm;
begin
  Section('Init(nil) fails cleanly');

  LLocal := TVdxLayerNorm.Create();
  try
    Check(not LLocal.Init(nil, CEps), 'Init(nil) returns False');
    Check(LLocal.GetErrors().HasFatal(),
      'FErrors.HasFatal after Init(nil)');
    Check(not LLocal.Initialized,
      'Initialized is False after failed Init');
    FlushErrors(LLocal.GetErrors());
  finally
    LLocal.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecApplyUninitialized — fresh local LayerNorm (never Init'd), Apply
// must return False and log VDX_ERROR_LN_NOT_INIT. Proves callers
// can't dispatch on a half-built pipeline.
// ---------------------------------------------------------------------------
procedure TLayerNormTest.SecApplyUninitialized();
var
  LLocal: TVdxLayerNorm;
  LDummy: TVdxGpuBuffer;
begin
  Section('Apply on uninitialized LayerNorm fails cleanly');

  LDummy := Default(TVdxGpuBuffer);
  LLocal := TVdxLayerNorm.Create();
  try
    Check(not LLocal.Apply(LDummy, LDummy, 256),
      'Apply before Init returns False');
    FlushErrors(LLocal.GetErrors());
  finally
    LLocal.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecApplyBasic — single-token, in-place RMSNorm.
//   input:  x[i] = (i+1)  (avoid zero)
//   weight: w[i] = 1.0    (isolates the norm step)
// Uploads input + weight, dispatches Apply, downloads mutated input,
// compares to RMSNormRef within tolerance.
// ---------------------------------------------------------------------------
procedure TLayerNormTest.SecApplyBasic();
var
  LBufIn, LBufW: TVdxGpuBuffer;
  LIn, LW, LOut, LRef: array of Single;
  LI: Integer;
  LByteCount: UInt64;
  LMaxErr: Single;
  LMatches: Boolean;
begin
  Section('Apply (single-token in-place)');

  SetLength(LIn, CHiddenDim);
  SetLength(LW, CHiddenDim);
  SetLength(LOut, CHiddenDim);
  SetLength(LRef, CHiddenDim);
  for LI := 0 to Integer(CHiddenDim) - 1 do
  begin
    LIn[LI] := Single(LI + 1);
    LW[LI]  := 1.0;
  end;

  LByteCount := UInt64(CHiddenDim) * SizeOf(Single);

  LBufIn := FCompute.CreateGpuBuffer(LByteCount,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  LBufW := FCompute.CreateGpuBuffer(LByteCount,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcWeight);
  FlushErrors(FCompute.GetErrors());

  try
    FCompute.UploadToBuffer(LBufIn, @LIn[0], LByteCount);
    FCompute.UploadToBuffer(LBufW,  @LW[0],  LByteCount);
    FlushErrors(FCompute.GetErrors());

    Check(FLayerNorm.Apply(LBufIn, LBufW, CHiddenDim),
      'Apply returns True');
    FlushErrors(FCompute.GetErrors());

    FCompute.DownloadFromBuffer(LBufIn, @LOut[0], LByteCount);
    FlushErrors(FCompute.GetErrors());

    RMSNormRef(LIn, LW, LRef, CEps);

    LMatches := CompareArrays(LOut, LRef, CTol, LMaxErr);
    TVdxUtils.PrintLn('    Max abs error: %.6g', [LMaxErr]);
    Check(LMatches, 'GPU output matches CPU reference within tol');
  finally
    FCompute.DestroyGpuBuffer(LBufIn);
    FCompute.DestroyGpuBuffer(LBufW);
  end;
end;

// ---------------------------------------------------------------------------
// SecApplyCopy — fused copy+norm. Source must be untouched; dest must
// match CPU reference of RMSNorm(source).
// ---------------------------------------------------------------------------
procedure TLayerNormTest.SecApplyCopy();
var
  LBufSrc, LBufW, LBufDst: TVdxGpuBuffer;
  LSrc, LW, LDst, LRef, LSrcAfter: array of Single;
  LI: Integer;
  LByteCount: UInt64;
  LMaxErr: Single;
  LMatches, LSourceUntouched: Boolean;
begin
  Section('ApplyCopy (single-token fused src->dst)');

  SetLength(LSrc, CHiddenDim);
  SetLength(LW, CHiddenDim);
  SetLength(LDst, CHiddenDim);
  SetLength(LRef, CHiddenDim);
  SetLength(LSrcAfter, CHiddenDim);

  for LI := 0 to Integer(CHiddenDim) - 1 do
  begin
    LSrc[LI] := Single((LI mod 17) + 1) * 0.5;
    LW[LI]   := Single(LI mod 3) * 0.25 + 1.0;
    LDst[LI] := -999.0;
  end;

  LByteCount := UInt64(CHiddenDim) * SizeOf(Single);

  LBufSrc := FCompute.CreateGpuBuffer(LByteCount,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  LBufW := FCompute.CreateGpuBuffer(LByteCount,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcWeight);
  LBufDst := FCompute.CreateGpuBuffer(LByteCount,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  FlushErrors(FCompute.GetErrors());

  try
    FCompute.UploadToBuffer(LBufSrc, @LSrc[0], LByteCount);
    FCompute.UploadToBuffer(LBufW,   @LW[0],   LByteCount);
    FCompute.UploadToBuffer(LBufDst, @LDst[0], LByteCount);
    FlushErrors(FCompute.GetErrors());

    Check(FLayerNorm.ApplyCopy(LBufSrc, LBufW, LBufDst, CHiddenDim),
      'ApplyCopy returns True');
    FlushErrors(FCompute.GetErrors());

    FCompute.DownloadFromBuffer(LBufSrc, @LSrcAfter[0], LByteCount);
    FCompute.DownloadFromBuffer(LBufDst, @LDst[0],      LByteCount);
    FlushErrors(FCompute.GetErrors());

    LSourceUntouched := True;
    for LI := 0 to Integer(CHiddenDim) - 1 do
      if LSrcAfter[LI] <> LSrc[LI] then
      begin
        LSourceUntouched := False;
        Break;
      end;
    Check(LSourceUntouched, 'Source buffer unchanged by ApplyCopy');

    RMSNormRef(LSrc, LW, LRef, CEps);
    LMatches := CompareArrays(LDst, LRef, CTol, LMaxErr);
    TVdxUtils.PrintLn('    Max abs error: %.6g', [LMaxErr]);
    Check(LMatches, 'Dest matches CPU reference within tol');
  finally
    FCompute.DestroyGpuBuffer(LBufSrc);
    FCompute.DestroyGpuBuffer(LBufW);
    FCompute.DestroyGpuBuffer(LBufDst);
  end;
end;

// ---------------------------------------------------------------------------
// SecApplyBatch — [NumTokens x HiddenDim] matrix, in-place. Each row
// must be normalized independently; cross-row contamination shows up
// as any row disagreeing with its own CPU reference.
// ---------------------------------------------------------------------------
procedure TLayerNormTest.SecApplyBatch();
var
  LBufM, LBufW: TVdxGpuBuffer;
  LM, LW, LOut, LRowRef, LRowIn, LRowOut: array of Single;
  LRow, LI: Integer;
  LElemCount: UInt32;
  LByteCount: UInt64;
  LMaxErr, LRowMaxErr: Single;
  LAllMatch: Boolean;
begin
  Section('ApplyBatch (multi-token in-place)');

  LElemCount := CBatchTokens * CHiddenDim;
  LByteCount := UInt64(LElemCount) * SizeOf(Single);

  SetLength(LM, LElemCount);
  SetLength(LW, CHiddenDim);
  SetLength(LOut, LElemCount);
  SetLength(LRowRef, CHiddenDim);
  SetLength(LRowIn, CHiddenDim);
  SetLength(LRowOut, CHiddenDim);

  for LRow := 0 to Integer(CBatchTokens) - 1 do
    for LI := 0 to Integer(CHiddenDim) - 1 do
      LM[LRow * Integer(CHiddenDim) + LI] :=
        Single(LI + 1) * (Single(LRow) + 1.0) * 0.1;

  for LI := 0 to Integer(CHiddenDim) - 1 do
    LW[LI] := 1.0 + Single(LI mod 5) * 0.1;

  LBufM := FCompute.CreateGpuBuffer(LByteCount,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  LBufW := FCompute.CreateGpuBuffer(
    UInt64(CHiddenDim) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcWeight);
  FlushErrors(FCompute.GetErrors());

  try
    FCompute.UploadToBuffer(LBufM, @LM[0], LByteCount);
    FCompute.UploadToBuffer(LBufW, @LW[0],
      UInt64(CHiddenDim) * SizeOf(Single));
    FlushErrors(FCompute.GetErrors());

    Check(FLayerNorm.ApplyBatch(LBufM, LBufW, CHiddenDim, CBatchTokens),
      'ApplyBatch returns True');
    FlushErrors(FCompute.GetErrors());

    FCompute.DownloadFromBuffer(LBufM, @LOut[0], LByteCount);
    FlushErrors(FCompute.GetErrors());

    LAllMatch := True;
    LMaxErr := 0;
    for LRow := 0 to Integer(CBatchTokens) - 1 do
    begin
      for LI := 0 to Integer(CHiddenDim) - 1 do
      begin
        LRowIn[LI]  := LM[LRow * Integer(CHiddenDim) + LI];
        LRowOut[LI] := LOut[LRow * Integer(CHiddenDim) + LI];
      end;
      RMSNormRef(LRowIn, LW, LRowRef, CEps);
      if not CompareArrays(LRowOut, LRowRef, CTol, LRowMaxErr) then
      begin
        LAllMatch := False;
        TVdxUtils.PrintLn('    Row %d mismatch (max err %.6g)',
          [LRow, LRowMaxErr]);
        Break;
      end;
      if LRowMaxErr > LMaxErr then
        LMaxErr := LRowMaxErr;
    end;
    TVdxUtils.PrintLn('    Max abs error across all rows: %.6g', [LMaxErr]);
    Check(LAllMatch, 'All rows match their CPU references within tol');
  finally
    FCompute.DestroyGpuBuffer(LBufM);
    FCompute.DestroyGpuBuffer(LBufW);
  end;
end;

// ---------------------------------------------------------------------------
// SecApplyCopyBatch — fused multi-token copy+norm. Source untouched;
// each dest row matches RMSNorm of its source row.
// ---------------------------------------------------------------------------
procedure TLayerNormTest.SecApplyCopyBatch();
var
  LBufSrc, LBufW, LBufDst: TVdxGpuBuffer;
  LSrc, LW, LDst, LSrcAfter, LRowRef, LRowIn, LRowOut: array of Single;
  LRow, LI: Integer;
  LElemCount: UInt32;
  LByteCount: UInt64;
  LMaxErr, LRowMaxErr: Single;
  LAllMatch, LSourceUntouched: Boolean;
begin
  Section('ApplyCopyBatch (multi-token fused src->dst)');

  LElemCount := CBatchTokens * CHiddenDim;
  LByteCount := UInt64(LElemCount) * SizeOf(Single);

  SetLength(LSrc, LElemCount);
  SetLength(LW, CHiddenDim);
  SetLength(LDst, LElemCount);
  SetLength(LSrcAfter, LElemCount);
  SetLength(LRowRef, CHiddenDim);
  SetLength(LRowIn, CHiddenDim);
  SetLength(LRowOut, CHiddenDim);

  for LRow := 0 to Integer(CBatchTokens) - 1 do
    for LI := 0 to Integer(CHiddenDim) - 1 do
      LSrc[LRow * Integer(CHiddenDim) + LI] :=
        Single(LI + 1) * (Single(LRow) + 2.0) * 0.05;

  for LI := 0 to Integer(CHiddenDim) - 1 do
    LW[LI] := 0.8 + Single(LI mod 7) * 0.05;

  for LI := 0 to Integer(LElemCount) - 1 do
    LDst[LI] := -999.0;

  LBufSrc := FCompute.CreateGpuBuffer(LByteCount,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  LBufW := FCompute.CreateGpuBuffer(
    UInt64(CHiddenDim) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcWeight);
  LBufDst := FCompute.CreateGpuBuffer(LByteCount,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    vcBuffer);
  FlushErrors(FCompute.GetErrors());

  try
    FCompute.UploadToBuffer(LBufSrc, @LSrc[0], LByteCount);
    FCompute.UploadToBuffer(LBufW,   @LW[0],
      UInt64(CHiddenDim) * SizeOf(Single));
    FCompute.UploadToBuffer(LBufDst, @LDst[0], LByteCount);
    FlushErrors(FCompute.GetErrors());

    Check(FLayerNorm.ApplyCopyBatch(LBufSrc, LBufW, LBufDst,
      CHiddenDim, CBatchTokens),
      'ApplyCopyBatch returns True');
    FlushErrors(FCompute.GetErrors());

    FCompute.DownloadFromBuffer(LBufSrc, @LSrcAfter[0], LByteCount);
    FCompute.DownloadFromBuffer(LBufDst, @LDst[0],      LByteCount);
    FlushErrors(FCompute.GetErrors());

    LSourceUntouched := True;
    for LI := 0 to Integer(LElemCount) - 1 do
      if LSrcAfter[LI] <> LSrc[LI] then
      begin
        LSourceUntouched := False;
        Break;
      end;
    Check(LSourceUntouched, 'Source matrix unchanged by ApplyCopyBatch');

    LAllMatch := True;
    LMaxErr := 0;
    for LRow := 0 to Integer(CBatchTokens) - 1 do
    begin
      for LI := 0 to Integer(CHiddenDim) - 1 do
      begin
        LRowIn[LI]  := LSrc[LRow * Integer(CHiddenDim) + LI];
        LRowOut[LI] := LDst[LRow * Integer(CHiddenDim) + LI];
      end;
      RMSNormRef(LRowIn, LW, LRowRef, CEps);
      if not CompareArrays(LRowOut, LRowRef, CTol, LRowMaxErr) then
      begin
        LAllMatch := False;
        TVdxUtils.PrintLn('    Row %d mismatch (max err %.6g)',
          [LRow, LRowMaxErr]);
        Break;
      end;
      if LRowMaxErr > LMaxErr then
        LMaxErr := LRowMaxErr;
    end;
    TVdxUtils.PrintLn('    Max abs error across all rows: %.6g', [LMaxErr]);
    Check(LAllMatch, 'All dest rows match CPU references within tol');
  finally
    FCompute.DestroyGpuBuffer(LBufSrc);
    FCompute.DestroyGpuBuffer(LBufW);
    FCompute.DestroyGpuBuffer(LBufDst);
  end;
end;

// ---------------------------------------------------------------------------
// SecUploadNormWeights — open Gemma 3 4B GGUF, upload layer 0's six
// norm tensors, verify every handle non-null, check VRAM grew, then
// free and verify VRAM returns to baseline (no leaked handles).
// ---------------------------------------------------------------------------
procedure TLayerNormTest.SecUploadNormWeights();
var
  LReader: TVdxGGUFReader;
  LWeights: TVdxNormLayerWeights;
  LBaseline, LAfterUpload, LAfterFree: TVdxVRAMUsage;
begin
  Section('UploadNormWeights (Gemma 3 4B layer 0)');

  LBaseline := FCompute.GetVRAMUsage();

  LReader := TVdxGGUFReader.Create();
  LReader.SetErrors(FCompute.GetErrors());
  try
    Check(LReader.Open(CModelPath),
      Format('Open model file (%s)', [CModelPath]));
    FlushErrors(FCompute.GetErrors());

    Check(FLayerNorm.UploadNormWeights(LReader, 0, LWeights),
      'UploadNormWeights(layer=0) returns True');
    FlushErrors(FCompute.GetErrors());

    Check(LWeights.AttnNormGpu.Buffer     <> VK_NULL_HANDLE,
      'AttnNormGpu handle non-null');
    Check(LWeights.PostAttnNormGpu.Buffer <> VK_NULL_HANDLE,
      'PostAttnNormGpu handle non-null');
    Check(LWeights.FFNNormGpu.Buffer      <> VK_NULL_HANDLE,
      'FFNNormGpu handle non-null');
    Check(LWeights.PostFFNNormGpu.Buffer  <> VK_NULL_HANDLE,
      'PostFFNNormGpu handle non-null');
    Check(LWeights.QNormGpu.Buffer        <> VK_NULL_HANDLE,
      'QNormGpu handle non-null');
    Check(LWeights.KNormGpu.Buffer        <> VK_NULL_HANDLE,
      'KNormGpu handle non-null');

    LAfterUpload := FCompute.GetVRAMUsage();
    Check(LAfterUpload.WeightsBytes > LBaseline.WeightsBytes,
      'Weights VRAM bucket grew after UploadNormWeights');
    TVdxUtils.PrintLn('    Weights VRAM delta: %d bytes',
      [LAfterUpload.WeightsBytes - LBaseline.WeightsBytes]);

    FLayerNorm.FreeNormWeights(LWeights);
    FlushErrors(FCompute.GetErrors());

    LAfterFree := FCompute.GetVRAMUsage();
    Check(LAfterFree.WeightsBytes = LBaseline.WeightsBytes,
      'Weights VRAM returns to baseline after FreeNormWeights');
  finally
    LReader.Free();
  end;
end;

end.
