{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.FFNWeights;

interface

uses
  VindexLLM.TestCase,
  VindexLLM.Compute,
  VindexLLM.FFNWeights;

type

  { TFFNWeightsTest }
  TFFNWeightsTest = class(TVdxTestCase)
  private
    procedure SecCreateDestroy();
    procedure SecInitNilCompute();
    procedure SecInitSuccess();
    procedure SecInitFull();
    procedure SecResolveWeights();
    procedure SecForwardSmoke();
    procedure SecForwardBatchSmoke();
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
  VindexLLM.Vulkan,
  VindexLLM.GGUFReader;

const
  // Real Gemma 3 4B GGUF — same model UTest.Attention uses.
  CModelPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';

  // Gemma 3 4B FFN-relevant dimensions. Attention-specific dims
  // (Q/KV heads, head_dim, layer count, max seq) are not needed by
  // TVdxFFN.Init.
  CHiddenDim: UInt32 = 2560;
  CFFNWidth:  UInt32 = 10240;

{ TFFNWeightsTest }

constructor TFFNWeightsTest.Create();
begin
  inherited;
  Title := 'Test_FFNWeights';
end;

procedure TFFNWeightsTest.Run();
begin
  SecCreateDestroy();
  SecInitNilCompute();
  SecInitSuccess();
  SecInitFull();
  SecResolveWeights();
  SecForwardSmoke();
  SecForwardBatchSmoke();
end;

// ---------------------------------------------------------------------------
// SecCreateDestroy — instantiate TVdxFFN, verify default uninitialized
// state, destroy cleanly. No TVdxCompute dependency.
// ---------------------------------------------------------------------------
procedure TFFNWeightsTest.SecCreateDestroy();
var
  LFFN: TVdxFFN;
begin
  Section('Create + Destroy (no compute)');

  LFFN := TVdxFFN.Create();
  try
    Check(LFFN <> nil, 'Create returned a non-nil instance');
    Check(not LFFN.Initialized, 'Initialized is False before Init');
    FlushErrors(LFFN.GetErrors());
  finally
    LFFN.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecInitNilCompute — Init(nil, ...) must fail cleanly with
// VDX_ERROR_FFN_COMPUTE_NIL, leave Initialized False, and leave FErrors
// in a fatal state.
// ---------------------------------------------------------------------------
procedure TFFNWeightsTest.SecInitNilCompute();
var
  LFFN: TVdxFFN;
begin
  Section('Init(nil) fails cleanly');

  LFFN := TVdxFFN.Create();
  try
    Check(not LFFN.Init(nil, CHiddenDim, CFFNWidth),
      'Init(nil) returns False');
    Check(LFFN.GetErrors().HasFatal(),
      'FErrors.HasFatal after Init(nil)');
    Check(not LFFN.Initialized,
      'Initialized still False after failed Init');
    FlushErrors(LFFN.GetErrors());
  finally
    LFFN.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecInitSuccess — real TVdxCompute + Gemma 3 4B FFN dims. Init returns
// True, Initialized flips to True, and a second Init call fails with
// VDX_ERROR_FFN_ALREADY_INIT without mutating state.
// ---------------------------------------------------------------------------
procedure TFFNWeightsTest.SecInitSuccess();
var
  LCompute: TVdxCompute;
  LFFN:     TVdxFFN;
begin
  Section('Init success + already-init guard');

  LCompute := TVdxCompute.Create();
  try
    Check(LCompute.Initialize(-1),
      'FCompute.Initialize(-1) returns True');
    FlushErrors(LCompute.GetErrors());

    LFFN := TVdxFFN.Create();
    try
      LFFN.SetErrors(LCompute.GetErrors());

      Check(LFFN.Init(LCompute, CHiddenDim, CFFNWidth),
        'Init with valid compute returns True');
      Check(LFFN.Initialized,
        'Initialized is True after successful Init');
      FlushErrors(LFFN.GetErrors());

      // Second Init on the same instance must fail.
      Check(not LFFN.Init(LCompute, CHiddenDim, CFFNWidth),
        'Second Init returns False');
      Check(LFFN.GetErrors().HasErrors(),
        'FErrors has entries after second Init');
      Check(LFFN.Initialized,
        'Initialized stays True after rejected second Init');
      FlushErrors(LFFN.GetErrors());
    finally
      LFFN.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecInitFull — real compute, full shader/pipeline/buffer construction.
// Verifies that the entire Init path completes without error and that the
// shared streaming-staging pool on TVdxCompute was grown to a non-zero
// capacity (sized to the F16 upper bound of the largest gate/up/down
// projection slice — all three share the same byte count in Gemma 3).
// ---------------------------------------------------------------------------
procedure TFFNWeightsTest.SecInitFull();
var
  LCompute: TVdxCompute;
  LFFN:     TVdxFFN;
begin
  Section('Init full (shaders + pipelines + buffers + staging grow)');

  LCompute := TVdxCompute.Create();
  try
    Check(LCompute.Initialize(-1),
      'FCompute.Initialize(-1) returns True');
    FlushErrors(LCompute.GetErrors());

    Check(LCompute.GetStagingCapacity() = 0,
      'Staging capacity is 0 before FFN.Init');
    Check(LCompute.GetStagingCount() = 0,
      'Staging count is 0 before FFN.Init');

    LFFN := TVdxFFN.Create();
    try
      LFFN.SetErrors(LCompute.GetErrors());

      Check(LFFN.Init(LCompute, CHiddenDim, CFFNWidth),
        'Full Init returns True');
      Check(LFFN.Initialized,
        'Initialized is True after full Init');
      Check(not LFFN.GetErrors().HasFatal(),
        'No fatal errors after full Init');
      Check(LCompute.GetStagingCount() = 3,
        'Staging pool grew to 3 pairs (gate/up/down concurrent in-flight)');
      Check(LCompute.GetStagingCapacity() > 0,
        'Staging capacity non-zero after FFN.Init');

      // Expected F16 upper bound: HiddenDim * FFNWidth * 2 bytes per pair.
      // Gate, up, and down all share this byte count (gate and up are
      // [HiddenDim x FFNWidth]; down is [FFNWidth x HiddenDim]).
      //   Gemma 3 4B: 2560 * 10240 * 2 = 52,428,800 bytes = 50 MiB
      Check(LCompute.GetStagingCapacity() =
        UInt64(CHiddenDim) * CFFNWidth * 2,
        'Staging capacity equals expected F16 max-slice size');
      FlushErrors(LFFN.GetErrors());
    finally
      LFFN.Free();
    end;

    Check(LCompute.GetStagingCount() = 3,
      'Staging pool persists after FFN.Free (owned by TVdxCompute)');
    FlushErrors(LCompute.GetErrors());
  finally
    LCompute.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecResolveWeights — open real Gemma 3 4B GGUF, full Init, resolve
// layer 0's gate / up / down weights. Verifies pointers are non-nil, byte
// sizes match the expected F16 product, and the reported weight type is
// gtF16.
// ---------------------------------------------------------------------------
procedure TFFNWeightsTest.SecResolveWeights();
var
  LCompute:    TVdxCompute;
  LReader:     TVdxGGUFReader;
  LFFN:        TVdxFFN;
  LWeights:    TVdxFFNLayerWeights;
  LExpectGate: UInt64;
  LExpectUp:   UInt64;
  LExpectDown: UInt64;
begin
  Section('ResolveFFNWeights (Gemma 3 4B layer 0)');

  LCompute := TVdxCompute.Create();
  try
    Check(LCompute.Initialize(-1),
      'FCompute.Initialize(-1)');
    FlushErrors(LCompute.GetErrors());

    LReader := TVdxGGUFReader.Create();
    try
      LReader.SetErrors(LCompute.GetErrors());
      Check(LReader.Open(CModelPath),
        Format('Open model file (%s)', [CModelPath]));
      FlushErrors(LReader.GetErrors());

      LFFN := TVdxFFN.Create();
      try
        LFFN.SetErrors(LCompute.GetErrors());

        Check(LFFN.Init(LCompute, CHiddenDim, CFFNWidth),
          'FFN.Init');
        FlushErrors(LFFN.GetErrors());

        Check(LFFN.ResolveFFNWeights(LReader, 0, LWeights),
          'ResolveFFNWeights(0) returns True');
        FlushErrors(LFFN.GetErrors());

        Check(LWeights.GateWeightPtr <> nil,
          'Gate weight pointer non-nil');
        Check(LWeights.UpWeightPtr <> nil,
          'Up weight pointer non-nil');
        Check(LWeights.DownWeightPtr <> nil,
          'Down weight pointer non-nil');

        // F16 byte sizes:
        //   Gate: [HiddenDim x FFNWidth]   = 2560 x 10240 x 2 bytes
        //   Up:   same shape as gate       = same bytes
        //   Down: [FFNWidth x HiddenDim]   = 10240 x 2560 x 2 (same total)
        LExpectGate := UInt64(CHiddenDim) * CFFNWidth * 2;
        LExpectUp   := LExpectGate;
        LExpectDown := UInt64(CFFNWidth) * CHiddenDim * 2;

        Check(LWeights.GateWeightBytes = LExpectGate,
          Format('Gate bytes == %d', [LExpectGate]));
        Check(LWeights.UpWeightBytes = LExpectUp,
          Format('Up bytes == %d', [LExpectUp]));
        Check(LWeights.DownWeightBytes = LExpectDown,
          Format('Down bytes == %d', [LExpectDown]));

        Check(LWeights.WeightType = gtF16,
          'WeightType is gtF16');
      finally
        LFFN.Free();
      end;
    finally
      LReader.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecForwardSmoke — real Gemma 3 4B, layer 0. Full Forward dispatch with
// gate/up/down streamed through staging pairs 0/1/2 inside a batch.
// Verifies no errors surface, output is not all-zero, and every element is
// finite (no NaN / Inf). Not a numerical-equivalence test — that lands in
// Phase 20 (Paris regression). This proves the streaming architecture runs
// end-to-end without corrupting data.
// ---------------------------------------------------------------------------
procedure TFFNWeightsTest.SecForwardSmoke();
var
  LCompute:     TVdxCompute;
  LReader:      TVdxGGUFReader;
  LFFN:         TVdxFFN;
  LWeights:     TVdxFFNLayerWeights;
  LInputBuf:    TVdxGpuBuffer;
  LOutBuf:      TVdxGpuBuffer;
  LHiddenBytes: UInt64;
  LInput:       array of Single;
  LOutput:      array of Single;
  LI:           Integer;
  LAnyNaN:      Boolean;
  LAnyInf:      Boolean;
  LAllZero:     Boolean;
  LNonZero:     Integer;
  LMaxAbs:      Single;
begin
  Section('Forward smoke (Gemma 3 4B, layer 0)');

  LHiddenBytes := UInt64(CHiddenDim) * SizeOf(Single);

  LCompute := TVdxCompute.Create();
  try
    Check(LCompute.Initialize(-1),
      'FCompute.Initialize(-1)');
    FlushErrors(LCompute.GetErrors());

    LReader := TVdxGGUFReader.Create();
    try
      LReader.SetErrors(LCompute.GetErrors());
      Check(LReader.Open(CModelPath),
        'Open model file');
      FlushErrors(LReader.GetErrors());

      LFFN := TVdxFFN.Create();
      try
        LFFN.SetErrors(LCompute.GetErrors());

        Check(LFFN.Init(LCompute, CHiddenDim, CFFNWidth),
          'FFN.Init');
        FlushErrors(LFFN.GetErrors());

        Check(LFFN.ResolveFFNWeights(LReader, 0, LWeights),
          'ResolveFFNWeights(0)');
        FlushErrors(LFFN.GetErrors());

        //------------------------------------------------------------
        // Input residual — small varied pattern so the signal doesn't
        // vanish or saturate through gate/up/down. Host-visible so we
        // can upload without an intermediate staging allocation.
        // Represents a pre-normed residual (no norm is applied in this
        // test — the FFN unit takes normed input from the caller).
        //------------------------------------------------------------
        SetLength(LInput, CHiddenDim);
        for LI := 0 to Integer(CHiddenDim) - 1 do
          LInput[LI] := 0.01 * Single((LI mod 7) - 3);

        LInputBuf := LCompute.CreateGpuBuffer(LHiddenBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          vcBuffer);
        LOutBuf := LCompute.CreateGpuBuffer(LHiddenBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          vcBuffer);
        try
          Check(LCompute.UploadToBuffer(LInputBuf, @LInput[0], LHiddenBytes),
            'Upload dummy input vector');
          FlushErrors(LCompute.GetErrors());

          //----------------------------------------------------------
          // Forward must run inside an active batch — streaming
          // depends on CopyBuffer commands being recorded, not
          // executed synchronously. This is the same contract the
          // Model class (Phase 13) will use per token.
          //----------------------------------------------------------
          LCompute.BeginBatch();
          try
            LFFN.Forward(LInputBuf, LWeights, LOutBuf);
          finally
            LCompute.EndBatch();
          end;
          Check(not LFFN.GetErrors().HasFatal(),
            'No fatal errors after Forward');
          FlushErrors(LFFN.GetErrors());

          //----------------------------------------------------------
          // Download + inspect output
          //----------------------------------------------------------
          SetLength(LOutput, CHiddenDim);
          Check(LCompute.DownloadFromBuffer(LOutBuf, @LOutput[0], LHiddenBytes),
            'Download output');
          FlushErrors(LCompute.GetErrors());

          LAnyNaN  := False;
          LAnyInf  := False;
          LAllZero := True;
          LNonZero := 0;
          LMaxAbs  := 0.0;
          for LI := 0 to Integer(CHiddenDim) - 1 do
          begin
            if IsNan(LOutput[LI]) then LAnyNaN := True;
            if IsInfinite(LOutput[LI]) then LAnyInf := True;
            if LOutput[LI] <> 0.0 then
            begin
              LAllZero := False;
              Inc(LNonZero);
            end;
            if Abs(LOutput[LI]) > LMaxAbs then
              LMaxAbs := Abs(LOutput[LI]);
          end;

          TVdxUtils.PrintLn('    non-zero=%d / %d   max|out|=%g',
            [LNonZero, CHiddenDim, Double(LMaxAbs)]);

          Check(not LAnyNaN,
            'Output contains no NaN');
          Check(not LAnyInf,
            'Output contains no Inf');
          Check(not LAllZero,
            'Output is not entirely zero (FFN produced signal)');
        finally
          LCompute.DestroyGpuBuffer(LInputBuf);
          LCompute.DestroyGpuBuffer(LOutBuf);
        end;
      finally
        LFFN.Free();
      end;
    finally
      LReader.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecForwardBatchSmoke — prefill path with 4 dummy tokens, Gemma 3 4B
// layer 0. Streams gate/up/down weight slices through the 3-pair staging
// pool. Proves the batched path runs end-to-end without corruption and
// every element of the [4 x HiddenDim] output is finite and at least
// partially non-zero.
// ---------------------------------------------------------------------------
procedure TFFNWeightsTest.SecForwardBatchSmoke();
const
  CNumTokens: UInt32 = 4;
var
  LCompute:    TVdxCompute;
  LReader:     TVdxGGUFReader;
  LFFN:        TVdxFFN;
  LWeights:    TVdxFFNLayerWeights;
  LInputMat:   TVdxGpuBuffer;
  LGateMat:    TVdxGpuBuffer;
  LUpMat:      TVdxGpuBuffer;
  LOutMat:     TVdxGpuBuffer;
  LInputBytes: UInt64;
  LFFNBytes:   UInt64;
  LInput:      array of Single;
  LOutput:     array of Single;
  LI:          Integer;
  LAnyNaN:     Boolean;
  LAnyInf:     Boolean;
  LNonZero:    Integer;
  LMaxAbs:     Single;
begin
  Section('ForwardBatch smoke (Gemma 3 4B, layer 0, 4 tokens)');

  LInputBytes := UInt64(CNumTokens) * CHiddenDim * SizeOf(Single);
  LFFNBytes   := UInt64(CNumTokens) * CFFNWidth  * SizeOf(Single);

  LCompute := TVdxCompute.Create();
  try
    Check(LCompute.Initialize(-1),
      'FCompute.Initialize(-1)');
    FlushErrors(LCompute.GetErrors());

    LReader := TVdxGGUFReader.Create();
    try
      LReader.SetErrors(LCompute.GetErrors());
      Check(LReader.Open(CModelPath),
        'Open model file');
      FlushErrors(LReader.GetErrors());

      LFFN := TVdxFFN.Create();
      try
        LFFN.SetErrors(LCompute.GetErrors());

        Check(LFFN.Init(LCompute, CHiddenDim, CFFNWidth),
          'FFN.Init');
        FlushErrors(LFFN.GetErrors());

        Check(LFFN.ResolveFFNWeights(LReader, 0, LWeights),
          'ResolveFFNWeights(0)');
        FlushErrors(LFFN.GetErrors());

        SetLength(LInput, CNumTokens * CHiddenDim);
        for LI := 0 to High(LInput) do
          LInput[LI] := 0.01 * Single((LI mod 11) - 5);

        LInputMat := LCompute.CreateGpuBuffer(LInputBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          vcBuffer);
        LGateMat := LCompute.CreateGpuBuffer(LFFNBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          vcBuffer);
        LUpMat := LCompute.CreateGpuBuffer(LFFNBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          vcBuffer);
        LOutMat := LCompute.CreateGpuBuffer(LInputBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          vcBuffer);
        try
          Check(LCompute.UploadToBuffer(LInputMat, @LInput[0], LInputBytes),
            'Upload input matrix (4 tokens x HiddenDim)');
          FlushErrors(LCompute.GetErrors());

          LCompute.BeginBatch();
          try
            LFFN.ForwardBatch(
              LInputMat,
              LWeights,
              CNumTokens,
              LGateMat, LUpMat,
              LOutMat);
          finally
            LCompute.EndBatch();
          end;
          Check(not LFFN.GetErrors().HasFatal(),
            'No fatal errors after ForwardBatch');
          FlushErrors(LFFN.GetErrors());

          SetLength(LOutput, CNumTokens * CHiddenDim);
          Check(LCompute.DownloadFromBuffer(LOutMat, @LOutput[0], LInputBytes),
            'Download output matrix');
          FlushErrors(LCompute.GetErrors());

          LAnyNaN  := False;
          LAnyInf  := False;
          LNonZero := 0;
          LMaxAbs  := 0.0;
          for LI := 0 to High(LOutput) do
          begin
            if IsNan(LOutput[LI]) then LAnyNaN := True;
            if IsInfinite(LOutput[LI]) then LAnyInf := True;
            if LOutput[LI] <> 0.0 then Inc(LNonZero);
            if Abs(LOutput[LI]) > LMaxAbs then
              LMaxAbs := Abs(LOutput[LI]);
          end;

          TVdxUtils.PrintLn('    non-zero=%d / %d   max|out|=%g',
            [LNonZero, Length(LOutput), Double(LMaxAbs)]);

          Check(not LAnyNaN,
            'Output contains no NaN');
          Check(not LAnyInf,
            'Output contains no Inf');
          Check(LNonZero > 0,
            'Output is not entirely zero across all 4 tokens');
        finally
          LCompute.DestroyGpuBuffer(LInputMat);
          LCompute.DestroyGpuBuffer(LGateMat);
          LCompute.DestroyGpuBuffer(LUpMat);
          LCompute.DestroyGpuBuffer(LOutMat);
        end;
      finally
        LFFN.Free();
      end;
    finally
      LReader.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

end.
