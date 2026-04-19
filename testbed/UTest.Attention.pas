{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.Attention;

interface

uses
  VindexLLM.TestCase,
  VindexLLM.Compute,
  VindexLLM.Attention;

type

  { TAttentionTest }
  TAttentionTest = class(TVdxTestCase)
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
  VindexLLM.GGUFReader,
  VindexLLM.LayerNorm;

const
  // Real Gemma 3 4B GGUF — same model UTest.LayerNorm uses.
  CModelPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';

  // Gemma 3 4B dimensions — used by SecInitSuccess to verify the
  // signature accepts the real model shape even though Phase 7A
  // doesn't build any pipelines from it yet.
  CHiddenDim:  UInt32 = 2560;
  CNumQHeads:  UInt32 = 8;
  CNumKVHeads: UInt32 = 4;
  CHeadDim:    UInt32 = 256;
  CNumLayers:  UInt32 = 34;
  CMaxSeqLen:  UInt32 = 2048;
  CFFNWidth:   UInt32 = 10240;

{ TAttentionTest }

constructor TAttentionTest.Create();
begin
  inherited;
  Title := 'Test_Attention';
end;

procedure TAttentionTest.Run();
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
// SecCreateDestroy — instantiate TVdxAttention, verify default
// uninitialized state, destroy cleanly. No TVdxCompute dependency.
// ---------------------------------------------------------------------------
procedure TAttentionTest.SecCreateDestroy();
var
  LAttn: TVdxAttention;
begin
  Section('Create + Destroy (no compute)');

  LAttn := TVdxAttention.Create();
  try
    Check(LAttn <> nil, 'Create returned a non-nil instance');
    Check(not LAttn.Initialized, 'Initialized is False before Init');
    FlushErrors(LAttn.GetErrors());
  finally
    LAttn.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecInitNilCompute — Init(nil, ...) must fail cleanly with
// VDX_ERROR_ATTN_COMPUTE_NIL, leave Initialized False, and leave
// FErrors in a fatal state.
// ---------------------------------------------------------------------------
procedure TAttentionTest.SecInitNilCompute();
var
  LAttn: TVdxAttention;
begin
  Section('Init(nil) fails cleanly');

  LAttn := TVdxAttention.Create();
  try
    Check(not LAttn.Init(nil,
      CHiddenDim, CNumQHeads, CNumKVHeads, CHeadDim,
      CNumLayers, CMaxSeqLen, CFFNWidth),
      'Init(nil) returns False');
    Check(LAttn.GetErrors().HasFatal(),
      'FErrors.HasFatal after Init(nil)');
    Check(not LAttn.Initialized,
      'Initialized still False after failed Init');
    FlushErrors(LAttn.GetErrors());
  finally
    LAttn.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecInitSuccess — real TVdxCompute + Gemma 3 4B dims. Init returns
// True, Initialized flips to True, and a second Init call fails with
// VDX_ERROR_ATTN_ALREADY_INIT without mutating state.
// ---------------------------------------------------------------------------
procedure TAttentionTest.SecInitSuccess();
var
  LCompute: TVdxCompute;
  LAttn:    TVdxAttention;
begin
  Section('Init success + already-init guard');

  LCompute := TVdxCompute.Create();
  try
    Check(LCompute.Initialize(-1),
      'FCompute.Initialize(-1) returns True');
    FlushErrors(LCompute.GetErrors());

    LAttn := TVdxAttention.Create();
    try
      LAttn.SetErrors(LCompute.GetErrors());

      Check(LAttn.Init(LCompute,
        CHiddenDim, CNumQHeads, CNumKVHeads, CHeadDim,
        CNumLayers, CMaxSeqLen, CFFNWidth),
        'Init with valid compute returns True');
      Check(LAttn.Initialized,
        'Initialized is True after successful Init');
      FlushErrors(LAttn.GetErrors());

      // Second Init on the same instance must fail.
      Check(not LAttn.Init(LCompute,
        CHiddenDim, CNumQHeads, CNumKVHeads, CHeadDim,
        CNumLayers, CMaxSeqLen, CFFNWidth),
        'Second Init returns False');
      Check(LAttn.GetErrors().HasErrors(),
        'FErrors has entries after second Init');
      Check(LAttn.Initialized,
        'Initialized stays True after rejected second Init');
      FlushErrors(LAttn.GetErrors());
    finally
      LAttn.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecInitFull — real compute, full shader/pipeline/buffer construction.
// Verifies that the entire Init path completes without error and that
// the shared streaming-staging pool on TVdxCompute was grown to a
// non-zero capacity (sized to the F16 upper bound of the largest
// Q/K/V/O projection slice).
// ---------------------------------------------------------------------------
procedure TAttentionTest.SecInitFull();
var
  LCompute: TVdxCompute;
  LAttn:    TVdxAttention;
begin
  Section('Init full (shaders + pipelines + buffers + staging grow)');

  LCompute := TVdxCompute.Create();
  try
    Check(LCompute.Initialize(-1),
      'FCompute.Initialize(-1) returns True');
    FlushErrors(LCompute.GetErrors());

    Check(LCompute.GetStagingCapacity() = 0,
      'Staging capacity is 0 before Attention.Init');
    Check(LCompute.GetStagingCount() = 0,
      'Staging count is 0 before Attention.Init');

    LAttn := TVdxAttention.Create();
    try
      LAttn.SetErrors(LCompute.GetErrors());

      Check(LAttn.Init(LCompute,
        CHiddenDim, CNumQHeads, CNumKVHeads, CHeadDim,
        CNumLayers, CMaxSeqLen, CFFNWidth),
        'Full Init returns True');
      Check(LAttn.Initialized,
        'Initialized is True after full Init');
      Check(not LAttn.GetErrors().HasFatal(),
        'No fatal errors after full Init');
      Check(LCompute.GetStagingCount() = 4,
        'Staging pool grew to 4 pairs (Q/K/V/O concurrent in-flight)');
      Check(LCompute.GetStagingCapacity() > 0,
        'Staging capacity non-zero after Attention.Init');

      // Expected F16 upper bound: max(Q=HiddenDim*NumQHeads*HeadDim,
      // KV=HiddenDim*NumKVHeads*HeadDim) * 2 bytes per pair
      //   = max(2560*2048, 2560*1024) * 2 = 10,485,760
      Check(LCompute.GetStagingCapacity() =
        UInt64(CHiddenDim) * (CNumQHeads * CHeadDim) * 2,
        'Staging capacity equals expected F16 max-slice size');
      FlushErrors(LAttn.GetErrors());
    finally
      LAttn.Free();
    end;

    Check(LCompute.GetStagingCount() = 4,
      'Staging pool persists after Attention.Free (owned by TVdxCompute)');
    FlushErrors(LCompute.GetErrors());
  finally
    LCompute.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecResolveWeights — open real Gemma 3 4B GGUF, full Init, resolve
// layer 0's Q/K/V/O weights. Verifies pointers are non-nil, byte
// sizes match the expected F16 product, and the reported weight
// type is gtF16.
// ---------------------------------------------------------------------------
procedure TAttentionTest.SecResolveWeights();
var
  LCompute: TVdxCompute;
  LReader:  TVdxGGUFReader;
  LAttn:    TVdxAttention;
  LWeights: TVdxAttnLayerWeights;
  LExpectQ: UInt64;
  LExpectK: UInt64;
  LExpectV: UInt64;
  LExpectO: UInt64;
begin
  Section('ResolveAttnWeights (Gemma 3 4B layer 0)');

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

      LAttn := TVdxAttention.Create();
      try
        LAttn.SetErrors(LCompute.GetErrors());

        Check(LAttn.Init(LCompute,
          CHiddenDim, CNumQHeads, CNumKVHeads, CHeadDim,
          CNumLayers, CMaxSeqLen, CFFNWidth),
          'Attention.Init');
        FlushErrors(LAttn.GetErrors());

        Check(LAttn.ResolveAttnWeights(LReader, 0, LWeights),
          'ResolveAttnWeights(0) returns True');
        FlushErrors(LAttn.GetErrors());

        Check(LWeights.QWeightPtr <> nil,
          'Q weight pointer non-nil');
        Check(LWeights.KWeightPtr <> nil,
          'K weight pointer non-nil');
        Check(LWeights.VWeightPtr <> nil,
          'V weight pointer non-nil');
        Check(LWeights.OWeightPtr <> nil,
          'O weight pointer non-nil');

        // F16 byte sizes — each tensor is [HiddenDim x OutDim] *
        // SizeOf(Half) or [OutDim x HiddenDim] * SizeOf(Half).
        //   Q: HiddenDim x (NumQHeads * HeadDim)   = 2560 x 2048
        //   K: HiddenDim x (NumKVHeads * HeadDim)  = 2560 x 1024
        //   V: HiddenDim x (NumKVHeads * HeadDim)  = 2560 x 1024
        //   O: (NumQHeads * HeadDim) x HiddenDim   = 2048 x 2560
        LExpectQ := UInt64(CHiddenDim) * (CNumQHeads * CHeadDim) * 2;
        LExpectK := UInt64(CHiddenDim) * (CNumKVHeads * CHeadDim) * 2;
        LExpectV := LExpectK;
        LExpectO := UInt64(CNumQHeads * CHeadDim) * CHiddenDim * 2;

        Check(LWeights.QWeightBytes = LExpectQ,
          Format('Q bytes == %d', [LExpectQ]));
        Check(LWeights.KWeightBytes = LExpectK,
          Format('K bytes == %d', [LExpectK]));
        Check(LWeights.VWeightBytes = LExpectV,
          Format('V bytes == %d', [LExpectV]));
        Check(LWeights.OWeightBytes = LExpectO,
          Format('O bytes == %d', [LExpectO]));

        Check(LWeights.WeightType = gtF16,
          'WeightType is gtF16');
      finally
        LAttn.Free();
      end;
    finally
      LReader.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecForwardSmoke — real Gemma 3 4B, layer 0, position 0. Full Forward
// dispatch with Q/K/V/O streamed through staging pairs 0/1/2/3 inside a
// batch. Verifies no errors surface, output is not all-zero, and every
// element is finite (no NaN / Inf). Not a numerical-equivalence test —
// that lands in Phase 20 (Paris regression). This proves the streaming
// architecture runs end-to-end without corrupting data.
// ---------------------------------------------------------------------------
procedure TAttentionTest.SecForwardSmoke();
var
  LCompute:     TVdxCompute;
  LReader:      TVdxGGUFReader;
  LLN:          TVdxLayerNorm;
  LAttn:        TVdxAttention;
  LNormW:       TVdxNormLayerWeights;
  LWeights:     TVdxAttnLayerWeights;
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
  Section('Forward smoke (Gemma 3 4B, layer 0, pos 0)');

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

      LLN   := TVdxLayerNorm.Create();
      LAttn := TVdxAttention.Create();
      try
        LLN.SetErrors(LCompute.GetErrors());
        LAttn.SetErrors(LCompute.GetErrors());

        Check(LLN.Init(LCompute, 1e-6),
          'LayerNorm.Init');
        Check(LAttn.Init(LCompute,
          CHiddenDim, CNumQHeads, CNumKVHeads, CHeadDim,
          CNumLayers, CMaxSeqLen, CFFNWidth),
          'Attention.Init');
        FlushErrors(LAttn.GetErrors());

        Check(LLN.UploadNormWeights(LReader, 0, LNormW),
          'UploadNormWeights(0) — loads attn_q/k_norm among others');
        Check(LAttn.ResolveAttnWeights(LReader, 0, LWeights),
          'ResolveAttnWeights(0)');
        FlushErrors(LAttn.GetErrors());

        //------------------------------------------------------------
        // Input residual — small varied pattern so the signal doesn't
        // vanish under norms or saturate quantized matvecs. Host-
        // visible so we can upload without an intermediate staging
        // allocation.
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
            LAttn.Forward(
              LInputBuf,
              LWeights,
              LNormW.QNormGpu,
              LNormW.KNormGpu,
              0,        // layer index
              0,        // position
              10000.0,  // Gemma 3 sliding-window theta for layer 0
              LOutBuf);
          finally
            LCompute.EndBatch();
          end;
          Check(not LAttn.GetErrors().HasFatal(),
            'No fatal errors after Forward');
          FlushErrors(LAttn.GetErrors());

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
            'Output is not entirely zero (attention produced signal)');
        finally
          LCompute.DestroyGpuBuffer(LInputBuf);
          LCompute.DestroyGpuBuffer(LOutBuf);
        end;

        LLN.FreeNormWeights(LNormW);
      finally
        LAttn.Free();
        LLN.Free();
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
// layer 0, StartPos=0. Streams Q/K/V/O weight slices through the same
// 4-pair staging pool. Proves the batched path runs end-to-end without
// corruption and every element of the [4 x 2560] output is finite and
// at least partially non-zero.
// ---------------------------------------------------------------------------
procedure TAttentionTest.SecForwardBatchSmoke();
const
  CNumTokens: UInt32 = 4;
var
  LCompute:    TVdxCompute;
  LReader:     TVdxGGUFReader;
  LLN:         TVdxLayerNorm;
  LAttn:       TVdxAttention;
  LNormW:      TVdxNormLayerWeights;
  LWeights:    TVdxAttnLayerWeights;
  LInputMat:   TVdxGpuBuffer;
  LQMat:       TVdxGpuBuffer;
  LKMat:       TVdxGpuBuffer;
  LVMat:       TVdxGpuBuffer;
  LOutMat:     TVdxGpuBuffer;
  LInputBytes: UInt64;
  LQBytes:     UInt64;
  LKVBytes:    UInt64;
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
  LQBytes     := UInt64(CNumTokens) * (CNumQHeads  * CHeadDim) * SizeOf(Single);
  LKVBytes    := UInt64(CNumTokens) * (CNumKVHeads * CHeadDim) * SizeOf(Single);

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

      LLN   := TVdxLayerNorm.Create();
      LAttn := TVdxAttention.Create();
      try
        LLN.SetErrors(LCompute.GetErrors());
        LAttn.SetErrors(LCompute.GetErrors());

        Check(LLN.Init(LCompute, 1e-6),
          'LayerNorm.Init');
        Check(LAttn.Init(LCompute,
          CHiddenDim, CNumQHeads, CNumKVHeads, CHeadDim,
          CNumLayers, CMaxSeqLen, CFFNWidth),
          'Attention.Init');
        FlushErrors(LAttn.GetErrors());

        Check(LLN.UploadNormWeights(LReader, 0, LNormW),
          'UploadNormWeights(0)');
        Check(LAttn.ResolveAttnWeights(LReader, 0, LWeights),
          'ResolveAttnWeights(0)');
        FlushErrors(LAttn.GetErrors());

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
        LQMat := LCompute.CreateGpuBuffer(LQBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          vcBuffer);
        LKMat := LCompute.CreateGpuBuffer(LKVBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          vcBuffer);
        LVMat := LCompute.CreateGpuBuffer(LKVBytes,
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
            'Upload input matrix (4 tokens x 2560)');
          FlushErrors(LCompute.GetErrors());

          LCompute.BeginBatch();
          try
            LAttn.ForwardBatch(
              LInputMat,
              LWeights,
              LNormW.QNormGpu,
              LNormW.KNormGpu,
              0,        // layer
              CNumTokens,
              0,        // start pos — fresh prefill
              10000.0,  // sliding-window theta
              LQMat, LKMat, LVMat,
              LOutMat,
              False);   // decoder causal (not bidir)
          finally
            LCompute.EndBatch();
          end;
          Check(not LAttn.GetErrors().HasFatal(),
            'No fatal errors after ForwardBatch');
          FlushErrors(LAttn.GetErrors());

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
          LCompute.DestroyGpuBuffer(LQMat);
          LCompute.DestroyGpuBuffer(LKMat);
          LCompute.DestroyGpuBuffer(LVMat);
          LCompute.DestroyGpuBuffer(LOutMat);
        end;

        LLN.FreeNormWeights(LNormW);
      finally
        LAttn.Free();
        LLN.Free();
      end;
    finally
      LReader.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

end.
