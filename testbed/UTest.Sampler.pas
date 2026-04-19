{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.Sampler;

interface

uses
  VindexLLM.TestCase,
  VindexLLM.Sampler;

type

  { TSamplerTest }
  TSamplerTest = class(TVdxTestCase)
  private
    procedure SecCreateDestroy();
    procedure SecGreedyArgmax();
    procedure SecDeterministicReseed();
    procedure SecTopKBound();
    procedure SecRepeatPenalty();
  protected
    procedure Run(); override;
  public
    constructor Create(); override;
  end;

implementation

uses
  System.SysUtils;

const
  // Vocabulary size used by every section — small enough to inspect
  // by eye, large enough that non-trivial sampling is exercised.
  CVocabSize = 10;

{ TSamplerTest }

constructor TSamplerTest.Create();
begin
  inherited;
  Title := 'Test_Sampler';
end;

procedure TSamplerTest.Run();
begin
  SecCreateDestroy();
  SecGreedyArgmax();
  SecDeterministicReseed();
  SecTopKBound();
  SecRepeatPenalty();
end;

// ---------------------------------------------------------------------------
// SecCreateDestroy — build a sampler with the default constructor, verify
// the default config matches the spec (greedy / no penalty / 64-slot
// repeat window / seed 0), and free it cleanly.
// ---------------------------------------------------------------------------
procedure TSamplerTest.SecCreateDestroy();
var
  LSampler: TVdxSampler;
  LCfg: TVdxSamplerConfig;
begin
  Section('Create / Destroy + default config');

  LSampler := TVdxSampler.Create();
  try
    Check(LSampler <> nil, 'Create returned a non-nil instance');

    LCfg := LSampler.GetConfig();
    Check(LCfg.Temperature = 0.0,    'Default Temperature = 0.0');
    Check(LCfg.TopK = 0,             'Default TopK = 0');
    Check(LCfg.TopP = 1.0,           'Default TopP = 1.0');
    Check(LCfg.MinP = 0.0,           'Default MinP = 0.0');
    Check(LCfg.RepeatPenalty = 1.0,  'Default RepeatPenalty = 1.0');
    Check(LCfg.RepeatWindow = 64,    'Default RepeatWindow = 64');
    Check(LCfg.Seed = 0,             'Default Seed = 0');

    FlushErrors(LSampler.GetErrors());
  finally
    LSampler.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecGreedyArgmax — Temperature = 0 must produce pure argmax regardless
// of PRNG seed. Plant two different max indices in succession and check
// both are returned.
// ---------------------------------------------------------------------------
procedure TSamplerTest.SecGreedyArgmax();
var
  LSampler: TVdxSampler;
  LLogits: array[0..CVocabSize - 1] of Single;
  LCfg: TVdxSamplerConfig;
  LI: Integer;
  LResult: Integer;
begin
  Section('Greedy argmax (Temperature = 0)');

  // Uniform floor of 0.1, single spike at index 7.
  for LI := 0 to CVocabSize - 1 do
    LLogits[LI] := 0.1;
  LLogits[7] := 5.0;

  LSampler := TVdxSampler.Create();
  try
    LCfg := TVdxSampler.DefaultConfig();
    LCfg.Temperature := 0.0;
    LSampler.SetConfig(LCfg);

    LResult := LSampler.Process(@LLogits[0], CVocabSize);
    Check(LResult = 7, 'Argmax returns index 7');

    // Move the max; greedy must track it.
    LLogits[7] := 0.1;
    LLogits[2] := 9.9;
    LResult := LSampler.Process(@LLogits[0], CVocabSize);
    Check(LResult = 2, 'Argmax tracks new max at index 2');

    FlushErrors(LSampler.GetErrors());
  finally
    LSampler.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecDeterministicReseed — build two fresh samplers with the same non-zero
// seed, draw a sequence from each over identical logits, and verify the
// two sequences are byte-identical. This exercises xoshiro256** + the
// SeedPRNG path through SetConfig (seed-changed branch).
// ---------------------------------------------------------------------------
procedure TSamplerTest.SecDeterministicReseed();
const
  CIters = 50;
var
  LSamplerA: TVdxSampler;
  LSamplerB: TVdxSampler;
  LLogits: array[0..CVocabSize - 1] of Single;
  LCfg: TVdxSamplerConfig;
  LSeqA: array[0..CIters - 1] of Integer;
  LSeqB: array[0..CIters - 1] of Integer;
  LI: Integer;
  LMatch: Boolean;
begin
  Section('Deterministic re-seed (seed = 42)');

  // Spread logits so the distribution is non-uniform but every token
  // has non-zero probability at Temperature = 1.
  for LI := 0 to CVocabSize - 1 do
    LLogits[LI] := LI * 0.25;

  LCfg := TVdxSampler.DefaultConfig();
  LCfg.Temperature := 1.0;
  LCfg.Seed := 42;

  LSamplerA := TVdxSampler.Create();
  try
    LSamplerA.SetConfig(LCfg);
    for LI := 0 to CIters - 1 do
      LSeqA[LI] := LSamplerA.Process(@LLogits[0], CVocabSize);
  finally
    LSamplerA.Free();
  end;

  LSamplerB := TVdxSampler.Create();
  try
    LSamplerB.SetConfig(LCfg);
    for LI := 0 to CIters - 1 do
      LSeqB[LI] := LSamplerB.Process(@LLogits[0], CVocabSize);
  finally
    LSamplerB.Free();
  end;

  LMatch := True;
  for LI := 0 to CIters - 1 do
    if LSeqA[LI] <> LSeqB[LI] then
    begin
      LMatch := False;
      Break;
    end;

  Check(LMatch,
    'Two seed=42 samplers produce byte-identical 50-token sequences');
end;

// ---------------------------------------------------------------------------
// SecTopKBound — with TopK = 3 and three clearly-dominant logits, every
// sampled token must land in the top-3 index set. Verifies the Top-K
// code path prunes the non-top-K entries to zero probability.
// ---------------------------------------------------------------------------
procedure TSamplerTest.SecTopKBound();
const
  CIters = 200;
  CTopK  = 3;
var
  LSampler: TVdxSampler;
  LLogits: array[0..CVocabSize - 1] of Single;
  LCfg: TVdxSamplerConfig;
  LI: Integer;
  LResult: Integer;
  LInBounds: Boolean;
begin
  Section('Top-K bound (K = 3 over 10 logits)');

  // Three dominant entries at 1, 4, 8 — delta large enough that the
  // remaining seven are never in the top 3.
  for LI := 0 to CVocabSize - 1 do
    LLogits[LI] := 0.0;
  LLogits[1] := 10.0;
  LLogits[4] := 10.0;
  LLogits[8] := 10.0;

  LCfg := TVdxSampler.DefaultConfig();
  LCfg.Temperature := 1.0;
  LCfg.TopK := CTopK;
  LCfg.Seed := 42;

  LSampler := TVdxSampler.Create();
  try
    LSampler.SetConfig(LCfg);

    LInBounds := True;
    for LI := 0 to CIters - 1 do
    begin
      LResult := LSampler.Process(@LLogits[0], CVocabSize);
      if (LResult <> 1) and (LResult <> 4) and (LResult <> 8) then
      begin
        LInBounds := False;
        Break;
      end;
    end;
    Check(LInBounds,
      'All 200 samples fall within top-3 index set {1, 4, 8}');

    FlushErrors(LSampler.GetErrors());
  finally
    LSampler.Free();
  end;
end;

// ---------------------------------------------------------------------------
// SecRepeatPenalty — with a uniform distribution, token 3 is sampled
// roughly 1/10 of the time. After seeding token 3 into history and
// enabling RepeatPenalty > 1, its observed frequency should drop. Also
// sanity-check the baseline isn't pathologically low.
//
// Process() modifies logits in place when the penalty is active, so
// every iteration refreshes LLogitsCopy from the pristine source.
// ---------------------------------------------------------------------------
procedure TSamplerTest.SecRepeatPenalty();
const
  CIters = 200;
var
  LSampler: TVdxSampler;
  LLogits: array[0..CVocabSize - 1] of Single;
  LLogitsCopy: array[0..CVocabSize - 1] of Single;
  LCfg: TVdxSamplerConfig;
  LI: Integer;
  LResult: Integer;
  LHits3Plain: Integer;
  LHits3Penalized: Integer;
begin
  Section('Repeat penalty reduces frequency of penalized token');

  // Uniform logits — every token has equal baseline probability.
  for LI := 0 to CVocabSize - 1 do
    LLogits[LI] := 1.0;

  // --- Baseline: no penalty ---
  LCfg := TVdxSampler.DefaultConfig();
  LCfg.Temperature := 1.0;
  LCfg.RepeatPenalty := 1.0;
  LCfg.Seed := 42;

  LHits3Plain := 0;
  LSampler := TVdxSampler.Create();
  try
    LSampler.SetConfig(LCfg);
    for LI := 0 to CIters - 1 do
    begin
      Move(LLogits, LLogitsCopy, SizeOf(LLogits));
      LResult := LSampler.Process(@LLogitsCopy[0], CVocabSize);
      if LResult = 3 then
        Inc(LHits3Plain);
    end;
  finally
    LSampler.Free();
  end;

  // --- With penalty: token 3 in history, penalty 2.0 ---
  LCfg.RepeatPenalty := 2.0;
  LCfg.Seed := 42;

  LHits3Penalized := 0;
  LSampler := TVdxSampler.Create();
  try
    LSampler.SetConfig(LCfg);
    LSampler.AddToHistory(3);
    for LI := 0 to CIters - 1 do
    begin
      Move(LLogits, LLogitsCopy, SizeOf(LLogits));
      LResult := LSampler.Process(@LLogitsCopy[0], CVocabSize);
      if LResult = 3 then
        Inc(LHits3Penalized);
    end;
  finally
    LSampler.Free();
  end;

  Check(LHits3Penalized < LHits3Plain,
    Format('Penalized token 3 picked less often (%d < %d)',
      [LHits3Penalized, LHits3Plain]));

  // Sanity — uniform baseline across 10 tokens, 200 draws: expected ~20.
  Check(LHits3Plain > 5,
    Format('Baseline hit count is non-degenerate (%d > 5)',
      [LHits3Plain]));
end;

end.
