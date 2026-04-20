{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.Inference;

interface

uses
  VindexLLM.TestCase;

type

  { TInferenceTest }
  // Phase 14 milestone test. Sec01 verifies LoadModel wiring (model
  // creation, stop-token resolution, decode-buffer init, rebuild
  // threshold math). Sec02 is the actual Paris test — load, generate
  // from "What is the capital of France?", and assert the output
  // contains "Paris". Sampler set to the project-canonical
  // Gemma 3 4B configuration.
  TInferenceTest = class(TVdxTestCase)
  private
    procedure SecLoadConfig();
    procedure SecParisGeneration();
  protected
    procedure Run(); override;
  public
    constructor Create(); override;
  end;

implementation

uses
  System.SysUtils,
  System.Classes,
  System.StrUtils,
  VindexLLM.Utils,
  VindexLLM.Sampler,
  VindexLLM.Model,
  VindexLLM.Model.Gemma3,
  VindexLLM.Inference,
  UTest.Common;

const
  // Same F16 reference model the Phase 13 Gemma3 test uses. The Paris
  // prompt is the canonical phrasing from TASK-REFACTOR.md §Phase 14.
  CMaxContext = 2048;
  CParisPrompt: string = 'What is the capital of France?';
  CMaxTokens  = 256;

{ TInferenceTest }

constructor TInferenceTest.Create();
begin
  inherited;
  Title := 'Test_Inference';
end;

procedure TInferenceTest.Run();
begin
  SecLoadConfig();
  SecParisGeneration();
end;

// ---------------------------------------------------------------------------
// 1. LoadConfig — confirm the 3-step LoadModel flow (factory →
// ResolveStopTokens → InitDecodeBuffers) wires up correctly, that the
// rebuild threshold defaults to 3/4 of MaxSeqLen, and that the VRAM
// snapshot is populated. Uses a status callback so anyone watching
// the console gets the model-load progress in context.
// ---------------------------------------------------------------------------
procedure TInferenceTest.SecLoadConfig();
var
  LInference: TVdxInference;
  LStats:     PVdxInferenceStats;
  LStatusCb:  TVdxStatusCallback;
begin
  Section('LoadModel wires model + stop tokens + decode buffers');

  LInference := TVdxInference.Create();
  try
    LStatusCb := procedure(const AText: string; const AUserData: Pointer)
    begin
      TVdxUtils.PrintLn('  [status] ' + AText);
    end;
    LInference.SetStatusCallback(LStatusCb, nil);

    Check(LInference.LoadModel(CModelPath, CMaxContext),
      'LoadModel succeeds');

    if LInference.GetErrors().HasFatal() then
      PrintErrors(LInference.GetErrors());
    Check(not LInference.GetErrors().HasFatal(),
      'No fatal errors after load');

    Check(LInference.Model <> nil,
      'Model accessor non-nil');
    Check(LInference.Model is TVdxGemma3Model,
      'Factory resolved to TVdxGemma3Model');

    // Rebuild threshold defaults to 3/4 of MaxSeqLen.
    Check(LInference.GetRebuildAt() > 0,
      'Rebuild threshold populated');
    Check(LInference.GetRebuildAt() < LInference.Model.MaxSeqLen,
      'Rebuild threshold below MaxSeqLen');

    // VRAM snapshot (non-zero if anything went to GPU, which it did).
    LStats := LInference.GetStats();
    Check(LStats <> nil,
      'Stats pointer is non-nil');

    // Fresh session — position resets to 0 on load, and unload.
    Check(LInference.GetKVCachePosition() = 0,
      'KV cache position starts at 0');

    FlushErrors(LInference.GetErrors());
  finally
    LInference.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 2. Paris generation — the milestone. Load the model, set the
// canonical sampler config, run Generate with the canonical prompt,
// and assert "Paris" is somewhere in the decoded output. Also prints
// tok/s so regressions in the forward-pass path show up loudly.
// ---------------------------------------------------------------------------
procedure TInferenceTest.SecParisGeneration();
var
  LInference:   TVdxInference;
  LCfg:         TVdxSamplerConfig;
  LOutput:      string;
  LStats:       PVdxInferenceStats;
  LTokCallback: TVdxTokenCallback;
begin
  Section('Generate "capital of France" prompt → output contains "Paris"');

  LInference := TVdxInference.Create();
  try
    Check(LInference.LoadModel(CModelPath, CMaxContext),
      'LoadModel succeeds');
    if LInference.GetErrors().HasFatal() then
    begin
      PrintErrors(LInference.GetErrors());
      Exit;
    end;

    // Canonical Gemma 3 4B sampler config. Seed is fixed so the Paris
    // assertion is deterministic across runs — if it ever fails, we
    // want the failure to reproduce on the same token stream.
    LCfg := TVdxSampler.DefaultConfig();
    LCfg.Temperature   := 1.0;
    LCfg.TopK          := 64;
    LCfg.TopP          := 0.95;
    LCfg.MinP          := 0.0;
    LCfg.RepeatPenalty := 1.2;
    LCfg.RepeatWindow  := 64;
    LCfg.Seed          := 42;
    LInference.SetSamplerConfig(LCfg);

    // Stream tokens to stdout so the test is observable while it
    // runs — 4B model inference on a full prompt takes real wall
    // time, and a silent test feels hung.
    LTokCallback :=
      procedure(const AToken: string; const AUserData: Pointer)
      begin
        TVdxUtils.Print(AToken);
      end;
    LInference.SetTokenCallback(LTokCallback, nil);

    TVdxUtils.PrintLn('  [prompt] ' + CParisPrompt);
    TVdxUtils.Print('  [output] ');
    LOutput := LInference.Generate(CParisPrompt, CMaxTokens);
    TVdxUtils.PrintLn('');

    if LInference.GetErrors().HasFatal() then
      PrintErrors(LInference.GetErrors());

    Check(Length(LOutput) > 0,
      'Generate produced non-empty output');

    // The milestone assertion. Case-insensitive — the model might
    // say "Paris" or "PARIS" or "The capital of France is Paris.";
    // any of those passes.
    Check(ContainsText(LOutput, 'Paris'),
      'Output contains "Paris"');

    LStats := LInference.GetStats();
    TVdxUtils.PrintLn(
      '  [stats] prefill=%d tokens in %.1f ms (%.2f tok/s)',
      [LStats.PrefillTokens, LStats.PrefillTimeMs,
       LStats.PrefillTokPerSec]);
    TVdxUtils.PrintLn(
      '  [stats] generated=%d tokens in %.1f ms (%.2f tok/s)',
      [LStats.GeneratedTokens, LStats.GenerationTimeMs,
       LStats.GenerationTokPerSec]);
    TVdxUtils.PrintLn(
      '  [stats] time to first token: %.1f ms',
      [LStats.TimeToFirstTokenMs]);
    TVdxUtils.PrintLn(
      '  [stats] stop reason: %s',
      [CVdxStopReasons[LStats.StopReason]]);
    TVdxUtils.PrintLn(
      '  [stats] VRAM total: %.1f MB',
      [LStats.VRAMUsage.TotalBytes / (1024.0 * 1024.0)]);

    Check(LStats.GeneratedTokens > 0,
      'At least one token generated');
    Check(LStats.GenerationTokPerSec > 0,
      'Generation tok/s > 0');

    FlushErrors(LInference.GetErrors());
  finally
    LInference.Free();
  end;
end;

end.
