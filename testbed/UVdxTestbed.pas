{===============================================================================
  VindexLLM� - Liberating LLM inference

  Copyright � 2026-present tinyBigGAMES� LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UVdxTestbed;

interface

procedure RunVdxTestbed();

implementation

uses
  WinAPI.Windows,
  System.SysUtils,
  System.IOUtils,
  System.Math,
  System.Generics.Collections,
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.Compute,
  VindexLLM.TurboQuant,
  VindexLLM.Inference,
  VindexLLM.Sampler;

procedure StatusCallback(const AText: string; const AUserData: Pointer);
begin
  TVdxUtils.PrintLn(AText);
end;

procedure PrintErrors(const AInference: TVdxInference);
var
  LErrors: TVdxErrors;
  LItems: TList<TVdxError>;
  LI: Integer;
  LErr: TVdxError;
  LColor: string;
  LLabel: string;
begin
  LErrors := AInference.GetErrors();
  if LErrors = nil then
    Exit;
  LItems := LErrors.GetItems();
  if LItems.Count = 0 then
    Exit;

  TVdxUtils.PrintLn('');
  for LI := 0 to LItems.Count - 1 do
  begin
    LErr := LItems[LI];
    case LErr.Severity of
      esHint:
      begin
        LColor := COLOR_CYAN;
        LLabel := 'HINT';
      end;
      esWarning:
      begin
        LColor := COLOR_YELLOW;
        LLabel := 'WARN';
      end;
      esError:
      begin
        LColor := COLOR_RED;
        LLabel := 'ERROR';
      end;
      esFatal:
      begin
        LColor := COLOR_MAGENTA;
        LLabel := 'FATAL';
      end;
    else
      LColor := COLOR_WHITE;
      LLabel := '?';
    end;

    if LErr.Code <> '' then
      TVdxUtils.PrintLn(LColor + '[%s] %s: %s', [LLabel, LErr.Code, LErr.Message])
    else
      TVdxUtils.PrintLn(LColor + '[%s] %s', [LLabel, LErr.Message]);
  end;
end;

procedure InferenceEventCallback(const AEvent: TVdxInferenceEvent;
  const AUserData: Pointer);
begin
  if AEVent = ieGenerateEnd then
    TVdxUtils.PrintLn();

  TVdxUtils.PrintLn(COLOR_GREEN + '[event] %s', [CVdxEventNames[AEvent]]);
end;

function CancelCallback(const AUserData: Pointer): Boolean;
begin
  Result := (GetAsyncKeyState(VK_ESCAPE) and $8000) <> 0;
end;

procedure PrintToken(const AToken: string; const AUserData: Pointer);
begin
  Write(AToken);
end;

procedure PrintStats(const AStats: PVdxInferenceStats);
var
  LStopColor: string;
begin
  TVdxUtils.PrintLn();

  // Prefill line
  TVdxUtils.Print(COLOR_WHITE + 'Prefill:    ');
  TVdxUtils.Print(COLOR_CYAN + '%d tokens in %.0fms ', [
    AStats.PrefillTokens, AStats.PrefillTimeMs]);
  TVdxUtils.PrintLn(COLOR_GREEN + '(%.1f tok/s)', [AStats.PrefillTokPerSec]);

  // Generation line
  TVdxUtils.Print(COLOR_WHITE + 'Generation: ');
  TVdxUtils.Print(COLOR_CYAN + '%d tokens in %.0fms ', [
    AStats.GeneratedTokens, AStats.GenerationTimeMs]);
  TVdxUtils.PrintLn(COLOR_GREEN + '(%.1f tok/s)', [AStats.GenerationTokPerSec]);

  // Timing + stop reason
  case AStats.StopReason of
    srEOS,
    srStopToken:   LStopColor := COLOR_GREEN;
    srMaxTokens,
    srContextFull: LStopColor := COLOR_YELLOW;
    srCancelled:   LStopColor := COLOR_RED;
  else
    LStopColor := COLOR_WHITE;
  end;
  TVdxUtils.Print(COLOR_WHITE + 'TTFT: ');
  TVdxUtils.Print(COLOR_CYAN + '%.0fms', [AStats.TimeToFirstTokenMs]);
  TVdxUtils.Print(COLOR_WHITE + ' | Total: ');
  TVdxUtils.Print(COLOR_CYAN + '%.0fms', [AStats.TotalTimeMs]);
  TVdxUtils.Print(COLOR_WHITE + ' | Stop: ');
  TVdxUtils.PrintLn(LStopColor + '%s', [CVdxStopReasons[AStats.StopReason]]);

  // VRAM usage
  TVdxUtils.Print(COLOR_WHITE + 'VRAM: ');
  if AStats.VRAMUsage.TotalBytes > UInt64(10) * 1024 * 1024 * 1024 then
    TVdxUtils.Print(COLOR_YELLOW + '%d MB ', [AStats.VRAMUsage.TotalBytes div (1024 * 1024)])
  else
    TVdxUtils.Print(COLOR_GREEN + '%d MB ', [AStats.VRAMUsage.TotalBytes div (1024 * 1024)]);
  TVdxUtils.PrintLn(COLOR_CYAN + '(weights: %d, cache: %d, buffers: %d)', [
    AStats.VRAMUsage.WeightsBytes div (1024 * 1024),
    AStats.VRAMUsage.CacheBytes div (1024 * 1024),
    AStats.VRAMUsage.BuffersBytes div (1024 * 1024)]);
end;

procedure Test01();
const
  CPrompt =
  '''
    Explain the differences between these three sorting algorithms: bubble sort,
    merge sort, and quicksort. For each one, describe how it works step by step,
    give the best-case and worst-case time complexity using big-O notation,
    explain when you would choose it over the others, and provide a real-world
    analogy that helps illustrate the concept. Also discuss whether each
    algorithm is stable or unstable, and what that means in practice.
  ''';
var
  LInference: TVdxInference;
  LConfig: TVdxSamplerConfig;
  LLoaded: Boolean;
begin
  LInference := TVdxInference.Create();
  try
    LInference.SetStatusCallback(StatusCallback, nil);
    LInference.SetTokenCallback(PrintToken, nil);
    LInference.SetInferenceEventCallback(InferenceEventCallback, nil);
    LInference.SetCancelCallback(CancelCallback, nil);

    LLoaded := LInference.LoadModel('C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.Q8_0.gguf');
    try
      PrintErrors(LInference);
      if not LLoaded then
        Exit;

      // recommended sampling config for gemma-3-4b-it-null-space-abliterated
      LConfig := TVdxSampler.DefaultConfig();
      LConfig.Temperature := 1.0;
      LConfig.TopK := 64;
      LConfig.TopP := 0.95;
      LConfig.RepeatPenalty := 1.2;
      LConfig.RepeatWindow := 64;
      LConfig.Seed := 0;  // 0 = random each run, >0 = reproducible output
      LInference.SetSamplerConfig(LConfig);

      LInference.Generate('create a short story about an AI', 1024);
      PrintErrors(LInference);
      PrintStats(LInference.GetStats());

      finally
        LInference.UnloadModel();
      end;
  finally
    LInference.Free();
  end;
end;

procedure RunVdxTestbed();
var
  LIndex: Integer;
begin
  try
    LIndex := 1;

    case LIndex of
      1: Test01();
    end;
  except
    on E: Exception do
    begin
      TVdxUtils.PrintLn('');
      TVdxUtils.PrintLn(COLOR_RED + 'EXCEPTION: %s', [E.Message]);
    end;
  end;

  if TVdxUtils.RunFromIDE() then
    TVdxUtils.Pause();
end;

end.
