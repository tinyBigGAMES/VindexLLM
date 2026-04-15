unit UVdxTestbed;

interface

procedure RunVdxTestbed();

implementation

uses
  WinAPI.Windows,
  System.SysUtils,
  System.IOUtils,
  System.Math,
  VindexLLM.Utils,
  VindexLLM.Inference;

procedure StatusCallback(const AText: string; const AUserData: Pointer);
begin
  TVdxUtils.PrintLn(AText);
end;

function PrintToken(const AToken: string; const AUserData: Pointer): Boolean;
begin
  Write(AToken);
  Result := (GetAsyncKeyState(VK_ESCAPE) and $8000) = 0;
end;

procedure PrintStats(const AStats: PVdxInferenceStats);
const
  CStopReasons: array[TVdxStopReason] of string = (
    'none', 'eos', 'stop_token', 'max_tokens', 'callback_stopped');
begin
  WriteLn;
  WriteLn(Format('Prefill:    %d tokens in %.0fms (%.1f tok/s)', [
    AStats.PrefillTokens, AStats.PrefillTimeMs, AStats.PrefillTokPerSec]));
  WriteLn(Format('Generation: %d tokens in %.0fms (%.1f tok/s)', [
    AStats.GeneratedTokens, AStats.GenerationTimeMs, AStats.GenerationTokPerSec]));
  WriteLn(Format('TTFT: %.0fms | Total: %.0fms | Stop: %s', [
    AStats.TimeToFirstTokenMs, AStats.TotalTimeMs,
    CStopReasons[AStats.StopReason]]));
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
begin
  TVdxUtils.Pause();
  LInference := TVdxInference.Create();
  try
    LInference.SetStatusCallback(StatusCallback, nil);
    //LInference.LoadModel('C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf');
    //LInference.LoadModel('C:\Dev\LLM\GGUF\gemma-3-4b-it-Q8_0.gguf');
    LInference.LoadModel('C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.Q8_0.gguf');
    LInference.SetTokenCallback(PrintToken, nil);
    //LInference.Generate('What is the capital of France?');
    //LInference.Generate('Who is bill gates?');
    LInference.Generate(CPrompt);
    PrintStats(LInference.GetStats());
    LInference.UnloadModel();
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
