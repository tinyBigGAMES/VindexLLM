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
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.Compute,
  VindexLLM.TurboQuant,
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
begin
  TVdxUtils.PrintLn();
  TVdxUtils.PrintLn('Prefill:    %d tokens in %.0fms (%.1f tok/s)', [
    AStats.PrefillTokens, AStats.PrefillTimeMs, AStats.PrefillTokPerSec]);
  TVdxUtils.PrintLn('Generation: %d tokens in %.0fms (%.1f tok/s)', [
    AStats.GeneratedTokens, AStats.GenerationTimeMs, AStats.GenerationTokPerSec]);
  TVdxUtils.PrintLn('TTFT: %.0fms | Total: %.0fms | Stop: %s', [
    AStats.TimeToFirstTokenMs, AStats.TotalTimeMs,
    CVdxStopReasons[AStats.StopReason]]);
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
    //LInference.LoadModel('C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.f16.gguf');
    LInference.LoadModel('C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.Q8_0.gguf');
    LInference.SetTokenCallback(PrintToken, nil);
    //LInference.Generate(CPrompt);
    LInference.Generate('how to make kno3?', 512);
    PrintStats(LInference.GetStats());
    LInference.UnloadModel();
  finally
    LInference.Free();
  end;
end;

procedure Test02();
const
  CNumBlocks = 256;   // 256 blocks x 32 = 8192 floats
  CNumFloats = CNumBlocks * CTQ3BlockSize;
var
  LOriginal: array of Single;
  LRestored: array of Single;
  LBlock: TVdxTQ3Block;
  LI: Integer;
  LB: Integer;
  LMSE: Double;
  LMaxErr: Double;
  LErr: Double;
begin
  TVdxUtils.PrintLn(COLOR_CYAN + '=== TurboQuant TQ3 Round-Trip Test (CPU) ===');

  // Generate random data with realistic distribution
  SetLength(LOriginal, CNumFloats);
  SetLength(LRestored, CNumFloats);
  RandSeed := 42;
  for LI := 0 to CNumFloats - 1 do
    LOriginal[LI] := (Random - 0.5) * 4.0;  // uniform [-2, +2]

  // Round-trip: quantize then dequantize each block
  for LB := 0 to CNumBlocks - 1 do
  begin
    TVdxTurboQuant.QuantizeBlockCPU(@LOriginal[LB * CTQ3BlockSize], LBlock);
    TVdxTurboQuant.DequantizeBlockCPU(LBlock, @LRestored[LB * CTQ3BlockSize]);
  end;

  // Compute MSE and max error
  LMSE := TVdxTurboQuant.ComputeMSE(@LOriginal[0], @LRestored[0], CNumFloats);
  LMaxErr := 0.0;
  for LI := 0 to CNumFloats - 1 do
  begin
    LErr := Abs(LOriginal[LI] - LRestored[LI]);
    if LErr > LMaxErr then
      LMaxErr := LErr;
  end;

  TVdxUtils.PrintLn('  Blocks:    %d (%d floats)', [CNumBlocks, CNumFloats]);
  TVdxUtils.PrintLn('  MSE:       %.6f (target ~0.034)', [LMSE]);
  TVdxUtils.PrintLn('  Max error: %.6f', [LMaxErr]);

  if LMSE < 0.1 then
    TVdxUtils.PrintLn(COLOR_GREEN + '  PASS: MSE within expected range')
  else
    TVdxUtils.PrintLn(COLOR_RED + '  FAIL: MSE too high');
end;

procedure RunVdxTestbed();
var
  LIndex: Integer;
begin
  try
    LIndex := 2;

    case LIndex of
      1: Test01();
      2: Test02();
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
