{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.Common;

interface

uses
  VindexLLM.Utils,
  VindexLLM.Inference,
  VindexLLM.TokenWriter;

const
  // Path to the main inference model — Gemma 3 4B instruction-tuned,
  // Handles text generation for chat responses.
  CModelPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-q4_0.gguf';

  // Path to the embedding model — EmbeddingGemma 300M, Q8_0 quantized.
  // Used by TVdxMemory to embed turns for cosine-similarity vector search.
  // Set to '' to disable vector search and use FTS5 keyword search only.
  CEmbedderPath = 'C:\Dev\LLM\GGUF\embeddinggemma-300m-qat-Q8_0.gguf';

  CFuncModelPath = 'C:\Dev\LLM\GGUF\functiongemma-270m-it-q8_0.gguf';


  CHiddenDim: UInt32 = 2560;

var
  GTokenWriter: TVdxConsoleTokenWriter;

procedure StatusCallback(const AText: string; const AUserData: Pointer);
procedure PrintErrors(const AInference: TVdxInference);
procedure InferenceEventCallback(const AEvent: TVdxInferenceEvent;
  const AUserData: Pointer);
function  CancelCallback(const AUserData: Pointer): Boolean;
procedure PrintToken(const AToken: string; const AUserData: Pointer);
procedure PrintStats(const AStats: PVdxInferenceStats);

implementation

uses
  WinAPI.Windows,
  System.Generics.Collections;

// ---------------------------------------------------------------------------
// StatusCallback — receives progress messages from the inference engine during
// model loading (weight uploads, config detection, tokenizer init, VRAM usage)
// and prints them to the console. Assigned via SetStatusCallback().
// ---------------------------------------------------------------------------
procedure StatusCallback(const AText: string; const AUserData: Pointer);
begin
  TVdxUtils.PrintLn(AText);
end;

// ---------------------------------------------------------------------------
// PrintErrors — reads the error collection from the inference engine and prints
// each error with color-coded severity (HINT=cyan, WARN=yellow, ERROR=red,
// FATAL=magenta). Called after LoadModel() and Generate() to surface any issues.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// InferenceEventCallback — receives lifecycle events from the inference engine:
//   ieLoadStart/End      — model loading (weight upload, pipeline creation)
//   ieUnloadStart/End    — model unloading (GPU resource cleanup)
//   iePrefillStart/End   — batched prefill of all prompt tokens
//   ieGenerateStart/End  — autoregressive token-by-token generation
// Prints each event name in green. Adds a newline before ieGenerateEnd so the
// stats output doesn't collide with the last streamed token.
// ---------------------------------------------------------------------------
procedure InferenceEventCallback(const AEvent: TVdxInferenceEvent;
  const AUserData: Pointer);
begin
  if AEVent = ieGenerateEnd then
    TVdxUtils.PrintLn();

  TVdxUtils.PrintLn(COLOR_GREEN + '[event] %s', [CVdxEventNames[AEvent]]);
end;

// ---------------------------------------------------------------------------
// CancelCallback — polled by the inference engine before each layer's forward
// pass (both during prefill and generation). Returns True if ESC is held,
// which causes the engine to stop immediately with srCancelled stop reason.
// This gives the user a way to abort long generations without killing the app.
// ---------------------------------------------------------------------------
function CancelCallback(const AUserData: Pointer): Boolean;
begin
  Result := (GetAsyncKeyState(VK_ESCAPE) and $8000) <> 0;
end;

// ---------------------------------------------------------------------------
// PrintToken — called by the inference engine each time a new token is decoded
// during generation. Writes the token text to stdout without a newline, giving
// a streaming "typewriter" effect as the model produces output in real time.
// ---------------------------------------------------------------------------
procedure PrintToken(const AToken: string; const AUserData: Pointer);
begin
  GTokenWriter.Write(AToken);
end;

// ---------------------------------------------------------------------------
// PrintStats — displays a formatted summary of the last Generate() call:
//   - Prefill: how many prompt tokens were processed, time, throughput (tok/s)
//   - Generation: how many tokens were produced, time, throughput (tok/s)
//   - TTFT (time to first token), total wall-clock time, stop reason
//   - VRAM usage breakdown: weights, KV cache, work buffers (in MB)
// Stop reason is color-coded: green=normal (EOS/stop token), yellow=limit
// reached (max tokens/context full), red=cancelled by user.
// ---------------------------------------------------------------------------
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

end.
