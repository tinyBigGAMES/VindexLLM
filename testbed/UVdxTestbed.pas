{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
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
  Write(AToken);
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

// ===========================================================================
// Test01 — Full inference demo
//
// This test demonstrates the complete VindexLLM inference pipeline:
//
//   1. Create the inference engine
//   2. Register callbacks (status, token streaming, events, cancel)
//   3. Load a Gemma 3 4B GGUF model from disk
//      - Opens and memory-maps the GGUF file
//      - Reads model architecture and dimensions from GGUF metadata
//      - Initializes Vulkan (loads vulkan-1.dll, creates device + compute queue)
//      - Creates all compute shader pipelines (30 shaders, embedded as resources)
//      - Uploads all weights to GPU VRAM via staging buffers:
//        attention (Q/K/V/O), FFN (gate/up/down), norms, embeddings
//      - Allocates TQ3-compressed KV cache for all 34 layers
//      - Loads BPE tokenizer vocabulary from GGUF
//      - Detects stop tokens (EOS, end-of-turn) from vocab
//      - Reports VRAM usage breakdown
//   4. Configure the token sampler
//   5. Generate text from a prompt
//      - Formats the prompt using Gemma 3 chat template
//      - Tokenizes with BPE (adds BOS token)
//      - Batched prefill: all prompt tokens processed in parallel
//        (matmul shaders for attention + FFN, causal mask for attention)
//      - Autoregressive generation loop:
//        each token → embed → 34 layers (attention + FFN) → unembed → sample
//      - Tokens are streamed to console via PrintToken callback as produced
//      - Stops on: EOS token, end-of-turn token, max tokens, context full,
//        or user pressing ESC (cancel callback)
//   6. Print errors (if any) and performance stats
//   7. Unload model (frees all GPU resources, closes GGUF)
//
// To use a different model, change the path in LoadModel(). The model must
// be a Gemma 3 4B GGUF in F16 or Q8_0 format. See the README for vetted
// model download links.
//
// To change generation behavior, adjust the sampler config:
//   Temperature=0 → greedy (always picks highest probability token)
//   Temperature>0 → sampling with randomness (higher = more creative)
//   TopK          → only consider the K most likely tokens
//   TopP          → nucleus sampling (cumulative probability threshold)
//   MinP          → floor relative to top token probability
//   RepeatPenalty → penalize tokens that appeared recently (>1.0 to enable)
//   Seed=0        → random each run, Seed>0 → deterministic/reproducible
// ===========================================================================
procedure Test01();
const
  // The prompt to send to the model. This is wrapped in Gemma 3 chat template
  // formatting internally (<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n)
  // so you just provide the raw user message here.
  CPrompt1 =
  '''
  You are a senior software architect. A junior developer asks: "Why can't we just
  put everything in one big function?" Give a clear, practical explanation using a
  real-world analogy, then show a short before-and-after code example in Python.
  Keep it under 400 words.
  ''';

  CPrompt2 =
  '''
  Compare three famous ancient cities — Rome, Chang'an, and Tenochtitlan — at the
  height of their power. For each, describe what a visitor would see walking through
  the main street, the population, and one thing that would surprise a modern person.
  Format as three short sections.
  ''';

  CPrompt3 =
  '''
  Creae a short story about an AI that becomes self-aware.
  ''';

var
  LInference: TVdxInference;
  LConfig: TVdxSamplerConfig;
  LLoaded: Boolean;
  LPrompt: string;
begin
  // --- Step 1: Create the inference engine ---
  // TVdxInference is the main orchestrator. It owns all subsystems (Vulkan
  // compute, attention, norms, tokenizer, sampler) and manages their lifecycle.
  LInference := TVdxInference.Create();
  try
    // --- Step 2: Register callbacks ---
    // StatusCallback:  receives progress messages during model loading
    // PrintToken:      streams each generated token to console in real time
    // EventCallback:   notifies on prefill/generate start/end transitions
    // CancelCallback:  polled per-layer; return True (ESC key) to abort
    LInference.SetStatusCallback(StatusCallback, nil);
    LInference.SetTokenCallback(PrintToken, nil);
    LInference.SetInferenceEventCallback(InferenceEventCallback, nil);
    LInference.SetCancelCallback(CancelCallback, nil);

    // --- Step 3: Load the model ---
    // This is the heavy operation. It will:
    //   - Memory-map the GGUF file (zero-copy access to weight data)
    //   - Detect architecture from GGUF metadata (must be "gemma3")
    //   - Read model dimensions (layers, hidden_dim, ffn_width, head counts)
    //   - Initialize Vulkan (find GPU, create device, compute queue)
    //   - Create all 30 compute shader pipelines from embedded SPIR-V
    //   - Upload ~4-8 GB of weights to GPU VRAM via staging buffers
    //   - Allocate TQ3-compressed KV cache (10.7x smaller than F32)
    //   - Load BPE tokenizer vocabulary directly from GGUF
    //   - Report total VRAM usage via status callback
    // Returns False if anything fails (wrong architecture, missing tensors, etc.)
    LLoaded := LInference.LoadModel('C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.Q8_0.gguf');

    try
      // Check for errors from model loading (architecture mismatch, missing
      // tensors, Vulkan init failure, etc.)
      PrintErrors(LInference);
      if not LLoaded then
        Exit;

      // --- Step 4: Configure the token sampler ---
      // Start from defaults (Temperature=0 = greedy argmax) then override
      // with recommended settings for this specific model variant.
      // These values are what is recommend by Google for gemma-3-4b-it:
      LConfig := TVdxSampler.DefaultConfig();
      LConfig.Temperature := 1.0;
      LConfig.TopK := 64;
      LConfig.TopP := 0.95;
      Lconfig.MinP := 0.0;
      LConfig.RepeatPenalty := 1.0;
      LConfig.RepeatWindow := 64;
      LConfig.Seed := 0;
      LInference.SetSamplerConfig(LConfig);

      // --- Step 5: Generate text ---
      // Generate() does the full pipeline:
      //   1. Format prompt with Gemma 3 chat template
      //   2. Tokenize (BPE encode with BOS token)
      //   3. Batched prefill — all prompt tokens in parallel through 34 layers
      //   4. Autoregressive generation — one token at a time, up to 1024 tokens
      //   5. Each token is decoded and sent to PrintToken callback (streaming output)
      //   6. Stops when: EOS, <end_of_turn>, 1024 tokens reached, context full, or ESC
      // The return value is the complete generated string (same text that was streamed).
      LPrompt := CPrompt3;
      LInference.Generate(LPrompt, 1024);

      // Check for errors from generation (prompt too long, context overflow, etc.)
      PrintErrors(LInference);

      // --- Step 6: Print performance stats ---
      // Shows prefill throughput, generation throughput, TTFT, stop reason,
      // and VRAM usage breakdown (weights, KV cache, work buffers)
      PrintStats(LInference.GetStats());

      finally
        // --- Step 7: Unload model ---
        // Frees all GPU resources: shader pipelines, descriptor sets/pools,
        // weight buffers, KV cache, work buffers, embedding table copy.
        // Closes the memory-mapped GGUF file. Destroys all subsystem objects.
        // After this call, the inference engine can load a different model.
        LInference.UnloadModel();
      end;
  finally
    // Destroy the inference engine itself
    LInference.Free();
  end;
end;

// ---------------------------------------------------------------------------
// RunVdxTestbed — entry point for the testbed application.
// Selects which test to run via LIndex, wraps in top-level exception handler,
// and pauses for keypress when running from the Delphi IDE so you can read
// the console output before the window closes.
// ---------------------------------------------------------------------------
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
