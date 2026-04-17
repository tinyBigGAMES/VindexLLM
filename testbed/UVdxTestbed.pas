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
  FireDAC.Stan.Intf,
  FireDAC.Stan.Option,
  FireDAC.Stan.Error,
  FireDAC.Stan.Def,
  FireDAC.Stan.Async,
  FireDAC.DApt,
  FireDAC.Phys,
  FireDAC.Phys.SQLite,
  FireDAC.Phys.SQLiteWrapper,
  FireDAC.Phys.SQLiteWrapper.Stat,
  FireDAC.Comp.Client,
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.Compute,
  VindexLLM.TurboQuant,
  VindexLLM.Inference,
  VindexLLM.Sampler,
  VindexLLM.TokenWriter,
  VindexLLM.Memory,
  VindexLLM.Embeddings,
  VindexLLM.Session;

var
  GTokenWriter: TVdxConsoleTokenWriter;

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
  // --- Step 0: Create the token writer for word-wrapped streaming output ---
  GTokenWriter := TVdxConsoleTokenWriter.Create();
  try
    GTokenWriter.MaxWidth := 118;

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
        GTokenWriter.Reset();
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
  finally
    // Destroy the token writer
    GTokenWriter.Free();
    GTokenWriter := nil;
  end;
end;

// ===========================================================================
// Test02 — KV cache continuation across Generate() calls
//
// Verifies that FCurrentPosition tracking works end-to-end. The engine must
// preserve its KV cache across Generate() calls in the same loaded model so
// that a second call, without a BOS reset, sees the conversation history
// from the first call.
//
// Three phases:
//
//   PHASE 1 — SETUP
//     Seed the cache with a fact ("my favorite color is purple").
//     Position starts at 0, ends > 0.
//
//   PHASE 2 — RECALL (continuation, the actual test)
//     WITHOUT ResetKVCache, ask "what is my favorite color?". If position
//     tracking is correct the model sees the setup prompt as prior context
//     and answers "purple". If broken, the answer will be a guess or refusal.
//
//   PHASE 3 — CONTROL
//     Call ResetKVCache and ask the same question. Model must NOT know —
//     this proves Phase 2's correct answer came from the cache, not model
//     knowledge about the concept "favorite color."
//
// Greedy sampling (Temperature=0, TopK=1) with fixed seed is used so output
// is deterministic across runs. Any change in behavior is a real regression,
// not sampler noise.
// ===========================================================================
procedure Test02();
const
  CSetupPrompt =
  '''
  Please remember this important fact for our conversation: my favorite color
  is purple. Just reply "OK, I'll remember." and nothing else.
  ''';

  CRecallPrompt =
  '''
  What is my favorite color?
  ''';

var
  LInference: TVdxInference;
  LConfig: TVdxSamplerConfig;
  LLoaded: Boolean;

  procedure Banner(const AText: string);
  begin
    TVdxUtils.PrintLn();
    TVdxUtils.PrintLn(COLOR_YELLOW + '========================================================');
    TVdxUtils.PrintLn(COLOR_YELLOW + '  %s', [AText]);
    TVdxUtils.PrintLn(COLOR_YELLOW + '========================================================');
    TVdxUtils.PrintLn();
  end;

  procedure PrintPosition(const ALabel: string;
    const AInference: TVdxInference);
  begin
    TVdxUtils.PrintLn(COLOR_CYAN + '[%s] KV cache position: %d',
      [ALabel, AInference.GetKVCachePosition()]);
  end;

begin
  GTokenWriter := TVdxConsoleTokenWriter.Create();
  try
    GTokenWriter.MaxWidth := 118;

    LInference := TVdxInference.Create();
    try
      LInference.SetStatusCallback(StatusCallback, nil);
      LInference.SetTokenCallback(PrintToken, nil);
      LInference.SetInferenceEventCallback(InferenceEventCallback, nil);
      LInference.SetCancelCallback(CancelCallback, nil);

      LLoaded := LInference.LoadModel(
        'C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.Q8_0.gguf');
      try
        PrintErrors(LInference);
        if not LLoaded then
          Exit;

        // Greedy + fixed seed → deterministic outputs across runs so any
        // behavior change is genuinely a regression, not sampler noise.
        LConfig := TVdxSampler.DefaultConfig();
        LConfig.Temperature := 0.0;
        LConfig.TopK := 1;
        LConfig.TopP := 1.0;
        LConfig.MinP := 0.0;
        LConfig.RepeatPenalty := 1.0;
        LConfig.RepeatWindow := 64;
        LConfig.Seed := 1;
        LInference.SetSamplerConfig(LConfig);

        // --- Phase 1: Setup. Seed the cache with a fact. ---
        Banner('PHASE 1: SETUP - Teach the model a fact');
        PrintPosition('before setup', LInference);
        GTokenWriter.Reset();
        LInference.Generate(CSetupPrompt, 128);
        PrintErrors(LInference);
        PrintPosition('after  setup', LInference);
        PrintStats(LInference.GetStats());

        // --- Phase 2: Recall (continuation). Position > 0, NO reset. ---
        // If FCurrentPosition tracking works, the model sees the whole
        // conversation and should answer "purple".
        Banner('PHASE 2: RECALL (continuation) - Should remember "purple"');
        PrintPosition('before recall', LInference);
        GTokenWriter.Reset();
        LInference.Generate(CRecallPrompt, 128);
        PrintErrors(LInference);
        PrintPosition('after  recall', LInference);
        PrintStats(LInference.GetStats());

        // --- Phase 3: Control. Reset and ask the same question. ---
        // After ResetKVCache, the model has no memory of the setup prompt
        // and should NOT know the color.
        Banner('PHASE 3: CONTROL (after ResetKVCache) - Should NOT know');
        LInference.ResetKVCache();
        PrintPosition('after  reset', LInference);
        GTokenWriter.Reset();
        LInference.Generate(CRecallPrompt, 128);
        PrintErrors(LInference);
        PrintPosition('after  control', LInference);
        PrintStats(LInference.GetStats());

      finally
        LInference.UnloadModel();
      end;
    finally
      LInference.Free();
    end;
  finally
    GTokenWriter.Free();
    GTokenWriter := nil;
  end;
end;

// ===========================================================================
// Test03 — Save KV cache to disk (half 1 of the cross-process test)
//
// Loads the model, seeds the KV cache with a fact ("favorite color is
// purple"), then saves the cache to session.kvc and exits.
//
// Pair this with Test04 to prove cross-process persistence: run Test03
// once, close the app, then run Test04. If Test04's recall answers
// "purple", the saved file is truly self-sufficient — it survives full
// process exit, GPU VRAM teardown, and cold reinitialization.
// ===========================================================================
procedure Test03();
const
  CSetupPrompt =
  '''
  Please remember this important fact for our conversation: my favorite color
  is purple. Just reply "OK, I'll remember." and nothing else.
  ''';

  CCacheFile = 'session.kvc';

var
  LInference: TVdxInference;
  LConfig: TVdxSamplerConfig;
  LLoaded: Boolean;
  LSaved: Boolean;

  procedure Banner(const AText: string);
  begin
    TVdxUtils.PrintLn();
    TVdxUtils.PrintLn(COLOR_YELLOW + '========================================================');
    TVdxUtils.PrintLn(COLOR_YELLOW + '  %s', [AText]);
    TVdxUtils.PrintLn(COLOR_YELLOW + '========================================================');
    TVdxUtils.PrintLn();
  end;

  procedure PrintPosition(const ALabel: string;
    const AInference: TVdxInference);
  begin
    TVdxUtils.PrintLn(COLOR_CYAN + '[%s] KV cache position: %d',
      [ALabel, AInference.GetKVCachePosition()]);
  end;

begin
  GTokenWriter := TVdxConsoleTokenWriter.Create();
  try
    GTokenWriter.MaxWidth := 118;

    LInference := TVdxInference.Create();
    try
      LInference.SetStatusCallback(StatusCallback, nil);
      LInference.SetTokenCallback(PrintToken, nil);
      LInference.SetInferenceEventCallback(InferenceEventCallback, nil);
      LInference.SetCancelCallback(CancelCallback, nil);

      LLoaded := LInference.LoadModel(
        'C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.Q8_0.gguf');
      try
        PrintErrors(LInference);
        if not LLoaded then
          Exit;

        LConfig := TVdxSampler.DefaultConfig();
        LConfig.Temperature := 0.0;
        LConfig.TopK := 1;
        LConfig.TopP := 1.0;
        LConfig.MinP := 0.0;
        LConfig.RepeatPenalty := 1.0;
        LConfig.RepeatWindow := 64;
        LConfig.Seed := 1;
        LInference.SetSamplerConfig(LConfig);

        // --- Seed the cache ---
        Banner('SETUP - Teach the model a fact');
        PrintPosition('before setup', LInference);
        GTokenWriter.Reset();
        LInference.Generate(CSetupPrompt, 128);
        PrintErrors(LInference);
        PrintPosition('after  setup', LInference);
        PrintStats(LInference.GetStats());

        // --- Save to disk ---
        Banner('SAVE - Dump cache to disk');
        TVdxUtils.PrintLn(COLOR_WHITE + 'Saving to %s...', [CCacheFile]);
        LSaved := LInference.SaveKVCache(CCacheFile);
        PrintErrors(LInference);
        if LSaved then
        begin
          TVdxUtils.PrintLn(COLOR_GREEN + 'Saved successfully.');
          TVdxUtils.PrintLn();
          TVdxUtils.PrintLn(COLOR_YELLOW +
            'NOW: close this app, switch to Test04 (LIndex := 4),');
          TVdxUtils.PrintLn(COLOR_YELLOW +
            '     and run again to verify cross-process cache load.');
        end
        else
          TVdxUtils.PrintLn(COLOR_RED + 'Save failed.');

      finally
        LInference.UnloadModel();
      end;
    finally
      LInference.Free();
    end;
  finally
    GTokenWriter.Free();
    GTokenWriter := nil;
  end;
end;

// ===========================================================================
// Test04 — Load KV cache from disk and verify (half 2 of the cross-process test)
//
// Must be run in a FRESH process after Test03 has completed. Loads the model
// cold (KV caches are GPU-zero from allocation — no leftover state), does a
// control Generate to prove this fresh instance genuinely doesn't know the
// secret fact, then LoadKVCache('session.kvc') and asks the recall question.
//
// Expected outcome:
//   - Control phase: model says "I don't know" (no cross-process leak possible).
//   - Recall phase after load: model says "Purple." — proving session.kvc
//     alone fully restores a conversation across a cold process boundary.
// ===========================================================================
procedure Test04();
const
  CRecallPrompt =
  '''
  What is my favorite color?
  ''';

  CCacheFile = 'session.kvc';

var
  LInference: TVdxInference;
  LConfig: TVdxSamplerConfig;
  LLoaded: Boolean;
  LRestored: Boolean;

  procedure Banner(const AText: string);
  begin
    TVdxUtils.PrintLn();
    TVdxUtils.PrintLn(COLOR_YELLOW + '========================================================');
    TVdxUtils.PrintLn(COLOR_YELLOW + '  %s', [AText]);
    TVdxUtils.PrintLn(COLOR_YELLOW + '========================================================');
    TVdxUtils.PrintLn();
  end;

  procedure PrintPosition(const ALabel: string;
    const AInference: TVdxInference);
  begin
    TVdxUtils.PrintLn(COLOR_CYAN + '[%s] KV cache position: %d',
      [ALabel, AInference.GetKVCachePosition()]);
  end;

begin
  GTokenWriter := TVdxConsoleTokenWriter.Create();
  try
    GTokenWriter.MaxWidth := 118;

    LInference := TVdxInference.Create();
    try
      LInference.SetStatusCallback(StatusCallback, nil);
      LInference.SetTokenCallback(PrintToken, nil);
      LInference.SetInferenceEventCallback(InferenceEventCallback, nil);
      LInference.SetCancelCallback(CancelCallback, nil);

      LLoaded := LInference.LoadModel(
        'C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.Q8_0.gguf');
      try
        PrintErrors(LInference);
        if not LLoaded then
          Exit;

        LConfig := TVdxSampler.DefaultConfig();
        LConfig.Temperature := 0.0;
        LConfig.TopK := 1;
        LConfig.TopP := 1.0;
        LConfig.MinP := 0.0;
        LConfig.RepeatPenalty := 1.0;
        LConfig.RepeatWindow := 64;
        LConfig.Seed := 1;
        LInference.SetSamplerConfig(LConfig);

        // --- Control: prove the fresh process doesn't know ---
        Banner('CONTROL (fresh process) - Should NOT know the color');
        PrintPosition('before control', LInference);
        GTokenWriter.Reset();
        LInference.Generate(CRecallPrompt, 128);
        PrintErrors(LInference);
        PrintPosition('after  control', LInference);
        PrintStats(LInference.GetStats());

        // Reset so LoadKVCache gets a clean slate.
        LInference.ResetKVCache();

        // --- Load + Recall ---
        Banner('LOAD + RECALL - Restore from disk, should remember "purple"');
        TVdxUtils.PrintLn(COLOR_WHITE + 'Loading from %s...', [CCacheFile]);
        LRestored := LInference.LoadKVCache(CCacheFile);
        PrintErrors(LInference);
        if not LRestored then
        begin
          TVdxUtils.PrintLn(COLOR_RED + 'Load failed — aborting.');
          Exit;
        end;
        TVdxUtils.PrintLn(COLOR_GREEN + 'Loaded successfully.');
        PrintPosition('after  load', LInference);

        GTokenWriter.Reset();
        LInference.Generate(CRecallPrompt, 128);
        PrintErrors(LInference);
        PrintPosition('after  recall', LInference);
        PrintStats(LInference.GetStats());

      finally
        LInference.UnloadModel();
      end;
    finally
      LInference.Free();
    end;
  finally
    GTokenWriter.Free();
    GTokenWriter := nil;
  end;
end;

// ===========================================================================
// Test05 — Multi-turn continuation across the full conversation
//
// Must be run in a FRESH process after Test03 has saved session.kvc.
//
// Proves that once a cache is loaded, you can keep calling Generate()
// turn after turn. Each call appends its prompt + response to the cache,
// and subsequent calls see the entire accumulated conversation.
//
// Flow:
//   - Load session.kvc (which ended after the model said "OK, I'll remember.")
//   - TURN 1: ask an unrelated math question. Model should answer "4" (or
//     equivalent) without mentioning purple.
//   - TURN 2: ask the color recall question. Model must still know purple —
//     the math turn did not evict the original fact.
//   - TURN 3: ask what was discussed before the color recall. Model must
//     reference the math question (2+2), proving it sees the full history.
//
// Watch the position counter grow monotonically across all three turns.
// If any turn loses earlier context the test fails.
// ===========================================================================
procedure Test05();
const
  CMathPrompt =
  '''
  Quick question: what is 2 plus 2? Just give me the number.
  ''';

  CRecallPrompt =
  '''
  What is my favorite color?
  ''';

  CHistoryPrompt =
  '''
  Earlier in our conversation, you helped me with a math problem. What was
  the answer you gave me?
  ''';

  CCacheFile = 'session.kvc';

var
  LInference: TVdxInference;
  LConfig: TVdxSamplerConfig;
  LLoaded: Boolean;
  LRestored: Boolean;

  procedure Banner(const AText: string);
  begin
    TVdxUtils.PrintLn();
    TVdxUtils.PrintLn(COLOR_YELLOW + '========================================================');
    TVdxUtils.PrintLn(COLOR_YELLOW + '  %s', [AText]);
    TVdxUtils.PrintLn(COLOR_YELLOW + '========================================================');
    TVdxUtils.PrintLn();
  end;

  procedure PrintPosition(const ALabel: string;
    const AInference: TVdxInference);
  begin
    TVdxUtils.PrintLn(COLOR_CYAN + '[%s] KV cache position: %d',
      [ALabel, AInference.GetKVCachePosition()]);
  end;

begin
  GTokenWriter := TVdxConsoleTokenWriter.Create();
  try
    GTokenWriter.MaxWidth := 118;

    LInference := TVdxInference.Create();
    try
      LInference.SetStatusCallback(StatusCallback, nil);
      LInference.SetTokenCallback(PrintToken, nil);
      LInference.SetInferenceEventCallback(InferenceEventCallback, nil);
      LInference.SetCancelCallback(CancelCallback, nil);

      LLoaded := LInference.LoadModel(
        'C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.Q8_0.gguf');
      try
        PrintErrors(LInference);
        if not LLoaded then
          Exit;

        LConfig := TVdxSampler.DefaultConfig();
        LConfig.Temperature := 0.0;
        LConfig.TopK := 1;
        LConfig.TopP := 1.0;
        LConfig.MinP := 0.0;
        LConfig.RepeatPenalty := 1.0;
        LConfig.RepeatWindow := 64;
        LConfig.Seed := 1;
        LInference.SetSamplerConfig(LConfig);

        // --- Load the saved session ---
        Banner('LOAD - Restore session from disk');
        TVdxUtils.PrintLn(COLOR_WHITE + 'Loading from %s...', [CCacheFile]);
        LRestored := LInference.LoadKVCache(CCacheFile);
        PrintErrors(LInference);
        if not LRestored then
        begin
          TVdxUtils.PrintLn(COLOR_RED + 'Load failed — aborting.');
          Exit;
        end;
        TVdxUtils.PrintLn(COLOR_GREEN + 'Loaded successfully.');
        PrintPosition('after  load', LInference);

        // --- TURN 1: unrelated math question ---
        Banner('TURN 1: MATH - Unrelated question, should NOT mention purple');
        PrintPosition('before turn 1', LInference);
        GTokenWriter.Reset();
        LInference.Generate(CMathPrompt, 128);
        PrintErrors(LInference);
        PrintPosition('after  turn 1', LInference);
        PrintStats(LInference.GetStats());

        // --- TURN 2: color recall (should still work) ---
        Banner('TURN 2: COLOR RECALL - Must still remember "purple"');
        PrintPosition('before turn 2', LInference);
        GTokenWriter.Reset();
        LInference.Generate(CRecallPrompt, 128);
        PrintErrors(LInference);
        PrintPosition('after  turn 2', LInference);
        PrintStats(LInference.GetStats());

        // --- TURN 3: history probe (references a prior turn) ---
        Banner('TURN 3: HISTORY - Must reference the math question from turn 1');
        PrintPosition('before turn 3', LInference);
        GTokenWriter.Reset();
        LInference.Generate(CHistoryPrompt, 128);
        PrintErrors(LInference);
        PrintPosition('after  turn 3', LInference);
        PrintStats(LInference.GetStats());

      finally
        LInference.UnloadModel();
      end;
    finally
      LInference.Free();
    end;
  finally
    GTokenWriter.Free();
    GTokenWriter := nil;
  end;
end;

// ---------------------------------------------------------------------------
// Test06_FTS5Probe — check whether FireDAC's statically-linked SQLite has the
// FTS5 full-text-search module compiled in. Opens an in-memory database, asks
// SQLite to report its version + compile flags, then tries to actually create
// an FTS5 virtual table (the definitive test — compile flag is informational).
// Also dumps PRAGMA compile_options so we know everything else that's in.
// ---------------------------------------------------------------------------
procedure Test06_FTS5Probe();
var
  LLink: TFDPhysSQLiteDriverLink;
  LConn: TFDConnection;
  LQuery: TFDQuery;
  LFlagValue: Integer;
  LCreateOK: Boolean;
  LErrMsg: string;
begin
  TVdxUtils.PrintLn('FTS5 PROBE - FireDAC static SQLite capability check');

  LLink := TFDPhysSQLiteDriverLink.Create(nil);
  LConn := TFDConnection.Create(nil);
  LQuery := TFDQuery.Create(nil);
  try
    LLink.EngineLinkage := slStatic;

    LConn.DriverName := 'SQLite';
    LConn.Params.Values['Database'] := ':memory:';
    LConn.LoginPrompt := False;
    LConn.Open();

    LQuery.Connection := LConn;

    // 1. SQLite version
    LQuery.SQL.Text := 'SELECT sqlite_version()';
    LQuery.Open();
    TVdxUtils.PrintLn(COLOR_WHITE + 'SQLite version: %s',
      [LQuery.Fields[0].AsString]);
    LQuery.Close();

    // 2. Compile flag (informational)
    LQuery.SQL.Text := 'SELECT sqlite_compileoption_used(''ENABLE_FTS5'')';
    LQuery.Open();
    LFlagValue := LQuery.Fields[0].AsInteger;
    TVdxUtils.PrintLn(COLOR_WHITE + 'ENABLE_FTS5 compile flag: %d',
      [LFlagValue]);
    LQuery.Close();

    // 3. Real test — try to create an FTS5 virtual table
    LCreateOK := False;
    LErrMsg := '';
    try
      LConn.ExecSQL('CREATE VIRTUAL TABLE probe_fts USING fts5(content)');
      LCreateOK := True;
    except
      on E: Exception do
        LErrMsg := E.Message;
    end;

    TVdxUtils.PrintLn('');
    if LCreateOK then
      TVdxUtils.PrintLn(COLOR_GREEN +
        'FTS5 AVAILABLE - CREATE VIRTUAL TABLE succeeded.')
    else
      TVdxUtils.PrintLn(COLOR_RED +
        'FTS5 NOT AVAILABLE - Error: %s', [LErrMsg]);

    // 4. Dump full compile options so we see everything else
    TVdxUtils.PrintLn('');
    TVdxUtils.PrintLn(COLOR_CYAN + '--- PRAGMA compile_options ---');
    LQuery.SQL.Text := 'PRAGMA compile_options';
    LQuery.Open();
    while not LQuery.Eof do
    begin
      TVdxUtils.PrintLn('  %s', [LQuery.Fields[0].AsString]);
      LQuery.Next();
    end;
    LQuery.Close();

    LConn.Close();
  finally
    LQuery.Free();
    LConn.Free();
    LLink.Free();
  end;
end;

// ---------------------------------------------------------------------------
// Test07_MemoryRoundtrip — end-to-end exercise of TVdxMemory. Writes ten
// alternating user/assistant turns to a fresh SQLite DB, asserts basic reads
// (count, meta, FTS5 search, recent-turns ordering), closes the session,
// re-opens the same file, and re-asserts everything to prove persistence
// across sessions. Deletes the DB file on the way in and on the way out so
// consecutive runs start from a known-clean state.
// ---------------------------------------------------------------------------
procedure Test07_MemoryRoundtrip();
const
  CDbFile = 'test_memory.db';
  CTurnRoles: array[0..9] of string = (
    'user', 'assistant', 'user', 'assistant', 'user',
    'assistant', 'user', 'assistant', 'user', 'assistant'
  );
  // Turn 2 carries the ''purple'' marker, turn 5 carries ''Paris''.
  // Remaining turns are neutral filler so the search ranks unambiguously.
  CTurnTexts: array[0..9] of string = (
    'Hello there, nice to meet you.',
    'Hi! Great to meet you too. How can I help?',
    'My favorite color is purple and I think about it often.',
    'That is a lovely preference to have.',
    'What is the capital city of France?',
    'The capital of France is Paris, located in the north.',
    'Interesting, thanks for that bit of trivia.',
    'You are very welcome anytime.',
    'What is 2 plus 2?',
    'The answer is 4.'
  );
var
  LMemory: TVdxMemory;
  LPass: Integer;
  LFail: Integer;
  LPath: string;
  LI: Integer;
  LHits: TArray<TVdxMemoryTurn>;
  LRecent: TArray<TVdxMemoryTurn>;
  LCount: Integer;

  procedure Banner(const AText: string);
  begin
    TVdxUtils.PrintLn();
    TVdxUtils.PrintLn(COLOR_YELLOW +
      '========================================================');
    TVdxUtils.PrintLn(COLOR_YELLOW + '  %s', [AText]);
    TVdxUtils.PrintLn(COLOR_YELLOW +
      '========================================================');
  end;

  procedure Check(const ACond: Boolean; const ALabel: string);
  begin
    if ACond then
    begin
      Inc(LPass);
      TVdxUtils.PrintLn(COLOR_GREEN + '  [PASS] %s', [ALabel]);
    end
    else
    begin
      Inc(LFail);
      TVdxUtils.PrintLn(COLOR_RED + '  [FAIL] %s', [ALabel]);
    end;
  end;

  procedure RunAssertions(const APhase: string);
  var
    LCI: Integer;
  begin
    Banner(Format('Assertions - %s', [APhase]));

    LCount := LMemory.GetTurnCount();
    Check(LCount = 10,
      Format('GetTurnCount() = 10 (got %d)', [LCount]));

    LHits := LMemory.SearchFTS5('purple', 3);
    Check(Length(LHits) > 0,
      Format('SearchFTS5(''purple'') returns >=1 hit (got %d)',
        [Length(LHits)]));
    if Length(LHits) > 0 then
      Check(Pos('purple', LowerCase(LHits[0].Text)) > 0,
        Format('Top ''purple'' hit contains ''purple'' (turn_index=%d)',
          [LHits[0].TurnIndex]));

    LHits := LMemory.SearchFTS5('Paris', 3);
    Check(Length(LHits) > 0,
      Format('SearchFTS5(''Paris'') returns >=1 hit (got %d)',
        [Length(LHits)]));
    if Length(LHits) > 0 then
      Check(Pos('paris', LowerCase(LHits[0].Text)) > 0,
        Format('Top ''Paris'' hit contains ''Paris'' (turn_index=%d)',
          [LHits[0].TurnIndex]));

    LRecent := LMemory.GetRecentTurns(3);
    Check(Length(LRecent) = 3,
      Format('GetRecentTurns(3) returns 3 items (got %d)',
        [Length(LRecent)]));
    if Length(LRecent) = 3 then
    begin
      for LCI := 0 to 2 do
        Check(LRecent[LCI].TurnIndex = 7 + LCI,
          Format('Recent[%d].TurnIndex = %d (got %d)',
            [LCI, 7 + LCI, LRecent[LCI].TurnIndex]));
    end;
  end;

begin
  LPass := 0;
  LFail := 0;
  LPath := TPath.Combine(ExtractFilePath(ParamStr(0)), CDbFile);

  TVdxUtils.PrintLn('MEMORY ROUNDTRIP - TVdxMemory end-to-end test');
  TVdxUtils.PrintLn(COLOR_WHITE + 'DB path: %s', [LPath]);

  // Fresh DB every run — drop any leftover from a prior aborted run.
  if TFile.Exists(LPath) then
    TFile.Delete(LPath);

  LMemory := TVdxMemory.Create();
  try
    // --- Phase A: open a fresh DB, populate, assert ---
    Banner('PHASE A - open fresh DB, write 10 turns');
    if not LMemory.OpenSession(LPath) then
    begin
      TVdxUtils.PrintLn(COLOR_RED + 'OpenSession failed - aborting.');
      Exit;
    end;

    LMemory.SetMeta('model_path', 'fake/model.gguf');
    LMemory.SetMeta('max_context', '4096');

    for LI := 0 to 9 do
      LMemory.AppendTurn(CTurnRoles[LI], CTurnTexts[LI],
        Length(CTurnTexts[LI]) div 4);

    Check(LMemory.GetMeta('model_path') = 'fake/model.gguf',
      'GetMeta(''model_path'') roundtrips');
    Check(LMemory.GetMeta('missing') = '',
      'GetMeta(''missing'') returns empty string');

    RunAssertions('fresh session');

    // --- Phase B: close, reopen, re-assert ---
    Banner('PHASE B - close session');
    LMemory.CloseSession();
    Check(not LMemory.IsOpen(),
      'IsOpen() = False after CloseSession()');

    Banner('PHASE B - reopen same DB file');
    if not LMemory.OpenSession(LPath) then
    begin
      TVdxUtils.PrintLn(COLOR_RED + 'Reopen failed - aborting.');
      Exit;
    end;
    Check(LMemory.IsOpen(),
      'IsOpen() = True after reopen');
    Check(LMemory.GetMeta('model_path') = 'fake/model.gguf',
      'session_meta survives reopen');

    RunAssertions('reopened session');

  finally
    LMemory.CloseSession();
    LMemory.Free();
  end;

  // Cleanup — remove the DB file so the next run starts clean.
  if TFile.Exists(LPath) then
    TFile.Delete(LPath);

  // --- Summary ---
  Banner('RESULTS');
  TVdxUtils.PrintLn(COLOR_GREEN + '  Passed: %d', [LPass]);
  if LFail = 0 then
    TVdxUtils.PrintLn(COLOR_GREEN + '  Failed: %d', [LFail])
  else
    TVdxUtils.PrintLn(COLOR_RED + '  Failed: %d', [LFail]);
end;

// ---------------------------------------------------------------------------
// Test08_VectorSearchRoundtrip — end-to-end exercise of TVdxMemory.SearchVector.
// Loads EmbeddingGemma, opens a fresh SQLite DB, attaches the embedder, writes
// 10 turns (5 color-themed RELATED + 5 math-themed UNRELATED), issues a
// semantic query and asserts that all five top-5 hits come from the related
// cluster. Also verifies cosine-DESC ordering, ATopK=0 empty-return semantics,
// BLOB persistence across close/reopen, and that SearchVector raises after
// DetachEmbeddings. DB file is cleaned up on the way in and on the way out.
// ---------------------------------------------------------------------------
procedure Test08_VectorSearchRoundtrip();
const
  CModelPath = 'C:\Dev\LLM\GGUF\embeddinggemma-300m-qat-Q8_0.gguf';
  CDbFile    = 'test_vector.db';

  // 10 turns, alternating user/assistant. Even indices (0,2,4,6,8) are
  // RELATED (colors, hues, lavender, violet); odd indices (1,3,5,7,9)
  // are UNRELATED (math, physics). Balanced 5/5 so the top-5 test is
  // a clean boundary assertion.
  CTurnRoles: array[0..9] of string = (
    'user', 'assistant', 'user', 'assistant', 'user',
    'assistant', 'user', 'assistant', 'user', 'assistant'
  );
  CTurnTexts: array[0..9] of string = (
    'My favorite color is deep purple.',
    'The derivative of x squared is two x.',
    'She painted the walls a soft lavender shade.',
    'An electron has negative charge of about 1.6e-19 coulombs.',
    'Violet and indigo sit at the short end of the visible spectrum.',
    'The integral of one over x from 1 to e equals one.',
    'Amethyst gemstones have a rich purple hue.',
    'Newton''s second law states force equals mass times acceleration.',
    'The jacaranda tree blooms with clusters of lilac flowers.',
    'The square root of one hundred forty four is twelve.'
  );
  CRelatedIndices: array[0..4] of Integer = (0, 2, 4, 6, 8);
  CSemanticQuery  = 'shades of purple and lavender';
var
  LEmb: TVdxEmbeddings;
  LMemory: TVdxMemory;
  LLoaded: Boolean;
  LPath: string;
  LI: Integer;
  LHits: TArray<TVdxMemoryTurn>;
  LCount: Integer;
  LIsRelated: Boolean;
  LRelatedFound: Integer;
  LPass: Integer;
  LFail: Integer;
  LRaised: Boolean;

  procedure Banner(const AText: string);
  begin
    TVdxUtils.PrintLn();
    TVdxUtils.PrintLn(COLOR_YELLOW +
      '========================================================');
    TVdxUtils.PrintLn(COLOR_YELLOW + '  %s', [AText]);
    TVdxUtils.PrintLn(COLOR_YELLOW +
      '========================================================');
  end;

  procedure Check(const ACond: Boolean; const ALabel: string);
  begin
    if ACond then
    begin
      Inc(LPass);
      TVdxUtils.PrintLn(COLOR_GREEN + '  [PASS] %s', [ALabel]);
    end
    else
    begin
      Inc(LFail);
      TVdxUtils.PrintLn(COLOR_RED + '  [FAIL] %s', [ALabel]);
    end;
  end;

  function IsRelatedIndex(const AIdx: Integer): Boolean;
  var
    LK: Integer;
  begin
    Result := False;
    for LK := 0 to High(CRelatedIndices) do
      if CRelatedIndices[LK] = AIdx then
      begin
        Result := True;
        Exit;
      end;
  end;

begin
  LPass := 0;
  LFail := 0;
  LPath := TPath.Combine(ExtractFilePath(ParamStr(0)), CDbFile);

  TVdxUtils.PrintLn(
    'VECTOR SEARCH ROUNDTRIP - TVdxMemory.SearchVector end-to-end test');
  TVdxUtils.PrintLn(COLOR_WHITE + 'Model:   %s', [CModelPath]);
  TVdxUtils.PrintLn(COLOR_WHITE + 'DB path: %s', [LPath]);

  // Fresh DB every run — drop any leftover from a prior aborted run.
  if TFile.Exists(LPath) then
    TFile.Delete(LPath);

  LEmb := TVdxEmbeddings.Create();
  try
    LEmb.SetStatusCallback(StatusCallback, nil);

    Banner('LOAD - open embedding GGUF and build GPU pipelines');
    LLoaded := LEmb.LoadModel(CModelPath);
    if not LLoaded then
    begin
      TVdxUtils.PrintLn(COLOR_RED + 'LoadModel failed - aborting.');
      Exit;
    end;

    LMemory := TVdxMemory.Create();
    try
      // --- Phase A: open DB, attach, write, search ---
      Banner('PHASE A - open fresh DB, attach embedder, write 10 turns');
      if not LMemory.OpenSession(LPath) then
      begin
        TVdxUtils.PrintLn(COLOR_RED + 'OpenSession failed - aborting.');
        Exit;
      end;

      LMemory.AttachEmbeddings(LEmb);

      for LI := 0 to 9 do
        LMemory.AppendTurn(CTurnRoles[LI], CTurnTexts[LI],
          Length(CTurnTexts[LI]) div 4);

      LCount := LMemory.GetTurnCount();
      Check(LCount = 10,
        Format('GetTurnCount() = 10 (got %d)', [LCount]));

      Banner('PHASE A - semantic query, expect all 5 top-5 hits RELATED');
      LHits := LMemory.SearchVector(CSemanticQuery, 5);
      Check(Length(LHits) = 5,
        Format('SearchVector returns 5 hits (got %d)', [Length(LHits)]));

      // Count how many of the top 5 are from the related cluster, and
      // print each rank with its cosine score for human sanity-check.
      LRelatedFound := 0;
      for LI := 0 to High(LHits) do
      begin
        LIsRelated := IsRelatedIndex(LHits[LI].TurnIndex);
        if LIsRelated then
          Inc(LRelatedFound);
        if LIsRelated then
          TVdxUtils.PrintLn(COLOR_CYAN +
            '  rank %d: turn_index=%d cosine=%.4f  [RELATED]   %s',
            [LI, LHits[LI].TurnIndex, LHits[LI].CosineScore,
             LHits[LI].Text])
        else
          TVdxUtils.PrintLn(COLOR_CYAN +
            '  rank %d: turn_index=%d cosine=%.4f  [UNRELATED] %s',
            [LI, LHits[LI].TurnIndex, LHits[LI].CosineScore,
             LHits[LI].Text]);
      end;
      Check(LRelatedFound = 5,
        Format('All 5 top-5 hits are RELATED (got %d/5)', [LRelatedFound]));

      // Monotonic non-increasing CosineScore across sorted results.
      for LI := 1 to High(LHits) do
        Check(LHits[LI - 1].CosineScore >= LHits[LI].CosineScore,
          Format('Score sorted DESC at rank %d: %.4f >= %.4f',
            [LI, LHits[LI - 1].CosineScore, LHits[LI].CosineScore]));

      Banner('PHASE A - ATopK=0 returns empty');
      LHits := LMemory.SearchVector('anything', 0);
      Check(Length(LHits) = 0,
        Format('ATopK=0 returns empty array (got %d)', [Length(LHits)]));

      // --- Phase B: close, reopen, re-attach, re-search ---
      Banner('PHASE B - close, reopen same DB, re-attach, search again');
      LMemory.CloseSession();
      Check(not LMemory.IsOpen(),
        'IsOpen() = False after CloseSession()');

      if not LMemory.OpenSession(LPath) then
      begin
        TVdxUtils.PrintLn(COLOR_RED + 'Reopen failed - aborting.');
        Exit;
      end;
      Check(LMemory.IsOpen(),
        'IsOpen() = True after reopen');

      LCount := LMemory.GetTurnCount();
      Check(LCount = 10,
        Format('GetTurnCount() = 10 after reopen (got %d)', [LCount]));

      LMemory.AttachEmbeddings(LEmb);

      LHits := LMemory.SearchVector(CSemanticQuery, 5);
      LRelatedFound := 0;
      for LI := 0 to High(LHits) do
        if IsRelatedIndex(LHits[LI].TurnIndex) then
          Inc(LRelatedFound);
      Check(LRelatedFound = 5,
        Format('After reopen, all 5 top-5 hits still RELATED (got %d/5)',
          [LRelatedFound]));

      // --- Phase C: detach, assert SearchVector raises ---
      Banner('PHASE C - detach embedder, SearchVector must raise');
      LMemory.DetachEmbeddings();
      LRaised := False;
      try
        LMemory.SearchVector('anything', 5);
      except
        on E: Exception do
        begin
          LRaised := True;
          TVdxUtils.PrintLn(COLOR_CYAN + '  expected exception: %s',
            [E.Message]);
        end;
      end;
      Check(LRaised,
        'SearchVector raises after DetachEmbeddings');

    finally
      LMemory.CloseSession();
      LMemory.Free();
    end;
  finally
    LEmb.UnloadModel();
    LEmb.Free();
  end;

  // Cleanup — remove the DB file so the next run starts clean.
  if TFile.Exists(LPath) then
    TFile.Delete(LPath);

  // --- Summary ---
  Banner('RESULTS');
  TVdxUtils.PrintLn(COLOR_GREEN + '  Passed: %d', [LPass]);
  if LFail = 0 then
    TVdxUtils.PrintLn(COLOR_GREEN + '  Failed: %d', [LFail])
  else
    TVdxUtils.PrintLn(COLOR_RED + '  Failed: %d', [LFail]);
end;

// ---------------------------------------------------------------------------
// Test09_EmbeddingsRoundtrip — loads EmbeddingGemma, embeds three sentences
// with semantic relationships, and asserts that the related pair scores
// higher cosine similarity than either does against the unrelated sentence.
// Proves: model loads, forward pass runs, pooling + normalize produce
// usable vectors, and the embeddings actually capture meaning.
// ---------------------------------------------------------------------------
procedure Test09_EmbeddingsRoundtrip();
const
  CModelPath = 'C:\Dev\LLM\GGUF\embeddinggemma-300m-qat-Q8_0.gguf';
  CTextPurple = 'I love the color purple.';
  CTextViolet = 'My favorite hue is violet.';
  CTextBicycle = 'How do bicycles work?';
var
  LEmb: TVdxEmbeddings;
  LLoaded: Boolean;
  LVecPurple: TArray<Single>;
  LVecViolet: TArray<Single>;
  LVecBicycle: TArray<Single>;
  LSimRelated: Single;
  LSimUnrelated1: Single;
  LSimUnrelated2: Single;
  LPass: Integer;
  LFail: Integer;

  procedure Banner(const AText: string);
  begin
    TVdxUtils.PrintLn();
    TVdxUtils.PrintLn(COLOR_YELLOW +
      '========================================================');
    TVdxUtils.PrintLn(COLOR_YELLOW + '  %s', [AText]);
    TVdxUtils.PrintLn(COLOR_YELLOW +
      '========================================================');
  end;

  procedure Check(const ACond: Boolean; const ALabel: string);
  begin
    if ACond then
    begin
      Inc(LPass);
      TVdxUtils.PrintLn(COLOR_GREEN + '  [PASS] %s', [ALabel]);
    end
    else
    begin
      Inc(LFail);
      TVdxUtils.PrintLn(COLOR_RED + '  [FAIL] %s', [ALabel]);
    end;
  end;

begin
  LPass := 0;
  LFail := 0;

  TVdxUtils.PrintLn('EMBEDDINGS ROUNDTRIP - TVdxEmbeddings end-to-end test');
  TVdxUtils.PrintLn(COLOR_WHITE + 'Model: %s', [CModelPath]);

  LEmb := TVdxEmbeddings.Create();
  try
    LEmb.SetStatusCallback(StatusCallback, nil);

    Banner('LOAD - open GGUF and build GPU pipelines');
    LLoaded := LEmb.LoadModel(CModelPath);
    if not LLoaded then
    begin
      TVdxUtils.PrintLn(COLOR_RED + 'LoadModel failed — aborting.');
      Exit;
    end;
    TVdxUtils.PrintLn(COLOR_GREEN + 'Loaded. arch=%s, dim=%d, max_seq=%d',
      [LEmb.GetArchitecture(), LEmb.GetEmbeddingDim(),
       LEmb.GetMaxSeqLen()]);
    Check(LEmb.IsLoaded(), 'IsLoaded() = True');
    Check(LEmb.GetEmbeddingDim() > 0, 'GetEmbeddingDim() > 0');

    Banner('EMBED - three sentences, document prefix');
    TVdxUtils.PrintLn('  "%s"', [CTextPurple]);
    LVecPurple := LEmb.Embed(CTextPurple, False);
    TVdxUtils.PrintLn('  "%s"', [CTextViolet]);
    LVecViolet := LEmb.Embed(CTextViolet, False);
    TVdxUtils.PrintLn('  "%s"', [CTextBicycle]);
    LVecBicycle := LEmb.Embed(CTextBicycle, False);

    Check(Length(LVecPurple) = LEmb.GetEmbeddingDim(),
      Format('purple vector length = %d', [LEmb.GetEmbeddingDim()]));
    Check(Length(LVecViolet) = LEmb.GetEmbeddingDim(),
      Format('violet vector length = %d', [LEmb.GetEmbeddingDim()]));
    Check(Length(LVecBicycle) = LEmb.GetEmbeddingDim(),
      Format('bicycle vector length = %d', [LEmb.GetEmbeddingDim()]));

    Banner('SIMILARITY - cosine between pairs');
    LSimRelated := TVdxEmbeddings.CosineSimilarity(LVecPurple, LVecViolet);
    LSimUnrelated1 := TVdxEmbeddings.CosineSimilarity(LVecPurple, LVecBicycle);
    LSimUnrelated2 := TVdxEmbeddings.CosineSimilarity(LVecViolet, LVecBicycle);

    TVdxUtils.PrintLn(COLOR_CYAN + '  purple <-> violet   : %.4f',
      [LSimRelated]);
    TVdxUtils.PrintLn(COLOR_CYAN + '  purple <-> bicycle  : %.4f',
      [LSimUnrelated1]);
    TVdxUtils.PrintLn(COLOR_CYAN + '  violet <-> bicycle  : %.4f',
      [LSimUnrelated2]);

    Check(LSimRelated > LSimUnrelated1,
      'sim(purple,violet) > sim(purple,bicycle)');
    Check(LSimRelated > LSimUnrelated2,
      'sim(purple,violet) > sim(violet,bicycle)');

  finally
    LEmb.UnloadModel();
    LEmb.Free();
  end;

  Banner('RESULTS');
  TVdxUtils.PrintLn(COLOR_GREEN + '  Passed: %d', [LPass]);
  if LFail = 0 then
    TVdxUtils.PrintLn(COLOR_GREEN + '  Failed: %d', [LFail])
  else
    TVdxUtils.PrintLn(COLOR_RED + '  Failed: %d', [LFail]);
end;

// ---------------------------------------------------------------------------
// Test10_RebuildThreshold — exercises the Phase 2.5a rebuild machinery in
// TVdxInference. Loads the model with a small context cap (512) and a small
// rebuild threshold (80 tokens), installs an anonymous-method rebuild
// callback that records its arguments, drives Generate() calls until the
// KV position crosses the threshold, then verifies that the next Generate()
// invokes the callback exactly once with the expected arguments and resets
// the cache to a position well below the pre-trigger value. A follow-up
// Generate() (while still under threshold) must not re-fire the callback.
//
// The callback's returned replacement prompt supplants the user's prompt for
// the triggering turn — this is the documented contract of SetRebuildCallback
// and is how Phase B of the retrieval cycle will hand-craft the full
// (system + retrieved + recent + new user) prefill in later subtasks.
// ---------------------------------------------------------------------------
procedure Test10_RebuildThreshold();
const
  CModelPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.Q8_0.gguf';
  CMaxCtx    = 512;
  CRebuildAt = 80;
  CMaxGen    = 64;
  CReplacementPrompt =
    'Hi. Please respond with the single word "ready".';
  CPrompts: array[0..5] of string = (
    'Describe a red apple in two sentences.',
    'Describe a blue sky in two sentences.',
    'Describe a green tree in two sentences.',
    'Describe a yellow sun in two sentences.',
    'Describe a white cloud in two sentences.',
    'Describe a black cat in two sentences.'
  );
var
  LInference: TVdxInference;
  LConfig: TVdxSamplerConfig;
  LLoaded: Boolean;
  LPass: Integer;
  LFail: Integer;
  LFireCount: Integer;
  LLastAPos: UInt32;
  LLastAMax: UInt32;
  LLastAPrompt: string;
  LI: Integer;
  LPosBefore: UInt32;
  LPosAfter: UInt32;
  LStageBPrompt: string;
  LRebuildHandler: TVdxRebuildCallback;

  procedure Banner(const AText: string);
  begin
    TVdxUtils.PrintLn();
    TVdxUtils.PrintLn(COLOR_YELLOW +
      '========================================================');
    TVdxUtils.PrintLn(COLOR_YELLOW + '  %s', [AText]);
    TVdxUtils.PrintLn(COLOR_YELLOW +
      '========================================================');
  end;

  procedure Check(const ACond: Boolean; const ALabel: string);
  begin
    if ACond then
    begin
      Inc(LPass);
      TVdxUtils.PrintLn(COLOR_GREEN + '  [PASS] %s', [ALabel]);
    end
    else
    begin
      Inc(LFail);
      TVdxUtils.PrintLn(COLOR_RED + '  [FAIL] %s', [ALabel]);
    end;
  end;

begin
  LPass := 0;
  LFail := 0;
  LFireCount := 0;
  LLastAPos := 0;
  LLastAMax := 0;
  LLastAPrompt := '';

  TVdxUtils.PrintLn('REBUILD THRESHOLD - TVdxInference rebuild callback test');
  TVdxUtils.PrintLn(COLOR_WHITE + 'MaxContext=%d RebuildAt=%d MaxGen=%d',
    [CMaxCtx, CRebuildAt, CMaxGen]);

  // Anonymous method closes over the counters above. Delphi captures locals
  // by reference, so the callback sees and mutates the enclosing state
  // directly. This keeps the test self-contained — no globals needed.
  LRebuildHandler :=
    function(const APosition: UInt32;
             const AMaxContext: UInt32;
             const APrompt: string;
             const AUserData: Pointer): string
    begin
      Inc(LFireCount);
      LLastAPos := APosition;
      LLastAMax := AMaxContext;
      LLastAPrompt := APrompt;
      TVdxUtils.PrintLn(COLOR_MAGENTA +
        '[rebuild callback] pos=%d max=%d prompt-len=%d',
        [APosition, AMaxContext, Length(APrompt)]);
      Result := CReplacementPrompt;
    end;

  GTokenWriter := TVdxConsoleTokenWriter.Create();
  try
    GTokenWriter.MaxWidth := 118;

    LInference := TVdxInference.Create();
    try
      LInference.SetStatusCallback(StatusCallback, nil);
      LInference.SetTokenCallback(PrintToken, nil);
      LInference.SetInferenceEventCallback(InferenceEventCallback, nil);
      LInference.SetCancelCallback(CancelCallback, nil);

      Banner('LOAD - small context, small rebuild threshold');
      LLoaded := LInference.LoadModel(CModelPath, CMaxCtx, CRebuildAt);
      PrintErrors(LInference);
      if not LLoaded then
      begin
        TVdxUtils.PrintLn(COLOR_RED + 'LoadModel failed - aborting.');
        Exit;
      end;
      Check(LInference.GetRebuildAt() = CRebuildAt,
        Format('GetRebuildAt() = %d (got %d)',
          [CRebuildAt, LInference.GetRebuildAt()]));

      // Deterministic sampling so turn sizes are reproducible across runs.
      LConfig := TVdxSampler.DefaultConfig();
      LConfig.Temperature := 0.0;
      LConfig.TopK := 1;
      LConfig.TopP := 1.0;
      LConfig.MinP := 0.0;
      LConfig.RepeatPenalty := 1.0;
      LConfig.RepeatWindow := 64;
      LConfig.Seed := 1;
      LInference.SetSamplerConfig(LConfig);

      LInference.SetRebuildCallback(LRebuildHandler, nil);

      // --- Stage A: accumulate turns until position crosses threshold ---
      Banner('STAGE A - accumulate turns, expect NO callback fires yet');
      LI := 0;
      while (LInference.GetKVCachePosition() < CRebuildAt) and
            (LI < Length(CPrompts)) do
      begin
        TVdxUtils.PrintLn(COLOR_CYAN + '[turn %d] pos-before=%d',
          [LI, LInference.GetKVCachePosition()]);
        GTokenWriter.Reset();
        LInference.Generate(CPrompts[LI], CMaxGen);
        PrintErrors(LInference);
        TVdxUtils.PrintLn(COLOR_CYAN + '[turn %d] pos-after =%d',
          [LI, LInference.GetKVCachePosition()]);
        Inc(LI);
      end;

      Check(LFireCount = 0,
        Format('Callback NOT fired while below threshold (fire_count=%d)',
          [LFireCount]));
      Check(LInference.GetKVCachePosition() >= CRebuildAt,
        Format('Position crossed threshold (%d >= %d)',
          [LInference.GetKVCachePosition(), CRebuildAt]));

      // --- Stage B: one more Generate(); callback should fire and reset ---
      Banner('STAGE B - one more turn, callback fires and resets cache');
      LPosBefore := LInference.GetKVCachePosition();
      LStageBPrompt := CPrompts[LI mod Length(CPrompts)];
      TVdxUtils.PrintLn(COLOR_CYAN + '[trigger] pos-before=%d prompt="%s"',
        [LPosBefore, LStageBPrompt]);
      GTokenWriter.Reset();
      LInference.Generate(LStageBPrompt, CMaxGen);
      PrintErrors(LInference);
      LPosAfter := LInference.GetKVCachePosition();
      TVdxUtils.PrintLn(COLOR_CYAN + '[trigger] pos-after =%d', [LPosAfter]);

      Check(LFireCount = 1,
        Format('Callback fired exactly once (fire_count=%d)', [LFireCount]));
      Check(LLastAPos = LPosBefore,
        Format('Callback APosition = pre-reset position (%d vs %d)',
          [LLastAPos, LPosBefore]));
      Check(LLastAMax = CMaxCtx,
        Format('Callback AMaxContext = %d (got %d)',
          [CMaxCtx, LLastAMax]));
      Check(LLastAPrompt = LStageBPrompt,
        'Callback APrompt matches Stage B user prompt');
      // After the rebuild, position reflects only the replacement prompt's
      // prefill + its generated tokens. It MUST be smaller than the
      // pre-trigger position, otherwise the reset didn't happen.
      Check(LPosAfter < LPosBefore,
        Format('Position after rebuild < pre-trigger (%d < %d)',
          [LPosAfter, LPosBefore]));

      // --- Stage C: next turn stays under threshold, callback must not re-fire ---
      Banner('STAGE C - next turn stays under threshold, callback quiet');
      TVdxUtils.PrintLn(COLOR_CYAN + '[stage-c] pos-before=%d',
        [LInference.GetKVCachePosition()]);
      GTokenWriter.Reset();
      LInference.Generate(CPrompts[(LI + 1) mod Length(CPrompts)], CMaxGen);
      PrintErrors(LInference);
      TVdxUtils.PrintLn(COLOR_CYAN + '[stage-c] pos-after =%d',
        [LInference.GetKVCachePosition()]);
      Check(LFireCount = 1,
        Format('Callback still fired only once (fire_count=%d)',
          [LFireCount]));

    finally
      LInference.UnloadModel();
      LInference.Free();
    end;
  finally
    GTokenWriter.Free();
    GTokenWriter := nil;
  end;

  Banner('RESULTS');
  TVdxUtils.PrintLn(COLOR_GREEN + '  Passed: %d', [LPass]);
  if LFail = 0 then
    TVdxUtils.PrintLn(COLOR_GREEN + '  Failed: %d', [LFail])
  else
    TVdxUtils.PrintLn(COLOR_RED + '  Failed: %d', [LFail]);
end;

// ---------------------------------------------------------------------------
// Test11_DedupPinPurge
// Phase 2.5c verification: exact-duplicate suppression via content_hash,
// pinned column, AddFact, PurgeTurn, PurgeAll, PurgeWhere. Loads the
// embedder so AddFact can be verified in both SearchFTS5 and SearchVector.
// ---------------------------------------------------------------------------
procedure Test11_DedupPinPurge();
const
  CModelPath = 'C:\Dev\LLM\GGUF\embeddinggemma-300m-qat-Q8_0.gguf';
  CDbFile    = 'test_dedup.db';

  CTurnTexts: array[0..4] of string = (
    'The capital of France is Paris.',
    'Quantum entanglement links distant particles.',
    'Chocolate cake is my favorite dessert.',
    'The Nile is the longest river in Africa.',
    'Rust is a systems programming language.'
  );
var
  LEmb: TVdxEmbeddings;
  LMemory: TVdxMemory;
  LLoaded: Boolean;
  LPath: string;
  LI: Integer;
  LIds: array[0..4] of Int64;
  LReIds: array[0..4] of Int64;
  LPass: Integer;
  LFail: Integer;
  LTurn: TVdxMemoryTurn;
  LFactId: Int64;
  LHits: TArray<TVdxMemoryTurn>;
  LFactFound: Boolean;

  procedure Banner(const AText: string);
  begin
    TVdxUtils.PrintLn();
    TVdxUtils.PrintLn(COLOR_YELLOW +
      '========================================================');
    TVdxUtils.PrintLn(COLOR_YELLOW + '  %s', [AText]);
    TVdxUtils.PrintLn(COLOR_YELLOW +
      '========================================================');
  end;

  procedure Check(const ACond: Boolean; const ALabel: string);
  begin
    if ACond then
    begin
      Inc(LPass);
      TVdxUtils.PrintLn(COLOR_GREEN + '  [PASS] %s', [ALabel]);
    end
    else
    begin
      Inc(LFail);
      TVdxUtils.PrintLn(COLOR_RED + '  [FAIL] %s', [ALabel]);
    end;
  end;

begin
  LPass := 0;
  LFail := 0;
  LPath := TPath.Combine(ExtractFilePath(ParamStr(0)), CDbFile);

  TVdxUtils.PrintLn('DEDUP / PIN / PURGE - Phase 2.5c test');
  TVdxUtils.PrintLn(COLOR_WHITE + 'DB path: %s', [LPath]);

  // Clean slate.
  if TFile.Exists(LPath) then
    TFile.Delete(LPath);

  LEmb := TVdxEmbeddings.Create();
  LMemory := TVdxMemory.Create();
  try
    // Load embedder for AddFact + SearchVector verification.
    Banner('LOAD EMBEDDER');
    LLoaded := LEmb.LoadModel(CModelPath);
    Check(LLoaded, 'Embedder loaded');
    if not LLoaded then
    begin
      TVdxUtils.PrintLn(COLOR_RED + 'Cannot load embedder - aborting.');
      Exit;
    end;

    if not LMemory.OpenSession(LPath) then
    begin
      TVdxUtils.PrintLn(COLOR_RED + 'OpenSession failed - aborting.');
      Exit;
    end;
    LMemory.AttachEmbeddings(LEmb);

    // --- Phase A: Dedup ---
    Banner('PHASE A - Dedup');

    // Write 5 unique turns.
    for LI := 0 to 4 do
      LIds[LI] := LMemory.AppendTurn('user', CTurnTexts[LI],
        Length(CTurnTexts[LI]) div 4);

    Check(LMemory.GetTurnCount() = 5,
      Format('Count = 5 after 5 unique inserts (got %d)',
        [LMemory.GetTurnCount()]));

    // Re-insert all 5 — should all be deduped.
    for LI := 0 to 4 do
      LReIds[LI] := LMemory.AppendTurn('user', CTurnTexts[LI],
        Length(CTurnTexts[LI]) div 4);

    Check(LMemory.GetTurnCount() = 5,
      Format('Count still 5 after 5 duplicate inserts (got %d)',
        [LMemory.GetTurnCount()]));

    // Returned IDs must match originals.
    for LI := 0 to 4 do
      Check(LReIds[LI] = LIds[LI],
        Format('Dedup ID match [%d]: original=%d, re-insert=%d',
          [LI, LIds[LI], LReIds[LI]]));

    // Dedup with trivial whitespace variation.
    Check(LMemory.AppendTurn('user', '  The capital of   France  is Paris. ',
      10) = LIds[0],
      'Whitespace-normalized dedup matches turn 0');

    Check(LMemory.GetTurnCount() = 5,
      Format('Count still 5 after whitespace-variant insert (got %d)',
        [LMemory.GetTurnCount()]));

    // --- Phase B: PurgeTurn ---
    Banner('PHASE B - PurgeTurn');

    // Purge turn 2.
    LMemory.PurgeTurn(LIds[2]);
    Check(LMemory.GetTurnCount() = 4,
      Format('Count = 4 after purging turn 2 (got %d)',
        [LMemory.GetTurnCount()]));

    // Verify turn 2 is gone.
    LTurn := LMemory.GetTurn(LIds[2]);
    Check(LTurn.TurnId = 0,
      'GetTurn for purged turn 2 returns empty record');

    // Verify turns 0 and 1 still present.
    LTurn := LMemory.GetTurn(LIds[0]);
    Check(LTurn.TurnId = LIds[0],
      Format('Turn 0 still present (id=%d)', [LTurn.TurnId]));
    LTurn := LMemory.GetTurn(LIds[1]);
    Check(LTurn.TurnId = LIds[1],
      Format('Turn 1 still present (id=%d)', [LTurn.TurnId]));

    // --- Phase C: PurgeAll ---
    Banner('PHASE C - PurgeAll');
    LMemory.PurgeAll();
    Check(LMemory.GetTurnCount() = 0,
      Format('Count = 0 after PurgeAll (got %d)',
        [LMemory.GetTurnCount()]));

    // --- Phase D: AddFact ---
    Banner('PHASE D - AddFact');
    LFactId := LMemory.AddFact('The speed of light is 299792458 m/s.');
    Check(LFactId > 0,
      Format('AddFact returned valid id (%d)', [LFactId]));
    Check(LMemory.GetTurnCount() = 1,
      Format('Count = 1 after AddFact (got %d)',
        [LMemory.GetTurnCount()]));

    // Verify fact record.
    LTurn := LMemory.GetTurn(LFactId);
    Check(LTurn.Role = 'fact',
      Format('Fact role = ''fact'' (got ''%s'')', [LTurn.Role]));
    Check(LTurn.Pinned = True,
      'Fact is pinned by default');

    // Verify fact appears in FTS5.
    LHits := LMemory.SearchFTS5('speed light', 5);
    LFactFound := False;
    for LI := 0 to High(LHits) do
      if LHits[LI].TurnId = LFactId then
        LFactFound := True;
    Check(LFactFound,
      'AddFact appears in SearchFTS5 results');

    // Verify fact appears in SearchVector.
    LHits := LMemory.SearchVector('speed of light meters per second', 5);
    LFactFound := False;
    for LI := 0 to High(LHits) do
      if LHits[LI].TurnId = LFactId then
        LFactFound := True;
    Check(LFactFound,
      'AddFact appears in SearchVector results');

    // AddFact dedup — same text should return same id.
    Check(LMemory.AddFact('The speed of light is 299792458 m/s.') = LFactId,
      'AddFact dedup returns same id for identical text');
    Check(LMemory.GetTurnCount() = 1,
      Format('Count still 1 after duplicate AddFact (got %d)',
        [LMemory.GetTurnCount()]));

  finally
    LMemory.DetachEmbeddings();
    LMemory.CloseSession();
    LMemory.Free();
    LEmb.UnloadModel();
    LEmb.Free();
  end;

  // Cleanup.
  if TFile.Exists(LPath) then
    TFile.Delete(LPath);

  Banner('RESULTS');
  TVdxUtils.PrintLn(COLOR_GREEN + '  Passed: %d', [LPass]);
  if LFail = 0 then
    TVdxUtils.PrintLn(COLOR_GREEN + '  Failed: %d', [LFail])
  else
    TVdxUtils.PrintLn(COLOR_RED + '  Failed: %d', [LFail]);
end;

// ---------------------------------------------------------------------------
// Test12_DocumentIngest — verifies document chunking, FTS5 retrieval of
// chunks, and cascade purge via PurgeDocument. No embedder needed.
// ---------------------------------------------------------------------------
procedure Test12_DocumentIngest();
const
  CDbFile = 'test_document.db';
  CChunkWords = 30;
  COverlapWords = 5;

  // A test document with distinct paragraphs on varied topics. Each
  // paragraph is separated by a blank line. Total word count ~200.
  CTestDocument =
    'The Amazon rainforest is the largest tropical rainforest in the world. ' +
    'It covers over five million square kilometers across nine countries in ' +
    'South America. The biodiversity found there is unmatched anywhere else ' +
    'on the planet with millions of species of insects and thousands of ' +
    'species of birds and mammals.' + #10#10 +

    'Quantum computing represents a fundamental shift in computational ' +
    'paradigms. Unlike classical computers that use binary bits, quantum ' +
    'computers leverage qubits which can exist in superposition states. ' +
    'This allows quantum machines to solve certain problems exponentially ' +
    'faster than traditional hardware.' + #10#10 +

    'The art of sourdough bread baking has experienced a remarkable ' +
    'renaissance in recent years. Bakers cultivate wild yeast starters ' +
    'that ferment flour and water over days. The resulting bread has a ' +
    'distinctive tangy flavor and chewy texture that cannot be replicated ' +
    'by commercial yeast products.' + #10#10 +

    'Volcanic eruptions on the ocean floor create hydrothermal vents that ' +
    'support unique ecosystems. Giant tube worms, eyeless shrimp, and ' +
    'chemosynthetic bacteria thrive in these extreme environments. These ' +
    'communities exist entirely without sunlight, deriving energy from ' +
    'chemical reactions involving hydrogen sulfide.' + #10#10 +

    'The history of typography spans centuries from Gutenberg movable ' +
    'type to modern digital fonts. Each typeface carries cultural weight ' +
    'and communicates subtle meaning beyond the words themselves. Designers ' +
    'carefully select fonts to evoke specific emotions and establish visual ' +
    'hierarchy in their compositions.';

var
  LMemory: TVdxMemory;
  LPath: string;
  LDocId: Int64;
  LTurnCount: Integer;
  LHits: TArray<TVdxMemoryTurn>;
  LPass: Integer;
  LFail: Integer;

  procedure Banner(const AText: string);
  begin
    TVdxUtils.PrintLn();
    TVdxUtils.PrintLn(COLOR_YELLOW +
      '=== Test12_DocumentIngest — %s ===', [AText]);
  end;

  procedure Check(const ALabel: string; const AOk: Boolean);
  begin
    if AOk then
    begin
      Inc(LPass);
      TVdxUtils.PrintLn(COLOR_GREEN + '  [PASS] %s', [ALabel]);
    end
    else
    begin
      Inc(LFail);
      TVdxUtils.PrintLn(COLOR_RED + '  [FAIL] %s', [ALabel]);
    end;
  end;

begin
  LPass := 0;
  LFail := 0;
  LPath := TPath.Combine(TPath.GetDirectoryName(ParamStr(0)), CDbFile);

  // Clean up from any prior run.
  if TFile.Exists(LPath) then
    TFile.Delete(LPath);

  Banner('INGEST');

  LMemory := TVdxMemory.Create();
  try
    LMemory.OpenSession(LPath);
    Check('Session opened', LMemory.IsOpen());

    // Ingest the document — no embedder attached, chunks get NULL embeddings.
    LDocId := LMemory.AddDocument(
      'test.txt', 'Test Document', CTestDocument,
      CChunkWords, COverlapWords, False);
    Check('AddDocument returned doc id > 0', LDocId > 0);

    LTurnCount := LMemory.GetTurnCount();
    TVdxUtils.PrintLn('  Chunk count: %d', [LTurnCount]);
    Check('At least 5 chunks created', LTurnCount >= 5);
    Check('No more than 20 chunks created', LTurnCount <= 20);

    Banner('FTS5 SEARCH');

    // 'sourdough' appears only in the bread-baking paragraph.
    LHits := LMemory.SearchFTS5('sourdough', 3);
    Check('SearchFTS5(sourdough) returned results', Length(LHits) > 0);
    if Length(LHits) > 0 then
      Check('Top hit contains sourdough',
        Pos('sourdough', LowerCase(LHits[0].Text)) > 0);

    // 'hydrothermal' appears only in the ocean-floor paragraph.
    LHits := LMemory.SearchFTS5('hydrothermal', 3);
    Check('SearchFTS5(hydrothermal) returned results', Length(LHits) > 0);
    if Length(LHits) > 0 then
      Check('Top hit contains hydrothermal',
        Pos('hydrothermal', LowerCase(LHits[0].Text)) > 0);

    // All chunks should have role 'chunk'.
    LHits := LMemory.GetRecentTurns(LTurnCount);
    if Length(LHits) > 0 then
      Check('First chunk has role = chunk',
        LHits[0].Role = CVdxMemRoleChunk)
    else
      Check('First chunk has role = chunk', False);

    Banner('PURGE');

    LMemory.PurgeDocument(LDocId);
    LTurnCount := LMemory.GetTurnCount();
    Check('PurgeDocument removed all chunks', LTurnCount = 0);

    LMemory.CloseSession();
  finally
    LMemory.Free();
  end;

  // Clean up.
  if TFile.Exists(LPath) then
    TFile.Delete(LPath);

  Banner('RESULTS');
  TVdxUtils.PrintLn(COLOR_GREEN + '  Passed: %d', [LPass]);
  if LFail = 0 then
    TVdxUtils.PrintLn(COLOR_GREEN + '  Failed: %d', [LFail])
  else
    TVdxUtils.PrintLn(COLOR_RED + '  Failed: %d', [LFail]);
end;

// ---------------------------------------------------------------------------
// Test13_SessionChat — end-to-end exercise of TVdxSession. Loads the
// inference model with a small context (512) and low rebuild threshold (80),
// runs a multi-turn conversation via Chat(), verifies turn logging, and
// drives enough turns to trigger the rebuild cycle. No embedder for this
// test — FTS5-only retrieval.
// ---------------------------------------------------------------------------
procedure Test13_SessionChat();
const
  CDbFile = 'test_session.db';
  CModelPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.Q8_0.gguf';
var
  LSession: TVdxSession;
  LConfig: TVdxSamplerConfig;
  LResponse: string;
  LLoaded: Boolean;
  LPath: string;
  LPass: Integer;
  LFail: Integer;

  procedure Banner(const AText: string);
  begin
    TVdxUtils.PrintLn();
    TVdxUtils.PrintLn(COLOR_YELLOW +
      '=== Test13_SessionChat - %s ===', [AText]);
  end;

  procedure Check(const ALabel: string; const AOk: Boolean);
  begin
    if AOk then
    begin
      Inc(LPass);
      TVdxUtils.PrintLn(COLOR_GREEN + '  [PASS] %s', [ALabel]);
    end
    else
    begin
      Inc(LFail);
      TVdxUtils.PrintLn(COLOR_RED + '  [FAIL] %s', [ALabel]);
    end;
  end;

  procedure PrintSessionErrors(const ASession: TVdxSession);
  var
    LErrors: TVdxErrors;
    LItems: TList<TVdxError>;
    LI: Integer;
    LErr: TVdxError;
  begin
    LErrors := ASession.GetErrors();
    if (LErrors = nil) or (LErrors.GetItems().Count = 0) then
      Exit;
    LItems := LErrors.GetItems();
    for LI := 0 to LItems.Count - 1 do
    begin
      LErr := LItems[LI];
      TVdxUtils.PrintLn(COLOR_YELLOW + '  [%s] %s: %s',
        [LErr.GetSeverityString(), LErr.Code, LErr.Message]);
    end;
  end;

begin
  LPass := 0;
  LFail := 0;
  LPath := TPath.Combine(TPath.GetTempPath(), CDbFile);

  // Clean up any leftover DB from a prior run
  if TFile.Exists(LPath) then
    TFile.Delete(LPath);

  GTokenWriter := TVdxConsoleTokenWriter.Create();
  try
    GTokenWriter.MaxWidth := 118;

    Banner('LOAD');

    LSession := TVdxSession.Create();
    try
      LSession.SetStatusCallback(StatusCallback, nil);
      LSession.SetTokenCallback(PrintToken, nil);

      // Load with small context and low rebuild threshold for testing.
      // No embedder (empty string) — FTS5-only retrieval on rebuild.
      LLoaded := LSession.LoadModel(CModelPath, LPath, '', 512, 80);
      PrintSessionErrors(LSession);
      Check('LoadModel succeeded', LLoaded);
      if not LLoaded then
      begin
        Banner('RESULTS');
        TVdxUtils.PrintLn(COLOR_RED + '  ABORT - model failed to load');
        Exit;
      end;

      Check('IsLoaded returns True', LSession.IsLoaded());

      // Configure sampler — deterministic seed for reproducibility
      LConfig := TVdxSampler.DefaultConfig();
      LConfig.Temperature := 1.0;
      LConfig.TopK := 64;
      LConfig.TopP := 0.95;
      LConfig.Seed := 42;
      LSession.SetSamplerConfig(LConfig);

      // Set system prompt
      LSession.SetSystemPrompt(
        'You are a helpful assistant. Keep answers brief.');

      // --- Turn 1 ---
      Banner('CHAT - Turn 1');
      LResponse := LSession.Chat(
        'What is the capital of France?', 64);
      TVdxUtils.PrintLn('');
      Check('Turn 1 returned non-empty', LResponse <> '');

      // --- Turn 2 (tests multi-turn continuation) ---
      Banner('CHAT - Turn 2');
      LResponse := LSession.Chat('What about Germany?', 64);
      TVdxUtils.PrintLn('');
      Check('Turn 2 returned non-empty', LResponse <> '');

      // --- Turn count ---
      Banner('TURN COUNT');
      TVdxUtils.PrintLn('  TurnCount: %d', [LSession.GetTurnCount()]);
      Check('GetTurnCount = 4 (2 user + 2 assistant)',
        LSession.GetTurnCount() = 4);

      // --- Drive more turns to cross rebuild threshold (80 tokens) ---
      Banner('CHAT - Turn 3 (driving toward rebuild)');
      LResponse := LSession.Chat(
        'Tell me about the solar system.', 128);
      TVdxUtils.PrintLn('');
      Check('Turn 3 returned non-empty', LResponse <> '');

      Banner('CHAT - Turn 4 (should trigger rebuild)');
      LResponse := LSession.Chat(
        'What is the largest planet?', 128);
      TVdxUtils.PrintLn('');
      Check('Turn 4 returned non-empty (post-rebuild)', LResponse <> '');

      // --- Unload ---
      Banner('UNLOAD');
      LSession.UnloadModel();
      Check('IsLoaded returns False after unload',
        not LSession.IsLoaded());

    finally
      LSession.Free();
    end;
  finally
    GTokenWriter.Free();
  end;

  // Clean up test DB
  if TFile.Exists(LPath) then
    TFile.Delete(LPath);

  Banner('RESULTS');
  TVdxUtils.PrintLn(COLOR_GREEN + '  Passed: %d', [LPass]);
  if LFail = 0 then
    TVdxUtils.PrintLn(COLOR_GREEN + '  Failed: %d', [LFail])
  else
    TVdxUtils.PrintLn(COLOR_RED + '  Failed: %d', [LFail]);
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
    TVdxUtils.Pause('Press any key to start inference...');

    LIndex := 13;

    case LIndex of
      1: Test01();
      2: Test02();
      3: Test03();
      4: Test04();
      5: Test05();
      6: Test06_FTS5Probe();
      7: Test07_MemoryRoundtrip();
      8: Test08_VectorSearchRoundtrip();
      9: Test09_EmbeddingsRoundtrip();
      10: Test10_RebuildThreshold();
      11: Test11_DedupPinPurge();
      12: Test12_DocumentIngest();
      13: Test13_SessionChat();
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
