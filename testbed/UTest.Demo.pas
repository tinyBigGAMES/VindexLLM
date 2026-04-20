{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.Demo;

interface

procedure Demo_Inference();
procedure Demo_Chat();

implementation

uses
  WinAPI.Windows,
  System.Generics.Collections,
  System.SysUtils,
  VindexLLM.Utils,
  VindexLLM.Inference,
  VindexLLM.Sampler,
  VindexLLM.TokenWriter,
  VindexLLM.Session,
  VindexLLM.Chat,
  VindexLLM.ConsoleChat,
  UTest.Common;

// ===========================================================================
// Demo_Inference — Full inference demo
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
procedure Demo_Inference();
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
      LLoaded := LInference.LoadModel(CModelPath);

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
// Test02 — Interactive multi-turn console chat with memory and RAG
//
// This test demonstrates the high-level TVdxConsoleChat facade, which wraps
// TVdxSession to provide a complete interactive chat experience:
//
//   1. Create TVdxConsoleChat and configure it
//      - Model path:    Gemma 3 4B GGUF for text generation
//      - Embedder path: EmbeddingGemma 300M GGUF for vector search
//      - Memory DB:     SQLite file for persistent conversation history
//      - System prompt: injected at the start of every context window
//      - Sampler:       temperature, top-K, top-P for response variety
//   2. Run the interactive loop (TVdxConsoleChat.Run)
//      - Loads the inference model and embedder, opens the memory DB
//      - Enters a read-eval-print loop: reads user input, generates a
//        response, logs both turns to the memory DB, and repeats
//      - Each turn triggers RAG retrieval: FTS5 keyword search and
//        cosine-similarity vector search over stored turns/facts,
//        merged and injected into the prompt as reference context
//      - Streaming output: tokens appear in the console as generated
//      - Type "quit" or press ESC during generation to stop
//   3. On exit, the model is unloaded and all resources are freed
//      - The memory DB persists on disk for future sessions
//
// Unlike Test01 (single-shot generation), Test02 maintains conversation
// state across turns and uses the embedder for semantic retrieval. The
// memory DB accumulates across runs — restarting the test resumes the
// same conversation history.
//
// To use a different model, change CModelPath. To disable vector search
// (keyword-only RAG), set CEmbedderPath to ''.
// ===========================================================================
procedure Demo_Chat();
const
  // SQLite database file for persistent conversation memory. Created on
  // first run; subsequent runs reopen the same DB and resume history.
  CMemoryDb = 'session.db';

  // Maximum number of tokens to generate per response. Generation stops
  // earlier if EOS, <end_of_turn>, context full, or user cancels (ESC).
  CMaxTokens = 1024;
var
  LChat: TVdxConsoleChat;
  LConfig: TVdxSamplerConfig;
begin
  // --- Create the console chat facade ---
  // TVdxConsoleChat owns a TVdxSession internally, which in turn owns
  // TVdxInference, TVdxMemory, and TVdxEmbeddings. Configuration is
  // set via properties before calling Run().
  LChat := TVdxConsoleChat.Create();
  try
    // --- Configure model paths and session parameters ---
    LChat.ModelPath := CModelPath;
    LChat.EmbedderPath := CEmbedderPath;
    LChat.MemoryDbPath := CMemoryDb;
    LChat.SystemPrompt := 'You are a helpful assistant.';
    LChat.MaxTokens := CMaxTokens;

    // --- Configure the token sampler ---
    // Google-recommended settings for gemma-3-4b-it. Temperature=1.0
    // with TopK=64 / TopP=0.95 gives varied but coherent responses.
    // Seed=0 means non-deterministic (re-seeded from system entropy
    // each Generate() call).
    LConfig := TVdxSampler.DefaultConfig();
    LConfig.Temperature := 1.0;
    LConfig.TopK := 64;
    LConfig.TopP := 0.95;
    LConfig.Seed := 0;
    LChat.SamplerConfig := LConfig;

    // --- Enter the interactive chat loop ---
    // Run() loads the model, opens the memory DB, attaches the embedder,
    // then loops: read user input → retrieve context → generate → log turns.
    // Returns when the user types "quit" or closes the console.
    LChat.Run();
  finally
    LChat.Free();
  end;
end;

end.
