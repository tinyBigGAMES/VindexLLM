<div align="center">

![VindexLLM](media/logo.png)

<br>

[![Discord](https://img.shields.io/discord/1457450179254026250?style=for-the-badge&logo=discord&label=Discord)](https://discord.gg/Wb6z8Wam7p) [![Follow on Bluesky](https://img.shields.io/badge/Bluesky-tinyBigGAMES-blue?style=for-the-badge&logo=bluesky)](https://bsky.app/profile/tinybiggames.com) 

</div>

## What is VindexLLM?

**VindexLLM** is a GPU-accelerated LLM inference engine written entirely in Delphi, using Vulkan compute shaders for all heavy computation. It loads standard GGUF model files, runs the full transformer forward pass on the GPU, and produces text — no Python, no CUDA toolkit, no external runtimes. The only dependency is `vulkan-1.dll`, which already ships with every modern GPU driver.

Feed it a prompt, get tokens back. Everything between — embedding lookup, 34 layers of attention and FFN, normalization, sampling — happens on the GPU via GLSL 450 compute shaders compiled to SPIR-V and embedded as Windows resources into the binary.

```delphi
var
  LInference: TVdxInference;
begin
  LInference := TVdxInference.Create();
  try
    // Stream tokens to console as they're generated
    LInference.SetTokenCallback(
      procedure(const AToken: string; const AUserData: Pointer)
      begin
        Write(AToken);
      end, nil);

    // Load model — initializes Vulkan, uploads weights to GPU
    if LInference.LoadModel('path\to\gemma-3-4b.Q8_0.gguf') then
    try
      // Generate up to 256 tokens
      LInference.Generate('Explain how a CPU works', 256);
    finally
      LInference.UnloadModel();
    end;
  finally
    LInference.Free();
  end;
end;
```

Load a model, generate text, done. For sampling configuration, event callbacks, cancel support, and performance stats, see the full example in `testbed\UVdxTestbed.pas`.

## Why VindexLLM?

Most LLM inference stacks depend on CUDA (NVIDIA-only, ~3GB toolkit install), Python environments, or large runtime libraries. VindexLLM takes a different approach:
- **Zero install** — Vulkan ships with every NVIDIA, AMD, and Intel GPU driver. No separate toolkit download, no PATH configuration, no DLL hell. The app calls `LoadLibrary('vulkan-1.dll')` and talks directly to the GPU.
- **No vendor lock-in** — Vulkan runs on any modern GPU. Not tied to NVIDIA hardware.
- **Self-contained binary** — All 35 compute shaders are compiled to SPIR-V at build time and embedded into the executable as Windows resources. No loose shader files, no runtime compilation.
- **No Python, no runtime** — Pure compiled Delphi. Starts instantly, no interpreter warmup, no dependency resolution.
- **Memory-mapped model loading** — GGUF files are mapped directly via `CreateFileMapping` / `MapViewOfFile`. Weights are accessed at their file offsets and uploaded to VRAM through staging buffers. No intermediate copies, no parsing into custom formats.
- **Everything on GPU** — The residual stream never leaves VRAM between layers. The only PCIe transfers are the initial token embedding (~10KB) and the final logits download for sampling. Everything else — attention, FFN, norms, activations, residual additions — runs as GPU compute dispatches.

## How It Works

VindexLLM implements the standard dense transformer forward pass. For each token, the engine runs 34 layers of attention and FFN computation on the GPU, then samples the next token from the output logits.

```
                         ┌──────────────┐
                         │  GGUF File   │
                         │  (mmap'd)    │
                         └──────┬───────┘
                                │ weights uploaded to VRAM at startup
                                ▼
  ┌────────────┐     ┌──────────────────────────────────────────────┐
  │  "prompt"  │────►│  Tokenize (BPE)  ──►  Embed (GPU lookup)     │
  └────────────┘     └──────────────────┬───────────────────────────┘
                                        │
                                        ▼  × 34 layers
                     ┌──────────────────────────────────────────────┐
                     │  PreAttnNorm ──► Attention (GQA + RoPE +     │
                     │    QK-norm + TQ3 KV cache) ──► PostAttnNorm  │
                     │  ──► residual += attn_out                    │
                     │                                              │
                     │  PreFFNNorm ──► gate(x), up(x) ──►           │
                     │    GELU(gate) * up ──► down(hidden)          │
                     │  ──► PostFFNNorm ──► residual += ffn_out     │
                     └──────────────────┬───────────────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │  Final RMSNorm ──► Unembed (tied weights)    │
                     │  ──► Sample (temp, top-K/P, min-P, repeat)   │
                     └──────────────────────────────────────────────┘
```
Prefill processes all prompt tokens in parallel using batched matmul shaders. Autoregressive generation runs one token at a time using matvec shaders. Both paths share the same KV cache, which uses TQ3 compression to reduce VRAM usage by 10.7× compared to F32.

## Current Status

Dense inference is working end-to-end. The engine loads a Gemma 3 4B GGUF, tokenizes a prompt with chat template formatting, runs the full forward pass on the GPU, and generates coherent text with configurable sampling.

**What works today:**

- Full Gemma 3 4B forward pass (34 layers, all on GPU, no CPU fallbacks)
- Batched prefill (all prompt tokens processed in parallel via matmul shaders)
- Autoregressive generation with streaming token callback
- F16, Q8_0, and Q4_0 weight format support (detected automatically from GGUF metadata)
- TurboQuant TQ3 KV cache compression with fused attention scoring (no separate dequant step)
- Pure Delphi BPE tokenizer loaded directly from GGUF vocabulary (no external SentencePiece library)
- Token sampling: temperature, top-K, top-P, min-P, repetition penalty, deterministic seeding (xoshiro256** PRNG)
- Gemma 3 chat template formatting
- Architecture validation (graceful error for unsupported models)
- Configurable context length with model-max clamping
- VRAM usage reporting (weights, cache, buffers breakdown)
- Context overflow detection (`srContextFull` stop reason)
- Inference event callbacks (load/unload/prefill/generate start/end)
- Cancel callback (poll per-layer, ESC to abort in testbed)
- Page-file-backed virtual buffers (`TVdxVirtualBuffer<T>`) for CPU workspace — allocates address space without committing physical RAM until touched
- 35 GLSL 450 compute shaders compiled to SPIR-V and embedded as Windows resources

## TurboQuant (TQ3)

TurboQuant is VindexLLM's custom 3-bit quantization format, designed specifically for KV cache compression. It uses a Walsh-Hadamard Transform (WHT) to decorrelate values within each 32-element block before quantizing to 3 bits using Lloyd-Max optimal centroids fitted to a standard normal distribution.

The result is 10.7× compression versus F32 (32 floats × 4 bytes = 128 bytes → 16 bytes per block) with quality that significantly outperforms naive 3-bit rounding because the WHT spreads information across all elements before quantization.

**TQ3 pipeline (per block of 32 values):**

1. Apply fixed sign-flip pattern (improves WHT energy distribution)
2. 5-stage butterfly WHT (in-place, no temporary buffers)
3. Normalize by 1/√32
4. Find absolute max, compute FP16 scale factor (gamma)
5. Quantize each value to nearest Lloyd-Max centroid (8 levels, 3 bits)
6. Pack: 2 low bits into `qs` words, 1 high bit into `qr` word, gamma into FP16

All four TQ3 phases run as GPU compute shaders: general quantize/dequantize (`tq3_quantize.comp`, `tq3_dequantize.comp`), KV-cache-specific quantize/dequantize (`tq3_kv_quantize.comp`, `tq3_kv_dequantize.comp`), fused batch KV store + quantize (`kv_cache_store_batch_tq3.comp`), and fused attention scores directly on TQ3-compressed keys (`attn_scores_mh_tq3.comp`) — eliminating the separate K dequantization step entirely.

CPU reference implementations exist in `VindexLLM.TurboQuant.pas` for validation.

## Supported Models

VindexLLM currently implements the Gemma 3 4B architecture. Two GGUF files have been vetted and confirmed to produce correct output. All vetted models are available for download from the [tinyBigGAMES Hugging Face account](https://huggingface.co/tinybiggames).

| Model | Format | Size | Download |
|-------|--------|------|----------|
| gemma-3-4b-it-null-space-abliterated | Q8_0 | ~4.1 GB | [Download](https://huggingface.co/tinybiggames/gemma-3-4b-it-null-space-abliterated-GGUF/resolve/main/gemma-3-4b-it-null-space-abliterated.Q8_0.gguf) |
| gemma-3-4b-it-null-space-abliterated | F16 | ~8.0 GB | [Download](https://huggingface.co/tinybiggames/gemma-3-4b-it-null-space-abliterated-GGUF/resolve/main/gemma-3-4b-it-null-space-abliterated.f16.gguf) |

Other Gemma 3 4B GGUF files may work but have not been tested. Non-Gemma architectures are not supported — the engine will report a clear error message if you attempt to load an unsupported model.

## Performance

Measured on an NVIDIA RTX 3060 12GB with Gemma 3 4B Q8_0:

| Metric | Value |
|--------|-------|
| Prefill throughput | 35.0 tok/s |
| Generation throughput | 24.4 tok/s |
| Time to first token | 457 ms |

All computation runs on the GPU. The only PCIe transfers per token are the embedding lookup input and the logits download for sampling.

## Getting Started

1. Download a vetted GGUF model from the links above
2. Clone the repository:

```bash
git clone https://github.com/tinyBigGAMES/VindexLLM.git
```

3. Open `src\VindexLLM - Liberating LLM inference.groupproj` in Delphi 12 or higher
4. Build the `VdxTestbed` project
5. Edit the model path in `testbed\UVdxTestbed.pas` (the `LoadModel()` call) to point to your downloaded GGUF
6. Run the testbed — it will load the model, print status messages during weight upload, then generate text with streaming output

The testbed demonstrates the full API: model loading, sampler configuration, token streaming callback, inference event callbacks, cancel callback (press ESC to abort), and stats reporting (prefill/generation throughput, VRAM usage).

## System Requirements

| | Requirement |
|---|---|
| **Host OS** | Windows 10/11 x64 |
| **GPU** | Any Vulkan 1.0+ capable GPU (NVIDIA, AMD, Intel) |
| **VRAM** | 6 GB minimum (Q8_0), 12 GB recommended (F16) |
| **RAM** | 16 GB minimum (GGUF is memory-mapped) |
| **Building from source** | Delphi 12.x or higher |

## Building from Source

1. Clone the repository
2. Open `src\VindexLLM - Liberating LLM inference.groupproj` in Delphi 12 or higher
3. Build the project group (Win64 target)

The shader SPIR-V binaries and compiled resource file (`VindexLLM.Shaders.res`) are checked into the repository, so you do not need the GLSL compiler to build. If you modify any `.comp` shader files, run `shaders\compile.cmd` to recompile them — this requires `glslangValidator.exe` from the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) or the [glslang releases](https://github.com/KhronosGroup/glslang/releases).

## Architecture

### Gemma 3 4B Model Specs

| Parameter | Value |
|-----------|-------|
| Layers | 34 |
| Hidden dimension | 2560 |
| FFN width | 10,240 |
| Q heads / KV heads | 8 / 4 (grouped-query attention) |
| Head dimension | 256 |
| QK-norm | Yes (per-head RMSNorm on Q and K) |
| Normalization | Sandwich RMSNorm (4 per layer + 2 QK-norms) |
| Activation | GELU with tanh approximation |
| Position encoding | RoPE (theta 10K sliding / 1M full-context layers) |
| Full-context layers | 5, 11, 17, 23, 29 |
| Embeddings | Tied (token_embd reused as output projection) |
| Vocabulary | 262,144 tokens |

### Source Units

| Unit | Purpose |
|------|---------|
| `VindexLLM.Inference.pas` | Orchestrator — model loading, forward pass, generation loop, stats |
| `VindexLLM.Attention.pas` | GQA attention, QKV projections, QK-norm, RoPE, KV cache, TQ3 cache, batched prefill attention |
| `VindexLLM.Compute.pas` | Vulkan device manager — instance, buffers, shader dispatch, batch mode |
| `VindexLLM.Vulkan.pas` | Vulkan API type definitions and function pointer bindings |
| `VindexLLM.LayerNorm.pas` | RMSNorm (single + batch, plain + fused copy variants) |
| `VindexLLM.FFNWeights.pas` | FFN weight management — mmap'd layer views, GPU upload via staging |
| `VindexLLM.TurboQuant.pas` | TQ3 quantization — GPU pipelines + CPU reference implementations |
| `VindexLLM.Tokenizer.pas` | Pure Delphi BPE tokenizer loaded from GGUF vocabulary |
| `VindexLLM.ChatTemplate.pas` | Chat template detection and prompt formatting |
| `VindexLLM.Sampler.pas` | Token sampling — temperature, top-K/P, min-P, repetition penalty, xoshiro256** |
| `VindexLLM.GGUFReader.pas` | GGUF parser — metadata, tensor info, memory-mapped file access |
| `VindexLLM.Shaders.pas` | Shader resource loader (SPIR-V binaries from embedded Windows resources) |
| `VindexLLM.VirtualBuffer.pas` | Page-file-backed generic buffer (`TVdxVirtualBuffer<T>`) |
| `VindexLLM.TokenWriter.pas` | Word-wrapping writer for streaming token output (with console subclass) |
| `VindexLLM.Config.pas` | Configuration management |
| `VindexLLM.Utils.pas` | Base classes (`TVdxBaseObject`, `TVdxStatusObject`, `TVdxErrorsObject`), utilities |
| `VindexLLM.TOML.pas` | TOML parser |
| `VindexLLM.Resources.pas` | Resource management |

### Compute Shaders (35 shaders)

| Category | Shaders |
|----------|---------|
| **Matrix-vector** (single token) | `matvec_f16`, `matvec_q8_0`, `matvec_q4_0` |
| **Matrix-matrix** (batched prefill) | `matmul_f16`, `matmul_q8_0`, `matmul_q4_0` |
| **RMSNorm** | `rmsnorm`, `rmsnorm_copy`, `rmsnorm_batch`, `rmsnorm_copy_batch` |
| **QK-norm + RoPE** | `qk_norm`, `rope`, `rope_batch` |
| **Attention** (single token) | `attn_scores_mh`, `softmax_mh`, `attn_value_mh` |
| **Attention** (batched prefill) | `attn_scores_prefill`, `softmax_prefill`, `attn_value_prefill` |
| **KV cache** | `kv_cache_store`, `kv_cache_store_batch` |
| **Embedding lookup** | `embed_lookup_f16`, `embed_lookup_q8`, `embed_lookup_q4_0`, `embed_lookup_batch_f16`, `embed_lookup_batch_q8`, `embed_lookup_batch_q4_0` |
| **TurboQuant TQ3** | `tq3_quantize`, `tq3_dequantize`, `tq3_kv_quantize`, `tq3_kv_dequantize`, `kv_cache_store_batch_tq3`, `attn_scores_mh_tq3` |
| **Activation + residual** | `gelu_mul`, `vec_add` |

All shaders are written in GLSL 450 with no Vulkan extensions required. They are compiled to SPIR-V via `glslangValidator` and embedded into the binary as Windows resources at build time.

> [!IMPORTANT]
> This repository is under active development. The engine currently supports the Gemma 3 4B architecture only. Other model architectures will require implementing their specific layer structures. Follow the repo or join the [Discord](https://discord.gg/Wb6z8Wam7p) to track progress.

## Contributing

VindexLLM is an open project. Whether you are fixing a bug, improving documentation, adding support for a new model architecture, or proposing a feature, contributions are welcome.

- **Report bugs**: Open an issue with a minimal reproduction. The smaller the example, the faster the fix.
- **Suggest features**: Describe the use case first. Features that emerge from real problems get traction fastest.
- **Submit pull requests**: Bug fixes, documentation improvements, new shader optimizations, and well-scoped features are all welcome. Keep changes focused.

Join the [Discord](https://discord.gg/Wb6z8Wam7p) to discuss development, ask questions, and share what you are building.

## Support the Project

VindexLLM is built in the open. If it saves you time or sparks something useful:

- ⭐ **Star the repo**: it costs nothing and helps others find the project
- 🗣️ **Spread the word**: write a post, mention it in a community you are part of
- 💬 **[Join us on Discord](https://discord.gg/Wb6z8Wam7p)**: share what you are building and help shape what comes next
- 💖 **[Become a sponsor](https://github.com/sponsors/tinyBigGAMES)**: sponsorship directly funds development and documentation
- 🦋 **[Follow on Bluesky](https://bsky.app/profile/tinybiggames.com)**: stay in the loop on releases and development

## License

VindexLLM is licensed under the **Apache License 2.0**. See [LICENSE](https://github.com/tinyBigGAMES/VindexLLM/tree/main?tab=License-1-ov-file#readme) for details.

Apache 2.0 is a permissive open source license that lets you use, modify, and distribute VindexLLM freely in both open source and commercial projects. You are not required to release your own source code. The license includes an explicit patent grant. Attribution is required; keep the copyright notice and license file in place.

## Links

- [Discord](https://discord.gg/Wb6z8Wam7p)
- [Bluesky](https://bsky.app/profile/tinybiggames.com)
- [Facebook Group](https://www.facebook.com/groups/vindexllm)
- [tinyBigGAMES](https://tinybiggames.com)

<div align="center">

**VindexLLM™** - Liberating LLM inference

Copyright &copy; 2026-present tinyBigGAMES™ LLC<br/>All Rights Reserved.

</div>