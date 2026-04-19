{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Model.Gemma3;

interface

uses
  VindexLLM.GGUFReader,
  VindexLLM.Model;

type
  // ---------------------------------------------------------------------------
  // TVdxGemma3Model
  //
  // Concrete TVdxModel for Google's Gemma 3 transformer family. Handles both
  // the causal chat variant (`general.architecture = gemma3`) and the
  // bidirectional encoder variant (`general.architecture = gemma-embedding`,
  // a.k.a. EmbeddingGemma) — they share the same layer recipe, so a single
  // class serves TVdxInference AND TVdxEmbeddings once those consumers
  // migrate to the TVdxModel factory in later phases.
  //
  // Distinguishing Gemma 3 features (filled in across Phases 13b–13f):
  //   • Sandwich norm — pre- AND post-norm around both attention and FFN.
  //   • QK-norm — additional RMSNorm on Q and K after projection, before
  //     RoPE.
  //   • GELU-tanh activation in the FFN (not SiLU).
  //   • Per-layer RoPE theta: sliding-window layers use 10K, global-attention
  //     layers (indices 5, 11, 17, 23, 29 — the `ALayer mod 6 = 5` pattern)
  //     use 1M.
  //   • Gemma 3 chat template with <start_of_turn> / <end_of_turn> markers.
  //   • EmbeddingGemma task prefixes for query vs document embedding.
  //
  // Phase 13a ships the scaffolding only: the class exists, declares which
  // architectures it handles, and self-registers. Nothing overrides the
  // forward-pass or weight-load hooks yet — so any attempt to drive this
  // class through LoadModel would raise from TVdxModel's default
  // forward-pass bodies. Wiring through TVdxInference / TVdxEmbeddings
  // happens in Phase 14+, by which time the hooks are populated.
  // ---------------------------------------------------------------------------
  TVdxGemma3Model = class(TVdxModel)
  public
    // Both architectures this class claims. The registry looks these up
    // case-insensitively.
    class function SupportedArchitectures(): TArray<string>; override;

    // Reads Gemma 3 config from the GGUF metadata. Populates the
    // transformer config fields (num_layers, hidden_dim, etc.), derives
    // head_dim from the Q weight tensor shape, and clamps max_seq_len
    // against the model's own context_length ceiling. Adopts the reader
    // via inherited on success, so the factory won't free it.
    function LoadModelConfig(const AReader: TVdxGGUFReader;
      const AMaxContext: Integer): Boolean; override;

    // Resolves per-layer streaming attention/FFN weight pointers, uploads
    // the 6-per-layer Gemma 3 norm tensors + global output_norm to VRAM,
    // and records the embedding-table mmap pointer + sqrt(hidden_dim)
    // scale. Embedding-table GPU mirror (batch prefill) lands with the
    // forward-pass sub-phase. Any failure populates FErrors and returns
    // False; the base destructor's FreeWeights teardown handles partial
    // cleanup automatically.
    function LoadWeights(): Boolean; override;

    // Single-token decode forward pass for one layer. Caller wraps
    // the layer loop in FCompute.BeginBatch / EndBatch and writes the
    // input token's scaled embedding into FResidualGpu before calling
    // this for layer 0. Gemma 3 layer recipe: sandwich-normed
    // attention branch + sandwich-normed FFN branch, two residual
    // accumulations per layer via the shared vec_add pipeline.
    // Per-layer RoPE theta follows the mod-6 pattern (10K sliding,
    // 1M on layers 5, 11, 17, 23, 29).
    procedure RunLayerForward(const ALayer: Integer;
      const APosition: Integer); override;

    // Batched prefill / encoder forward pass for one layer. Caller
    // populates FResidualMat via EmbedTokensBatch, wraps the layer
    // loop in FCompute.BeginBatch / EndBatch. ABidirectional=False
    // for causal generation (Gemma 3 decoder); True for encoder-style
    // attention (EmbeddingGemma). AStartPos is the absolute KV-cache
    // slot where this batch begins writing.
    procedure RunLayerForwardBatch(const ALayer: Integer;
      const ANumTokens, AStartPos: UInt32;
      const ABidirectional: Boolean); override;

    // Per-layer RoPE theta. Gemma 3 alternates sliding-window and
    // global-attention layers on a mod-6 pattern: layers where
    // ALayer mod 6 = 5 (indices 5, 11, 17, 23, 29) are the
    // global-attention layers and use theta = 1,000,000; all
    // others are sliding-window and use the standard 10,000.
    function GetRoPETheta(const ALayer: Integer): Single; override;

    // Gemma 3 chat template: wraps APrompt in <start_of_turn>user / model
    // markers with BOS at the front. The assistant-open marker at the
    // tail signals the model to begin generating a response.
    function FormatPrompt(const APrompt: string): string; override;

    // EmbeddingGemma task prefixes. Query text gets "task: search
    // result | query: " for retrieval-style search; document text gets
    // "title: none | text: " since we don't track a separate title.
    // Matches the google/embeddinggemma-300m recommendation for
    // asymmetric retrieval. AIsQuery=False is the document side.
    function FormatEmbedding(const AText: string;
      const AIsQuery: Boolean): string; override;

    // Stop tokens. Gemma 3 generation terminates on <end_of_turn>.
    // The tokenizer-derived EOS and the GGUF-metadata eot_token are
    // resolved to IDs by the consumer (TVdxInference), which calls
    // this to discover the string form.
    function GetStopTokenStrings(): TArray<string>; override;

    // Both Gemma 3 architectures this class handles (gemma3 and
    // gemma-embedding) expose a usable bidirectional encoder path —
    // so consumers that call TVdxEmbeddings.LoadModel against a
    // Gemma 3 GGUF get the embedding encoder, not a rejection.
    function SupportsEmbedding(): Boolean; override;
  end;

implementation

uses
  System.Math,
  VindexLLM.Vulkan,
  VindexLLM.Utils,
  VindexLLM.Model.Registry;

{ TVdxGemma3Model }

class function TVdxGemma3Model.SupportedArchitectures(): TArray<string>;
begin
  Result := ['gemma3', 'gemma-embedding'];
end;

function TVdxGemma3Model.LoadModelConfig(const AReader: TVdxGGUFReader;
  const AMaxContext: Integer): Boolean;
var
  LArch:      string;
  LQInfo:     TVdxGGUFTensorInfo;
  LEmbedInfo: TVdxGGUFTensorInfo;
  LModelMax:  UInt32;
begin
  Result := False;

  // Base adopts the reader and records AMaxContext. If inherited fails
  // we bail before touching metadata — nothing to undo yet.
  if not inherited LoadModelConfig(AReader, AMaxContext) then Exit;

  // FArchitecture was set by the factory (TVdxModel.LoadModel) before
  // this call. Gemma 3 metadata keys are namespaced by the arch string,
  // so the same path serves both 'gemma3' and 'gemma-embedding'.
  LArch := FArchitecture;
  Status('Architecture: %s', [LArch]);

  // Validate required metadata keys upfront. GetMetadataUInt32 silently
  // falls back to 0 on miss, which would produce mysterious downstream
  // failures; presence checks give clear per-key error messages.
  if not AReader.HasMetadata(LArch + '.block_count') then
  begin
    FErrors.Add(esFatal, 'CONF',
      'Missing required GGUF metadata key: "%s.block_count"', [LArch]);
    Exit;
  end;
  if not AReader.HasMetadata(LArch + '.embedding_length') then
  begin
    FErrors.Add(esFatal, 'CONF',
      'Missing required GGUF metadata key: "%s.embedding_length"', [LArch]);
    Exit;
  end;
  if not AReader.HasMetadata(LArch + '.feed_forward_length') then
  begin
    FErrors.Add(esFatal, 'CONF',
      'Missing required GGUF metadata key: "%s.feed_forward_length"',
      [LArch]);
    Exit;
  end;
  if not AReader.HasMetadata(LArch + '.attention.head_count') then
  begin
    FErrors.Add(esFatal, 'CONF',
      'Missing required GGUF metadata key: "%s.attention.head_count"',
      [LArch]);
    Exit;
  end;
  if not AReader.HasMetadata(LArch + '.attention.head_count_kv') then
  begin
    FErrors.Add(esFatal, 'CONF',
      'Missing required GGUF metadata key: "%s.attention.head_count_kv"',
      [LArch]);
    Exit;
  end;

  // Read validated config values.
  FNumLayers  := AReader.GetMetadataUInt32(LArch + '.block_count');
  FHiddenDim  := AReader.GetMetadataUInt32(LArch + '.embedding_length');
  FFFNWidth   := AReader.GetMetadataUInt32(LArch + '.feed_forward_length');
  FNumQHeads  := AReader.GetMetadataUInt32(LArch + '.attention.head_count');
  FNumKVHeads := AReader.GetMetadataUInt32(LArch + '.attention.head_count_kv');

  if FNumQHeads = 0 then
  begin
    FErrors.Add(esFatal, 'CONF',
      '%s.attention.head_count must be > 0', [LArch]);
    Exit;
  end;

  // head_dim derives from the Q weight tensor's second dimension —
  // some GGUF producers omit the metadata key, but the tensor shape is
  // authoritative. Same tensor also tells us the block-weight
  // quantization type for later shader dispatch decisions.
  if not AReader.HasTensor('blk.0.attn_q.weight') then
  begin
    FErrors.Add(esFatal, 'CONF',
      'Missing required tensor: "blk.0.attn_q.weight"');
    Exit;
  end;
  if not AReader.GetTensorInfo('blk.0.attn_q.weight', LQInfo) then
  begin
    FErrors.Add(esFatal, 'CONF',
      'Failed to read tensor info for "blk.0.attn_q.weight"');
    Exit;
  end;
  if Length(LQInfo.Dimensions) < 2 then
  begin
    FErrors.Add(esFatal, 'CONF',
      '"blk.0.attn_q.weight" has %d dimensions, expected >= 2',
      [Length(LQInfo.Dimensions)]);
    Exit;
  end;
  FHeadDim    := UInt32(LQInfo.Dimensions[1]) div FNumQHeads;
  FWeightType := LQInfo.TensorType;

  // Context clamp. Model's own ceiling bounds the user request;
  // fallback 8192 matches established behavior for GGUFs that omit
  // the key entirely. AMaxContext <= 0 means "no user cap" and
  // selects the model max directly.
  if AReader.HasMetadata(LArch + '.context_length') then
    LModelMax := AReader.GetMetadataUInt32(LArch + '.context_length')
  else
    LModelMax := 8192;
  if AMaxContext <= 0 then
    FMaxSeqLen := LModelMax
  else if UInt32(AMaxContext) < LModelMax then
    FMaxSeqLen := UInt32(AMaxContext)
  else
    FMaxSeqLen := LModelMax;
  Status('Context length: %d (model max: %d)', [FMaxSeqLen, LModelMax]);

  // Embedding tensor type. Distinct from the block-weight type because
  // the embedding table is often quantized differently from the
  // projection weights.
  if not AReader.HasTensor('token_embd.weight') then
  begin
    FErrors.Add(esFatal, 'CONF',
      'Missing required tensor: "token_embd.weight"');
    Exit;
  end;
  if not AReader.GetTensorInfo('token_embd.weight', LEmbedInfo) then
  begin
    FErrors.Add(esFatal, 'CONF',
      'Failed to read tensor info for "token_embd.weight"');
    Exit;
  end;
  FEmbedType := LEmbedInfo.TensorType;

  Status('Weight type: %s  Embed type: %s',
    [VdxGGMLTypeName(FWeightType), VdxGGMLTypeName(FEmbedType)]);
  Status('Config: layers=%d hidden=%d ffn=%d heads=%d/%d head_dim=%d',
    [FNumLayers, FHiddenDim, FFFNWidth, FNumQHeads, FNumKVHeads, FHeadDim]);

  Result := True;
end;

function TVdxGemma3Model.LoadWeights(): Boolean;
var
  LLayer:           Integer;
  LInfo:            TVdxGGUFTensorInfo;
  LOutputNormPtr:   PByte;
  LOutputNormBytes: UInt64;
  LEmbedTableBytes: UInt64;
begin
  Result := False;

  Status('Uploading weights...');

  // --- Per-layer streaming refs + permanent norm buffers ---
  // Attn/FFN refs are mmap pointers (zero VRAM cost at this stage);
  // norms are small permanent GPU buffers (6 tensors * FHiddenDim or
  // FHeadDim per layer).
  SetLength(FAttnWeights, FNumLayers);
  SetLength(FFFNWeights,  FNumLayers);
  SetLength(FNormWeights, FNumLayers);

  for LLayer := 0 to Integer(FNumLayers) - 1 do
  begin
    if not FAttn.ResolveAttnWeights(FReader, LLayer, FAttnWeights[LLayer]) then
      Exit;
    if not FFFN.ResolveFFNWeights(FReader, LLayer, FFFNWeights[LLayer]) then
      Exit;
    if not FNorm.UploadNormWeights(FReader, LLayer, FNormWeights[LLayer]) then
      Exit;
  end;
  Status('  Per-layer weights resolved (%d layers)', [FNumLayers]);

  // --- Global output norm (permanent GPU) ---
  // Small, hot, applied to the final residual before unembedding.
  // Uploaded manually since it's a one-off tensor — no batch helper
  // makes sense for a single global weight.
  if not FReader.HasTensor('output_norm.weight') then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Missing required tensor: "output_norm.weight"');
    Exit;
  end;
  if not FReader.GetTensorInfo('output_norm.weight', LInfo) then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Failed to read tensor info for "output_norm.weight"');
    Exit;
  end;
  LOutputNormPtr := FReader.GetTensorDataPtr('output_norm.weight');
  if LOutputNormPtr = nil then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Failed to resolve mmap pointer for "output_norm.weight"');
    Exit;
  end;
  LOutputNormBytes := UInt64(FHiddenDim) * SizeOf(Single);
  // DEVICE_LOCAL target — use the staged-upload helper (UploadToBuffer
  // alone only works on HOST_VISIBLE memory).
  FOutputNormGpu := UploadTensorToDevice(LOutputNormPtr, LOutputNormBytes);
  if FErrors.HasFatal() then Exit;

  // --- Embedding table (mmap-only at this sub-phase) ---
  // Decode-time single-token unembedding reads directly from mmap;
  // batch-prefill GPU mirror lands with the forward-pass sub-phase
  // once the embed_lookup shader pipelines are wired.
  FEmbedPtr := FReader.GetTensorDataPtr('token_embd.weight');
  if FEmbedPtr = nil then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Failed to resolve mmap pointer for "token_embd.weight"');
    Exit;
  end;

  // Validate embedding quant format — must be one of the four the
  // embed-lookup shader pipelines can handle. FEmbedType was captured
  // from the tensor info during LoadModelConfig.
  if (FEmbedType <> gtF16) and (FEmbedType <> gtF32)
     and (FEmbedType <> gtQ8_0) and (FEmbedType <> gtQ4_0) then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Unsupported embedding type: %s (need F16, F32, Q8_0, or Q4_0)',
      [VdxGGMLTypeName(FEmbedType)]);
    Exit;
  end;

  // Gemma 3 scales input embeddings by sqrt(hidden_dim) before the
  // first residual — applied at embed-lookup time, cached here so the
  // decode path doesn't recompute on every token.
  FEmbedScale := Sqrt(Single(FHiddenDim));

  // --- Upload embedding table to GPU for batched embed-lookup ---
  // Mirror of FEmbedPtr, used by EmbedTokensBatch. Large (vocab ×
  // HiddenDim × bytes-per-elem depending on quant) but necessary —
  // batch prefill reads thousands of rows concurrently and can't
  // amortize a CPU → GPU staging per row.
  if not FReader.GetTensorInfo('token_embd.weight', LInfo) then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Failed to re-read tensor info for "token_embd.weight"');
    Exit;
  end;
  if Length(LInfo.Dimensions) < 2 then
  begin
    FErrors.Add(esFatal, 'LOAD',
      '"token_embd.weight" has %d dimensions, expected >= 2',
      [Length(LInfo.Dimensions)]);
    Exit;
  end;
  Status('  Uploading embedding table to GPU...');
  LEmbedTableBytes := VdxGGMLTensorBytes(FEmbedType,
    LInfo.Dimensions[0], LInfo.Dimensions[1]);
  // Staged upload — 1+ GB at F16, can't map directly.
  FEmbedGpu := UploadTensorToDevice(FEmbedPtr, LEmbedTableBytes);
  if FErrors.HasFatal() then Exit;

  Status('Weights loaded. Embed scale: %.4f', [FEmbedScale]);
  Result := True;
end;

procedure TVdxGemma3Model.RunLayerForward(const ALayer: Integer;
  const APosition: Integer);
var
  LTheta:      Single;
  LVecAddPush: TVdxVecAddPush;
  LGroupsX:    UInt32;
begin
  // Gemma 3 layer recipe — sandwich norm on both branches, two
  // residual accumulations per layer. Caller must already be inside
  // FCompute.BeginBatch / EndBatch.

  LVecAddPush.Count := FHiddenDim;
  LGroupsX := (FHiddenDim + 255) div 256;

  // Per-layer RoPE theta: global-attention layers (indices 5, 11,
  // 17, 23, 29 — the ALayer mod 6 = 5 pattern) use 1,000,000;
  // sliding-window layers use the standard 10,000.
  LTheta := GetRoPETheta(ALayer);

  // === Attention branch: residual += PostAttnNorm(Attn(PreAttnNorm(residual))) ===

  // Fused copy + pre-attn norm: FResidualGpu → (norm) → FWorkBufA
  FNorm.ApplyCopy(FResidualGpu,
    FNormWeights[ALayer].AttnNormGpu, FWorkBufA, FHiddenDim);
  FCompute.BatchBarrier();

  // Full attention with QK-norm and streaming Q/K/V/O weights.
  FAttn.Forward(FWorkBufA, FAttnWeights[ALayer],
    FNormWeights[ALayer].QNormGpu, FNormWeights[ALayer].KNormGpu,
    ALayer, APosition, LTheta, FAttnOutBuf);

  // Post-attn norm in-place on FAttnOutBuf.
  FNorm.Apply(FAttnOutBuf, FNormWeights[ALayer].PostAttnNormGpu,
    FHiddenDim);
  FCompute.BatchBarrier();

  // residual += attn_out (pre-bound descriptor set).
  FCompute.DispatchComputeWithPush(
    FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
    FVecAddAttnDescSet, @LVecAddPush, SizeOf(LVecAddPush),
    LGroupsX);
  FCompute.BatchBarrier();

  // === FFN branch: residual += PostFFNNorm(FFN(PreFFNNorm(residual))) ===

  // Fused copy + pre-FFN norm: FResidualGpu → (norm) → FWorkBufA
  FNorm.ApplyCopy(FResidualGpu,
    FNormWeights[ALayer].FFNNormGpu, FWorkBufA, FHiddenDim);
  FCompute.BatchBarrier();

  // Streaming FFN — gate/up/gelu_mul/down handled internally.
  FFFN.Forward(FWorkBufA, FFFNWeights[ALayer], FFFNOutBuf);

  // Post-FFN norm in-place on FFFNOutBuf.
  FNorm.Apply(FFFNOutBuf, FNormWeights[ALayer].PostFFNNormGpu,
    FHiddenDim);
  FCompute.BatchBarrier();

  // residual += ffn_out.
  FCompute.DispatchComputeWithPush(
    FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
    FVecAddFFNDescSet, @LVecAddPush, SizeOf(LVecAddPush),
    LGroupsX);
  FCompute.BatchBarrier();
end;

function TVdxGemma3Model.GetRoPETheta(const ALayer: Integer): Single;
begin
  if ALayer mod 6 = 5 then
    Result := 1000000.0
  else
    Result := 10000.0;
end;

procedure TVdxGemma3Model.RunLayerForwardBatch(const ALayer: Integer;
  const ANumTokens, AStartPos: UInt32;
  const ABidirectional: Boolean);
var
  LTheta:      Single;
  LVecAddPush: TVdxVecAddPush;
  LGroupsX:    UInt32;
begin
  // Same layer recipe as RunLayerForward, but operates on
  // [NumTokens × HiddenDim] matrices. Caller must already be inside
  // FCompute.BeginBatch / EndBatch with FResidualMat populated.

  LVecAddPush.Count := ANumTokens * FHiddenDim;
  LGroupsX := (LVecAddPush.Count + 255) div 256;

  LTheta := GetRoPETheta(ALayer);

  // === Attention branch ===

  // Fused copy + pre-attn norm (batched).
  FNorm.ApplyCopyBatch(FResidualMat,
    FNormWeights[ALayer].AttnNormGpu, FWorkMat,
    FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();

  // Full batched attention with QK-norm and streaming weights.
  // ABidirectional=True skips the causal mask for encoder-style use.
  FAttn.ForwardBatch(FWorkMat, FAttnWeights[ALayer],
    FNormWeights[ALayer].QNormGpu, FNormWeights[ALayer].KNormGpu,
    ALayer, ANumTokens, AStartPos, LTheta,
    FQMat, FKMat, FVMat, FAttnOutMat, ABidirectional);

  // Post-attn norm in-place on the batched attention output.
  FNorm.ApplyBatch(FAttnOutMat, FNormWeights[ALayer].PostAttnNormGpu,
    FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();

  // residual_mat += attn_out_mat (pre-bound descriptor set for batch).
  FCompute.DispatchComputeWithPush(
    FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
    FVecAddBatchAttnDescSet, @LVecAddPush, SizeOf(LVecAddPush),
    LGroupsX);
  FCompute.BatchBarrier();

  // === FFN branch ===

  // Fused copy + pre-FFN norm (batched).
  FNorm.ApplyCopyBatch(FResidualMat,
    FNormWeights[ALayer].FFNNormGpu, FWorkMat,
    FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();

  // Streaming batched FFN — gate/up/gelu_mul/down handled internally.
  FFFN.ForwardBatch(FWorkMat, FFFNWeights[ALayer], ANumTokens,
    FGateMat, FUpMatBuf, FFFNOutMat);

  // Post-FFN norm in-place on the batched FFN output.
  FNorm.ApplyBatch(FFFNOutMat, FNormWeights[ALayer].PostFFNNormGpu,
    FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();

  // residual_mat += ffn_out_mat.
  FCompute.DispatchComputeWithPush(
    FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
    FVecAddBatchFFNDescSet, @LVecAddPush, SizeOf(LVecAddPush),
    LGroupsX);
  FCompute.BatchBarrier();
end;

function TVdxGemma3Model.FormatPrompt(const APrompt: string): string;
begin
  // Gemma 3 chat template. BOS is tokenizer-managed (EncodeWithBos on
  // the consumer side); we only emit the user/model turn markers and
  // leave the assistant turn open so generation continues into it.
  Result := '<start_of_turn>user' + #10 + APrompt + '<end_of_turn>' + #10 +
            '<start_of_turn>model' + #10;
end;

function TVdxGemma3Model.FormatEmbedding(const AText: string;
  const AIsQuery: Boolean): string;
begin
  // EmbeddingGemma asymmetric prefixes per google/embeddinggemma-300m.
  // "title: none" on the document side is the documented placeholder
  // when no explicit title is tracked.
  if AIsQuery then
    Result := 'task: search result | query: ' + AText
  else
    Result := 'title: none | text: ' + AText;
end;

function TVdxGemma3Model.GetStopTokenStrings(): TArray<string>;
begin
  Result := ['<end_of_turn>'];
end;

function TVdxGemma3Model.SupportsEmbedding(): Boolean;
begin
  Result := True;
end;

initialization
  // Self-register with the model registry. Consumers that pull this unit
  // into their uses clause automatically make Gemma 3 / EmbeddingGemma
  // available to TVdxModel.LoadModel.
  TVdxModelRegistry.RegisterClass(TVdxGemma3Model);

end.
