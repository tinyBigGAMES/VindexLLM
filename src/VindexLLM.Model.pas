{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Model;

interface

uses
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.GGUFReader,
  VindexLLM.Compute,
  VindexLLM.LayerNorm,
  VindexLLM.Attention,
  VindexLLM.FFNWeights,
  VindexLLM.Tokenizer;

type
  // Push constants for vec_add. Count is total elements in the flat
  // buffer — HiddenDim for single-token, NumTokens * HiddenDim for
  // batch. Same shader, same descriptor layout, different dispatch grid.
  TVdxVecAddPush = record
    Count: UInt32;
  end;

  // Push constants for the batched embedding lookup shaders. DimParam
  // is HiddenDim/2 for the F16 path (workgroup processes 2 elements
  // at a time via vec2 loads), HiddenDim for Q8_0 and Q4_0 paths.
  // EmbedScale is sqrt(hidden_dim) for Gemma 3 — applied by the shader
  // as it writes each output row.
  TVdxEmbedBatchPush = record
    DimParam:   UInt32;
    EmbedScale: Single;
    NumTokens:  UInt32;
  end;

  // Forward-declared so TVdxModelClass can be an alias for `class of
  // TVdxModel` above the full class body. The registry uses this metaclass
  // to store class references without instantiating the model itself.
  TVdxModel = class;
  TVdxModelClass = class of TVdxModel;

  // ---------------------------------------------------------------------------
  // TVdxModel
  //
  // Abstract base for every model family supported by VindexLLM. Owns the
  // transformer-family-specific forward pass and the arch→template
  // conventions. Concrete descendants (TVdxGemma3Model in Phase 13,
  // TVdxLlama3Model later, etc.) override the virtuals below.
  //
  // Consumers never instantiate descendants directly — they go through
  // TVdxModel.LoadModel, which inspects `general.architecture` from
  // the GGUF, looks up the concrete class in TVdxModelRegistry, and
  // returns a fully-loaded instance. Returns nil when the file can't be
  // opened, the architecture isn't registered, or loading fails.
  //
  // Phase 12 ships the abstract shape only. The shared state moves here
  // when Gemma3 lands in Phase 13 — doing it in two steps keeps each
  // commit small enough to review in isolation and each phase's diff
  // structurally meaningful.
  // ---------------------------------------------------------------------------
  TVdxModel = class(TVdxBaseObject)
  protected
    // The GGUF reader is owned by the model — it holds the mmap that
    // weight tensors stream from, so its lifetime must match the model's.
    FReader: TVdxGGUFReader;

    // Values read from the GGUF header. Populated by LoadModelConfig in
    // concrete descendants.
    FArchitecture: string;
    FGGUFPath:     string;
    FMaxContext:   Integer;

    // Transformer config — populated by descendant's LoadModelConfig.
    // Universal across Llama-family architectures; descendants read
    // them from GGUF metadata using the arch-prefixed key convention.
    // FVocabSize is populated later in InitSubsystems once the
    // tokenizer loads, since it isn't stored under a fixed GGUF key.
    FNumLayers:   UInt32;
    FHiddenDim:   UInt32;
    FFFNWidth:    UInt32;
    FNumQHeads:   UInt32;
    FNumKVHeads:  UInt32;
    FHeadDim:     UInt32;
    FVocabSize:   Integer;
    FMaxSeqLen:   UInt32;
    FWeightType:  TVdxGGMLType;
    FEmbedType:   TVdxGGMLType;

    // Owned GPU / inference subsystems. Created and error-wired in the
    // constructor; initialized (Vulkan up, pipelines compiled, tokenizer
    // parsed) in InitSubsystems once config dimensions are known.
    FCompute:   TVdxCompute;
    FNorm:      TVdxLayerNorm;
    FAttn:      TVdxAttention;
    FFFN:       TVdxFFN;
    FTokenizer: TVdxTokenizer;

    // Weight storage. Per-layer attn/FFN are streaming (mmap refs only);
    // per-layer norms + output norm are permanent GPU buffers (small,
    // hot, hit every layer). Embedding table stays mmap-resident — GPU
    // mirror for batch prefill arrives with the forward-pass sub-phase.
    FAttnWeights:   array of TVdxAttnLayerWeights;
    FFFNWeights:    array of TVdxFFNLayerWeights;
    FNormWeights:   array of TVdxNormLayerWeights;
    FOutputNormGpu: TVdxGpuBuffer;
    FEmbedPtr:      PByte;
    FEmbedScale:    Single;

    // Decode-path scratch buffers. Sized to HiddenDim F32s; allocated
    // in InitSubsystems once config dimensions land. FResidualGpu is
    // HOST_VISIBLE so the CPU-side single-token embed lookup can write
    // one row directly. Others are DEVICE_LOCAL — GPU-only.
    FResidualGpu: TVdxGpuBuffer;
    FWorkBufA:    TVdxGpuBuffer;
    FAttnOutBuf:  TVdxGpuBuffer;
    FFFNOutBuf:   TVdxGpuBuffer;

    // Shared vec_add pipeline — one shader, count-parameterized, serves
    // both decode and batch callers. Two pre-allocated descriptor sets
    // bind (residual, attn_out) and (residual, ffn_out) to avoid
    // per-dispatch descriptor updates in the hot loop.
    FVecAddShader:      VkShaderModule;
    FVecAddDescLayout:  VkDescriptorSetLayout;
    FVecAddBundle:      TVdxComputePipelineBundle;
    FVecAddDescPool:    VkDescriptorPool;
    FVecAddAttnDescSet: VkDescriptorSet;
    FVecAddFFNDescSet:  VkDescriptorSet;

    // Batch-path matrix scratch buffers. Sized to MaxSeqLen × HiddenDim
    // (or FFNWidth / KV dims as appropriate). All DEVICE_LOCAL; callers
    // write into FResidualMat via EmbedTokensBatch (GPU path). Reused
    // across layers so each forward-pass phase amortizes allocation.
    FResidualMat:  TVdxGpuBuffer;  // [NumTokens x HiddenDim]
    FWorkMat:      TVdxGpuBuffer;  // [NumTokens x HiddenDim] normed scratch
    FQMat:         TVdxGpuBuffer;  // [NumTokens x NumQHeads * HeadDim]
    FKMat:         TVdxGpuBuffer;  // [NumTokens x NumKVHeads * HeadDim]
    FVMat:         TVdxGpuBuffer;  // [NumTokens x NumKVHeads * HeadDim]
    FAttnOutMat:   TVdxGpuBuffer;  // [NumTokens x HiddenDim]
    FGateMat:      TVdxGpuBuffer;  // [NumTokens x FFNWidth]
    FUpMatBuf:     TVdxGpuBuffer;  // [NumTokens x FFNWidth]
    FFFNOutMat:    TVdxGpuBuffer;  // [NumTokens x HiddenDim]
    FTokenIdsGpu:  TVdxGpuBuffer;  // [MaxSeqLen] UInt32 — embed_batch input

    // Permanent GPU mirror of the embedding table — consumed by the
    // batched embed-lookup shaders. Separate from FEmbedPtr (which
    // stays mmap-resident for the single-token CPU path).
    FEmbedGpu: TVdxGpuBuffer;

    // Batched embed-lookup pipelines — one per supported embedding
    // quant format. Dispatched by EmbedTokensBatch based on FEmbedType.
    // Descriptor set is re-bound per call (output matrix may differ
    // between callers — Inference writes to FResidualMat, Embeddings
    // may point at its own batch buffer).
    FEmbedBatchF16Shader: VkShaderModule;
    FEmbedBatchQ8Shader:  VkShaderModule;
    FEmbedBatchQ4Shader:  VkShaderModule;
    FEmbedBatchF16Bundle: TVdxComputePipelineBundle;
    FEmbedBatchQ8Bundle:  TVdxComputePipelineBundle;
    FEmbedBatchQ4Bundle:  TVdxComputePipelineBundle;
    FEmbedBatchDescLayout: VkDescriptorSetLayout;
    FEmbedBatchDescPool:   VkDescriptorPool;
    FEmbedBatchDescSet:    VkDescriptorSet;

    // Batch vec_add descriptor sets. Share the decode pipeline +
    // layout, allocated from the same pool (which was sized for 4
    // total sets at decode-build time).
    FVecAddBatchAttnDescSet: VkDescriptorSet;
    FVecAddBatchFFNDescSet:  VkDescriptorSet;

    // Decode-resource lifecycle helpers. Called from InitSubsystems
    // and the destructor respectively. Both are nil-safe on partial
    // initialization — every destroy call guards against VK_NULL_HANDLE
    // / zeroed records.
    function  BuildDecodeResources(): Boolean;
    procedure FreeDecodeResources();

    // Batch-resource lifecycle helpers. Called after BuildDecodeResources
    // (re-uses vec_add pipeline + pool). Allocates large scratch
    // matrices, uploads the embedding table to VRAM, and builds the
    // three embed-batch pipelines.
    function  BuildBatchResources(): Boolean;
    procedure FreeBatchResources();

    // Staged upload helper — creates a HOST_VISIBLE staging buffer,
    // memcpys AData into it, creates a DEVICE_LOCAL target buffer,
    // GPU-copies staging → target, and destroys staging. Used by
    // LoadWeights for tensors that live permanently in VRAM
    // (output norm, embedding-table mirror). Caller is responsible
    // for destroying the returned buffer. On any failure the
    // returned record is zeroed and FErrors is populated.
    function UploadTensorToDevice(const AData: PByte;
      const ASize: UInt64): TVdxGpuBuffer;

  public
    constructor Create(); override;
    destructor  Destroy(); override;

    // -----------------------------------------------------------------------
    // Architecture identity — abstract.
    //
    // Returns every `general.architecture` value this class handles. For
    // example TVdxGemma3Model returns ['gemma3', 'gemma-embedding']
    // because EmbeddingGemma is structurally identical to Gemma 3.
    // -----------------------------------------------------------------------
    class function SupportedArchitectures(): TArray<string>; virtual;

    // -----------------------------------------------------------------------
    // Lifecycle hooks called by LoadFromGGUF in sequence. Each returns
    // False on failure; errors are written into the instance's error
    // buffer. Default bodies do nothing successfully — descendants
    // override what they actually implement.
    //
    // Order: LoadModelConfig → InitSubsystems → LoadWeights. Any returning
    // False aborts the factory with a nil result.
    // -----------------------------------------------------------------------
    function LoadModelConfig(const AReader: TVdxGGUFReader;
      const AMaxContext: Integer): Boolean; virtual;
    function InitSubsystems(): Boolean; virtual;
    function LoadWeights(): Boolean; virtual;

    // Release GPU/VRAM state owned by the descendant. Called from the
    // destructor — descendants must be safe against multiple Free calls
    // and partial initialization.
    procedure FreeWeights(); virtual;

    // -----------------------------------------------------------------------
    // Forward pass — the two paths both consumers drive.
    //
    // RunLayerForwardBatch: prefill + embedding. ABidirectional=False for
    // causal generation, True for encoder-style embedding.
    //
    // RunLayerForward: per-token decode. Inference-only path.
    //
    // Abstract in spirit — descendants MUST override. Base implementations
    // raise to make misuse loud during development.
    // -----------------------------------------------------------------------
    procedure RunLayerForwardBatch(const ALayer: Integer;
      const ANumTokens, AStartPos: UInt32;
      const ABidirectional: Boolean); virtual;
    procedure RunLayerForward(const ALayer: Integer;
      const APosition: Integer); virtual;

    // -----------------------------------------------------------------------
    // Per-layer RoPE theta base. Default 10000.0 — Gemma 3 overrides with
    // its mod-6 interleave of 10K (sliding) / 1M (global).
    // -----------------------------------------------------------------------
    function GetRoPETheta(const ALayer: Integer): Single; virtual;

    // -----------------------------------------------------------------------
    // Template surface — chat formatting, embedding task prefixes, stop
    // tokens. Defaults are safe pass-throughs so non-chat models don't
    // crash if a consumer asks; descendants override with the real thing.
    // -----------------------------------------------------------------------
    function FormatPrompt(const APrompt: string): string; virtual;
    function FormatEmbedding(const AText: string;
      const AIsQuery: Boolean): string; virtual;
    function GetStopTokenStrings(): TArray<string>; virtual;

    // Does this model support being used as an embedding encoder?
    // Defaults to False — only models that expose bidirectional + pooled
    // outputs override to True. TVdxEmbeddings.LoadModel will reject a
    // model that returns False here with a clear "not supported" error.
    function SupportsEmbedding(): Boolean; virtual;

    // Batched GPU embedding lookup. Uploads ATokenIds to FTokenIdsGpu,
    // re-binds the embed descriptor set with AOutputBuf, and dispatches
    // the embed-batch pipeline matching FEmbedType. Writes one HiddenDim
    // row per token, scaled by FEmbedScale = sqrt(HiddenDim). Caller
    // must be inside an active FCompute batch.
    procedure EmbedTokensBatch(const ATokenIds: TArray<Integer>;
      const ANumTokens: Integer;
      const AOutputBuf: TVdxGpuBuffer); virtual;

    // -----------------------------------------------------------------------
    // Factory. Opens AGGUFPath, reads `general.architecture`, resolves
    // the concrete class through TVdxModelRegistry, instantiates it, and
    // runs LoadModelConfig → InitSubsystems → LoadWeights.
    //
    // Returns a fully-loaded instance on success; nil on any failure
    // (missing file, unknown architecture, config/subsystem/weight-load
    // error). The registry itself is set up by the initialization
    // sections of the concrete model units — the caller just has to
    // pull them into its uses clause.
    // -----------------------------------------------------------------------
    class function LoadModel(const AGGUFPath: string;
      const AMaxContext: Integer;
      const AStatusCallback: TVdxStatusCallback = nil;
      const AStatusUserData: Pointer = nil): TVdxModel;

    // Accessors so consumers don't reach into protected fields.
    property Architecture: string  read FArchitecture;
    property GGUFPath:     string  read FGGUFPath;
    property MaxContext:   Integer read FMaxContext;
    property NumLayers:  UInt32 read FNumLayers;
    property HiddenDim:  UInt32 read FHiddenDim;
    property FFNWidth:   UInt32 read FFFNWidth;
    property NumQHeads:  UInt32 read FNumQHeads;
    property NumKVHeads: UInt32 read FNumKVHeads;
    property HeadDim:    UInt32 read FHeadDim;
    property VocabSize:  Integer read FVocabSize;
    property MaxSeqLen:  UInt32 read FMaxSeqLen;
    property WeightType: TVdxGGMLType read FWeightType;
    property EmbedType:  TVdxGGMLType read FEmbedType;
    property Compute:   TVdxCompute   read FCompute;
    property Norm:      TVdxLayerNorm read FNorm;
    property Attn:      TVdxAttention read FAttn;
    property FFN:       TVdxFFN       read FFFN;
    property Tokenizer: TVdxTokenizer read FTokenizer;
    property Reader: TVdxGGUFReader read FReader;
  end;

implementation

uses
  System.SysUtils,
  VindexLLM.Shaders,
  VindexLLM.Model.Registry;

{ TVdxModel }

constructor TVdxModel.Create();
begin
  inherited;
  FReader     := nil;
  FArchitecture := '';
  FGGUFPath   := '';
  FMaxContext := 0;
  FNumLayers   := 0;
  FHiddenDim   := 0;
  FFFNWidth    := 0;
  FNumQHeads   := 0;
  FNumKVHeads  := 0;
  FHeadDim     := 0;
  FVocabSize   := 0;
  FMaxSeqLen   := 0;
  FWeightType  := TVdxGGMLType(0);
  FEmbedType   := TVdxGGMLType(0);

  // Subsystems are cheap to construct (no Vulkan calls yet) — real
  // work happens in Init*/Initialize later. SetErrors composition
  // ensures diagnostics from any subsystem land in the model's
  // shared FErrors buffer, so callers get one unified error view.
  FCompute := TVdxCompute.Create();
  FCompute.SetErrors(FErrors);
  FNorm := TVdxLayerNorm.Create();
  FNorm.SetErrors(FErrors);
  FAttn := TVdxAttention.Create();
  FAttn.SetErrors(FErrors);
  FFFN := TVdxFFN.Create();
  FFFN.SetErrors(FErrors);
  FTokenizer := TVdxTokenizer.Create();
  FTokenizer.SetErrors(FErrors);

  FAttnWeights := nil;
  FFFNWeights  := nil;
  FNormWeights := nil;
  FOutputNormGpu := Default(TVdxGpuBuffer);
  FEmbedPtr    := nil;
  FEmbedScale  := 0.0;

  FResidualGpu := Default(TVdxGpuBuffer);
  FWorkBufA    := Default(TVdxGpuBuffer);
  FAttnOutBuf  := Default(TVdxGpuBuffer);
  FFFNOutBuf   := Default(TVdxGpuBuffer);

  FVecAddShader      := VK_NULL_HANDLE;
  FVecAddDescLayout  := VK_NULL_HANDLE;
  FVecAddBundle      := Default(TVdxComputePipelineBundle);
  FVecAddDescPool    := VK_NULL_HANDLE;
  FVecAddAttnDescSet := VK_NULL_HANDLE;
  FVecAddFFNDescSet  := VK_NULL_HANDLE;

  FResidualMat := Default(TVdxGpuBuffer);
  FWorkMat     := Default(TVdxGpuBuffer);
  FQMat        := Default(TVdxGpuBuffer);
  FKMat        := Default(TVdxGpuBuffer);
  FVMat        := Default(TVdxGpuBuffer);
  FAttnOutMat  := Default(TVdxGpuBuffer);
  FGateMat     := Default(TVdxGpuBuffer);
  FUpMatBuf    := Default(TVdxGpuBuffer);
  FFFNOutMat   := Default(TVdxGpuBuffer);
  FTokenIdsGpu := Default(TVdxGpuBuffer);
  FEmbedGpu    := Default(TVdxGpuBuffer);

  FEmbedBatchF16Shader  := VK_NULL_HANDLE;
  FEmbedBatchQ8Shader   := VK_NULL_HANDLE;
  FEmbedBatchQ4Shader   := VK_NULL_HANDLE;
  FEmbedBatchF16Bundle  := Default(TVdxComputePipelineBundle);
  FEmbedBatchQ8Bundle   := Default(TVdxComputePipelineBundle);
  FEmbedBatchQ4Bundle   := Default(TVdxComputePipelineBundle);
  FEmbedBatchDescLayout := VK_NULL_HANDLE;
  FEmbedBatchDescPool   := VK_NULL_HANDLE;
  FEmbedBatchDescSet    := VK_NULL_HANDLE;

  FVecAddBatchAttnDescSet := VK_NULL_HANDLE;
  FVecAddBatchFFNDescSet  := VK_NULL_HANDLE;
end;

destructor TVdxModel.Destroy();
begin
  // Descendants release VRAM in FreeWeights. Call unconditionally —
  // implementations must be safe against partial or absent
  // initialization.
  FreeWeights();

  // Batch machinery — larger scratch matrices + embed_batch pipelines
  // + embedding table GPU mirror. Tear down before decode resources so
  // vec_add descriptor sets die before their parent pool.
  FreeBatchResources();

  // Decode machinery — pipelines, descriptor pool, scratch buffers.
  // Must die while FCompute is still alive since these are handles
  // against its Vulkan device. Nil-safe on partial init.
  FreeDecodeResources();

  // Subsystems — torn down in reverse construction order. Each class
  // guards Cleanup() against partial init, so this is safe even if
  // Init* never completed.
  FTokenizer.Free();
  FFFN.Free();
  FAttn.Free();
  FNorm.Free();
  FCompute.Free();

  FReader.Free();

  inherited;
end;

// ---------------------------------------------------------------------------
// Base SupportedArchitectures returns an empty set. An instance of raw
// TVdxModel never handles any architecture — only concrete descendants
// do. The registry skips empty-name entries from RegisterClass, so the
// base registering itself here (which it doesn't) would be a no-op anyway.
// ---------------------------------------------------------------------------
class function TVdxModel.SupportedArchitectures(): TArray<string>;
begin
  Result := nil;
end;

// ---------------------------------------------------------------------------
// Default lifecycle bodies: accept and do nothing. This lets a descendant
// that only needs to override LoadWeights leave the other hooks alone
// without breaking the factory's Boolean contract.
// ---------------------------------------------------------------------------
function TVdxModel.LoadModelConfig(const AReader: TVdxGGUFReader;
  const AMaxContext: Integer): Boolean;
begin
  // Descendants override to read dims, num_layers, etc. from AReader.
  // The base records just the bookkeeping so the caller can still use
  // the accessor properties.
  FReader     := AReader;
  FMaxContext := AMaxContext;
  Result := True;
end;

function TVdxModel.InitSubsystems(): Boolean;
begin
  Result := False;

  // Vulkan boots first — Initialize(-1) auto-selects the first
  // discrete GPU, falling back to the first compute-capable device.
  // Per-model GPU targeting threads through LoadModel in a later
  // phase (concurrent independent instances on different GPUs).
  if not FCompute.Initialize() then Exit;

  // LayerNorm + Attention both consume the compute device; Attention
  // additionally needs every dimensional value LoadModelConfig
  // populated on the base.
  if not FNorm.Init(FCompute) then Exit;
  if not FAttn.Init(FCompute, FHiddenDim, FNumQHeads, FNumKVHeads,
    FHeadDim, FNumLayers, FMaxSeqLen, FFFNWidth) then Exit;
  if not FFFN.Init(FCompute, FHiddenDim, FFFNWidth) then Exit;

  // Tokenizer parses GGUF metadata directly — reader was adopted by
  // LoadModelConfig. Vocab size caches the tokenizer-derived fact
  // that every downstream component (sampler, stop-token resolution)
  // needs immediately.
  if not FTokenizer.LoadFromReader(FReader) then Exit;
  FVocabSize := FTokenizer.GetVocabSize();
  Status('Tokenizer loaded: %d tokens, BOS=%d, EOS=%d',
    [FVocabSize, FTokenizer.GetBosId(), FTokenizer.GetEosId()]);

  // Decode-path scratch buffers + vec_add pipeline. Sized by
  // HiddenDim (now available from LoadModelConfig). Same machinery
  // serves batch callers via count-parameterized dispatch.
  if not BuildDecodeResources() then Exit;

  // Batch-path scratch matrices + batched embed-lookup pipelines.
  // The embedding-table GPU mirror (FEmbedGpu) is uploaded by the
  // descendant's LoadWeights — EmbedTokensBatch re-binds its
  // descriptor set on every call, so initial binding doesn't matter.
  if not BuildBatchResources() then Exit;

  Result := True;
end;

function TVdxModel.LoadWeights(): Boolean;
begin
  Result := True;
end;

procedure TVdxModel.FreeWeights();
var
  LLayer: Integer;
begin
  // Per-layer permanent norm buffers. FNorm guards against partial
  // initialization internally; safe on unset handles and zeroed records.
  if FNorm <> nil then
    for LLayer := 0 to Length(FNormWeights) - 1 do
      FNorm.FreeNormWeights(FNormWeights[LLayer]);
  FNormWeights := nil;

  // Global output norm buffer. DestroyGpuBuffer is safe on zeroed
  // records, so this works even if LoadWeights never reached the
  // upload step.
  if FCompute <> nil then
    FCompute.DestroyGpuBuffer(FOutputNormGpu);

  // Streaming records — just mmap pointers, nothing to free. Zero the
  // arrays so a re-LoadWeights pass would start clean.
  FAttnWeights := nil;
  FFFNWeights  := nil;

  // Embedding pointer is into the mmap; validity tracks the reader.
  FEmbedPtr   := nil;
  FEmbedScale := 0.0;
end;

// ---------------------------------------------------------------------------
// Forward-pass default bodies raise — these MUST be overridden by any
// concrete model that a consumer actually drives. Making them raise
// (instead of silently returning) surfaces the omission the moment the
// forward pass runs rather than producing garbage output.
// ---------------------------------------------------------------------------
procedure TVdxModel.RunLayerForwardBatch(const ALayer: Integer;
  const ANumTokens, AStartPos: UInt32;
  const ABidirectional: Boolean);
begin
  raise ENotImplemented.CreateFmt(
    '%s.RunLayerForwardBatch must be overridden by concrete model class',
    [ClassName]);
end;

procedure TVdxModel.RunLayerForward(const ALayer: Integer;
  const APosition: Integer);
begin
  raise ENotImplemented.CreateFmt(
    '%s.RunLayerForward must be overridden by concrete model class',
    [ClassName]);
end;

// ---------------------------------------------------------------------------
// Default 10,000.0 — the Llama / GPT-NeoX convention. Gemma 3 overrides
// with the `ALayer mod 6 = 5 → 1,000,000.0` pattern for global-attention
// layers.
// ---------------------------------------------------------------------------
function TVdxModel.GetRoPETheta(const ALayer: Integer): Single;
begin
  Result := 10000.0;
end;

function TVdxModel.FormatPrompt(const APrompt: string): string;
begin
  Result := APrompt;
end;

function TVdxModel.FormatEmbedding(const AText: string;
  const AIsQuery: Boolean): string;
begin
  Result := AText;
end;

function TVdxModel.GetStopTokenStrings(): TArray<string>;
begin
  Result := nil;
end;

function TVdxModel.SupportsEmbedding(): Boolean;
begin
  Result := False;
end;

// ---------------------------------------------------------------------------
// BuildDecodeResources — allocate four HiddenDim scratch buffers and
// build the vec_add pipeline + two descriptor sets. Called once from
// InitSubsystems after dimensions land and after FCompute / FFFN /
// FAttn are up. On any failure the partial state is left for
// FreeDecodeResources to clean up (it's nil-safe).
// ---------------------------------------------------------------------------
function TVdxModel.BuildDecodeResources(): Boolean;
var
  LSpv:      TBytes;
  LBufSize:  UInt64;
begin
  Result := False;

  LBufSize := UInt64(FHiddenDim) * SizeOf(Single);

  // Residual — HOST_VISIBLE so CPU embed lookup can write directly.
  FResidualGpu := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if FErrors.HasFatal() then Exit;

  // WorkBufA — DEVICE_LOCAL, used as the normed-input scratch for
  // attention and FFN dispatches.
  FWorkBufA := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if FErrors.HasFatal() then Exit;

  // AttnOutBuf — DEVICE_LOCAL, attention sub-layer output.
  FAttnOutBuf := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if FErrors.HasFatal() then Exit;

  // FFNOutBuf — DEVICE_LOCAL, FFN sub-layer output.
  FFFNOutBuf := FCompute.CreateGpuBuffer(LBufSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if FErrors.HasFatal() then Exit;

  // --- vec_add pipeline ---
  LSpv := VdxLoadShader('VEC_ADD');
  FVecAddShader := FCompute.CreateShaderModule(@LSpv[0], Length(LSpv));
  if FErrors.HasFatal() then Exit;

  FVecAddDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  if FErrors.HasFatal() then Exit;

  FVecAddBundle := FCompute.CreateComputePipelineWithPush(
    FVecAddShader, 'main', FVecAddDescLayout, SizeOf(TVdxVecAddPush));
  if FErrors.HasFatal() then Exit;

  // Four sets total (decode attn/ffn + batch attn/ffn), 2 storage
  // bindings each = 8 bindings. Batch sets get allocated later by
  // BuildBatchResources against this same pool.
  FVecAddDescPool := FCompute.CreateDescriptorPoolForStorage(4, 8);
  if FErrors.HasFatal() then Exit;

  // Bind the two accumulator paths at pipeline-build time. These
  // never change during the decode loop — avoids per-dispatch
  // descriptor updates on the hot path.
  FVecAddAttnDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FVecAddDescPool, FVecAddDescLayout, [FResidualGpu, FAttnOutBuf]);
  if FErrors.HasFatal() then Exit;

  FVecAddFFNDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FVecAddDescPool, FVecAddDescLayout, [FResidualGpu, FFFNOutBuf]);
  if FErrors.HasFatal() then Exit;

  Result := True;
end;

// ---------------------------------------------------------------------------
// FreeDecodeResources — reverse-order teardown of everything
// BuildDecodeResources created. Nil-safe on all handles, so this works
// even after a partial init failure.
// ---------------------------------------------------------------------------
procedure TVdxModel.FreeDecodeResources();
begin
  if FCompute = nil then Exit;

  // Descriptor sets are released implicitly by the pool destroy;
  // no per-set cleanup needed.
  if FVecAddDescPool <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FVecAddDescPool);
    FVecAddDescPool := VK_NULL_HANDLE;
  end;
  FVecAddAttnDescSet := VK_NULL_HANDLE;
  FVecAddFFNDescSet  := VK_NULL_HANDLE;

  FCompute.DestroyComputePipelineBundle(FVecAddBundle);

  if FVecAddDescLayout <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorSetLayoutHandle(FVecAddDescLayout);
    FVecAddDescLayout := VK_NULL_HANDLE;
  end;

  if FVecAddShader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FVecAddShader);
    FVecAddShader := VK_NULL_HANDLE;
  end;

  FCompute.DestroyGpuBuffer(FFFNOutBuf);
  FCompute.DestroyGpuBuffer(FAttnOutBuf);
  FCompute.DestroyGpuBuffer(FWorkBufA);
  FCompute.DestroyGpuBuffer(FResidualGpu);
end;

// ---------------------------------------------------------------------------
// BuildBatchResources — allocate batch-path scratch matrices, build
// the three embed-batch pipelines, and allocate batch vec_add descriptor
// sets from the decode pool. The embedding-table GPU mirror (FEmbedGpu)
// is uploaded by the descendant's LoadWeights and bound per-call by
// EmbedTokensBatch via UpdateDescriptorSetBuffers.
// ---------------------------------------------------------------------------
function TVdxModel.BuildBatchResources(): Boolean;
var
  LSpv:          TBytes;
  LHiddenMatSize: UInt64;
  LKVMatSize:     UInt64;
  LFFNMatSize:    UInt64;
begin
  Result := False;

  // All matrix buffers sized to worst-case MaxSeqLen × dim. Reused
  // across prefill calls so one allocation amortizes the whole batch
  // path.
  LHiddenMatSize := UInt64(FMaxSeqLen) * UInt64(FHiddenDim) * SizeOf(Single);
  LKVMatSize     := UInt64(FMaxSeqLen) * UInt64(FNumKVHeads) *
                    UInt64(FHeadDim) * SizeOf(Single);
  LFFNMatSize    := UInt64(FMaxSeqLen) * UInt64(FFFNWidth) * SizeOf(Single);

  FResidualMat := FCompute.CreateGpuBuffer(LHiddenMatSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if FErrors.HasFatal() then Exit;

  FWorkMat := FCompute.CreateGpuBuffer(LHiddenMatSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if FErrors.HasFatal() then Exit;

  FQMat := FCompute.CreateGpuBuffer(LHiddenMatSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if FErrors.HasFatal() then Exit;

  FKMat := FCompute.CreateGpuBuffer(LKVMatSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if FErrors.HasFatal() then Exit;

  FVMat := FCompute.CreateGpuBuffer(LKVMatSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if FErrors.HasFatal() then Exit;

  FAttnOutMat := FCompute.CreateGpuBuffer(LHiddenMatSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if FErrors.HasFatal() then Exit;

  FGateMat := FCompute.CreateGpuBuffer(LFFNMatSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if FErrors.HasFatal() then Exit;

  FUpMatBuf := FCompute.CreateGpuBuffer(LFFNMatSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if FErrors.HasFatal() then Exit;

  FFFNOutMat := FCompute.CreateGpuBuffer(LHiddenMatSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if FErrors.HasFatal() then Exit;

  // Token-ID buffer for embed_batch — HOST_VISIBLE so uploads are
  // cheap (just a memcpy, no staging round-trip).
  FTokenIdsGpu := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * SizeOf(UInt32),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if FErrors.HasFatal() then Exit;

  // --- embed-batch pipelines (one per supported embed quant format) ---
  LSpv := VdxLoadShader('EMBED_LOOKUP_BATCH_F16');
  FEmbedBatchF16Shader := FCompute.CreateShaderModule(@LSpv[0], Length(LSpv));
  if FErrors.HasFatal() then Exit;

  LSpv := VdxLoadShader('EMBED_LOOKUP_BATCH_Q8');
  FEmbedBatchQ8Shader := FCompute.CreateShaderModule(@LSpv[0], Length(LSpv));
  if FErrors.HasFatal() then Exit;

  LSpv := VdxLoadShader('EMBED_LOOKUP_BATCH_Q4_0');
  FEmbedBatchQ4Shader := FCompute.CreateShaderModule(@LSpv[0], Length(LSpv));
  if FErrors.HasFatal() then Exit;

  // Three storage bindings: (embed_table, output_matrix, token_ids).
  FEmbedBatchDescLayout := FCompute.CreateStorageDescriptorSetLayout(3);
  if FErrors.HasFatal() then Exit;

  FEmbedBatchF16Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedBatchF16Shader, 'main', FEmbedBatchDescLayout,
    SizeOf(TVdxEmbedBatchPush));
  if FErrors.HasFatal() then Exit;

  FEmbedBatchQ8Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedBatchQ8Shader, 'main', FEmbedBatchDescLayout,
    SizeOf(TVdxEmbedBatchPush));
  if FErrors.HasFatal() then Exit;

  FEmbedBatchQ4Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedBatchQ4Shader, 'main', FEmbedBatchDescLayout,
    SizeOf(TVdxEmbedBatchPush));
  if FErrors.HasFatal() then Exit;

  // One descriptor set, rebound per call with current output matrix.
  FEmbedBatchDescPool := FCompute.CreateDescriptorPoolForStorage(1, 3);
  if FErrors.HasFatal() then Exit;

  FEmbedBatchDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FEmbedBatchDescPool, FEmbedBatchDescLayout,
    [FResidualMat, FResidualMat, FTokenIdsGpu]);
  if FErrors.HasFatal() then Exit;

  // --- batch vec_add descriptor sets (pool was sized for 4 in BuildDecodeResources) ---
  FVecAddBatchAttnDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FVecAddDescPool, FVecAddDescLayout, [FResidualMat, FAttnOutMat]);
  if FErrors.HasFatal() then Exit;

  FVecAddBatchFFNDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FVecAddDescPool, FVecAddDescLayout, [FResidualMat, FFFNOutMat]);
  if FErrors.HasFatal() then Exit;

  Result := True;
end;

// ---------------------------------------------------------------------------
// FreeBatchResources — mirror of BuildBatchResources. Descriptor sets
// allocated from FVecAddDescPool die when that pool is destroyed in
// FreeDecodeResources; we just null their references.
// ---------------------------------------------------------------------------
procedure TVdxModel.FreeBatchResources();
begin
  if FCompute = nil then Exit;

  FVecAddBatchAttnDescSet := VK_NULL_HANDLE;
  FVecAddBatchFFNDescSet  := VK_NULL_HANDLE;

  if FEmbedBatchDescPool <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorPoolHandle(FEmbedBatchDescPool);
    FEmbedBatchDescPool := VK_NULL_HANDLE;
  end;
  FEmbedBatchDescSet := VK_NULL_HANDLE;

  FCompute.DestroyComputePipelineBundle(FEmbedBatchQ4Bundle);
  FCompute.DestroyComputePipelineBundle(FEmbedBatchQ8Bundle);
  FCompute.DestroyComputePipelineBundle(FEmbedBatchF16Bundle);

  if FEmbedBatchDescLayout <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyDescriptorSetLayoutHandle(FEmbedBatchDescLayout);
    FEmbedBatchDescLayout := VK_NULL_HANDLE;
  end;

  if FEmbedBatchQ4Shader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FEmbedBatchQ4Shader);
    FEmbedBatchQ4Shader := VK_NULL_HANDLE;
  end;
  if FEmbedBatchQ8Shader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FEmbedBatchQ8Shader);
    FEmbedBatchQ8Shader := VK_NULL_HANDLE;
  end;
  if FEmbedBatchF16Shader <> VK_NULL_HANDLE then
  begin
    FCompute.DestroyShaderModuleHandle(FEmbedBatchF16Shader);
    FEmbedBatchF16Shader := VK_NULL_HANDLE;
  end;

  // Embedding-table GPU mirror. Uploaded by descendant LoadWeights;
  // torn down here because it's a base-level resource.
  FCompute.DestroyGpuBuffer(FEmbedGpu);

  FCompute.DestroyGpuBuffer(FTokenIdsGpu);
  FCompute.DestroyGpuBuffer(FFFNOutMat);
  FCompute.DestroyGpuBuffer(FUpMatBuf);
  FCompute.DestroyGpuBuffer(FGateMat);
  FCompute.DestroyGpuBuffer(FAttnOutMat);
  FCompute.DestroyGpuBuffer(FVMat);
  FCompute.DestroyGpuBuffer(FKMat);
  FCompute.DestroyGpuBuffer(FQMat);
  FCompute.DestroyGpuBuffer(FWorkMat);
  FCompute.DestroyGpuBuffer(FResidualMat);
end;

// ---------------------------------------------------------------------------
// UploadTensorToDevice — stage-then-copy upload for DEVICE_LOCAL
// target buffers. UploadToBuffer alone does vkMapMemory → memcpy,
// which only works on HOST_VISIBLE memory; anything destined for
// DEVICE_LOCAL has to go through a host-visible staging buffer and
// a GPU-side copy. Mirrors the reference's UploadWeightTensor helper.
// Caller owns the returned buffer and must DestroyGpuBuffer it.
// ---------------------------------------------------------------------------
function TVdxModel.UploadTensorToDevice(const AData: PByte;
  const ASize: UInt64): TVdxGpuBuffer;
var
  LStaging: TVdxGpuBuffer;
begin
  FillChar(Result, SizeOf(Result), 0);

  LStaging := FCompute.CreateGpuBuffer(ASize,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if FErrors.HasFatal() then Exit;
  try
    if not FCompute.UploadToBuffer(LStaging, AData, ASize) then Exit;

    Result := FCompute.CreateGpuBuffer(ASize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if FErrors.HasFatal() then
    begin
      FillChar(Result, SizeOf(Result), 0);
      Exit;
    end;

    if not FCompute.CopyBuffer(LStaging, Result, ASize) then
    begin
      FCompute.DestroyGpuBuffer(Result);
      FillChar(Result, SizeOf(Result), 0);
      Exit;
    end;
  finally
    FCompute.DestroyGpuBuffer(LStaging);
  end;
end;

// ---------------------------------------------------------------------------
// EmbedTokensBatch — GPU-side batched embedding lookup. Writes
// ANumTokens HiddenDim-sized rows into AOutputBuf, each scaled by
// FEmbedScale = sqrt(HiddenDim). Dispatch grid and DimParam differ by
// embed quant format (F16 uses 2D grid with vec2 loads; Q8_0/Q4_0 use
// 1D grid with one workgroup per token).
// ---------------------------------------------------------------------------
procedure TVdxModel.EmbedTokensBatch(const ATokenIds: TArray<Integer>;
  const ANumTokens: Integer; const AOutputBuf: TVdxGpuBuffer);
var
  LIds:  array of UInt32;
  LI:    Integer;
  LPush: TVdxEmbedBatchPush;
begin
  if ANumTokens <= 0 then Exit;

  // Integer → UInt32 repack; shader reads unsigned indices.
  SetLength(LIds, ANumTokens);
  for LI := 0 to ANumTokens - 1 do
    LIds[LI] := UInt32(ATokenIds[LI]);
  FCompute.UploadToBuffer(FTokenIdsGpu, @LIds[0],
    UInt64(ANumTokens) * SizeOf(UInt32));

  // Rebind descriptor set with current output matrix — caller decides
  // where the embedding rows land.
  FCompute.UpdateDescriptorSetBuffers(FEmbedBatchDescSet,
    [FEmbedGpu, AOutputBuf, FTokenIdsGpu]);

  LPush.EmbedScale := FEmbedScale;
  LPush.NumTokens  := UInt32(ANumTokens);

  if FEmbedType = gtQ4_0 then
  begin
    LPush.DimParam := FHiddenDim;
    FCompute.DispatchComputeWithPush(
      FEmbedBatchQ4Bundle.Pipeline, FEmbedBatchQ4Bundle.PipelineLayout,
      FEmbedBatchDescSet, @LPush, SizeOf(LPush),
      UInt32(ANumTokens));  // 1D: one workgroup per token
  end
  else if FEmbedType = gtQ8_0 then
  begin
    LPush.DimParam := FHiddenDim;
    FCompute.DispatchComputeWithPush(
      FEmbedBatchQ8Bundle.Pipeline, FEmbedBatchQ8Bundle.PipelineLayout,
      FEmbedBatchDescSet, @LPush, SizeOf(LPush),
      UInt32(ANumTokens));  // 1D: one workgroup per token
  end
  else
  begin
    // F16 / F32 path — 2D dispatch, vec2 loads process 2 dims at a time.
    LPush.DimParam := FHiddenDim div 2;
    FCompute.DispatchComputeWithPush(
      FEmbedBatchF16Bundle.Pipeline, FEmbedBatchF16Bundle.PipelineLayout,
      FEmbedBatchDescSet, @LPush, SizeOf(LPush),
      (FHiddenDim div 2 + 255) div 256, UInt32(ANumTokens));
  end;
end;

// ---------------------------------------------------------------------------
// LoadModel — the single public entry point consumers use to get a
// ready-to-drive model.
//
// Steps:
//   1. Sanity-check the path exists. Return nil on miss — no point
//      instantiating the reader to fail a moment later.
//   2. Create the reader and load the GGUF. Reader owns the mmap; model
//      owns the reader.
//   3. Read `general.architecture`, look up the concrete class.
//   4. Instantiate concrete class, wire status callback, run the three
//      lifecycle hooks. Any failure → free everything, return nil.
//
// On failure the function returns nil — the caller should null-check
// before use. Diagnostic messages for this build go to OutputDebugString
// / console via the reader's own error surface; a richer error-reporting
// contract for the factory can be added when a consumer actually needs it.
// ---------------------------------------------------------------------------
class function TVdxModel.LoadModel(const AGGUFPath: string;
  const AMaxContext: Integer;
  const AStatusCallback: TVdxStatusCallback;
  const AStatusUserData: Pointer): TVdxModel;
var
  LReader:    TVdxGGUFReader;
  LArch:      string;
  LModelCls:  TVdxModelClass;
  LModel:     TVdxModel;
  LOwnsReader: Boolean;
begin
  Result := nil;

  // Step 1 — the path must actually exist. Catches typos before we
  // spin up the reader or touch Vulkan.
  if (AGGUFPath = '') or not FileExists(AGGUFPath) then
    Exit;

  LReader := nil;
  LModel  := nil;
  LOwnsReader := True;
  try
    // Step 2 — open the GGUF. Reader's Open returns False and writes
    // to its own error buffer on failure; we just need to bail in
    // that case.
    LReader := TVdxGGUFReader.Create();
    if not LReader.Open(AGGUFPath) then Exit;

    // Step 3 — resolve the architecture string to a concrete class.
    // The reader treats missing keys as "not an error" (callers
    // decide significance) — we decide: no architecture, no load.
    if not LReader.HasMetadata('general.architecture') then Exit;
    LArch := LReader.GetMetadataString('general.architecture', '');
    if LArch = '' then Exit;

    LModelCls := TVdxModelRegistry.ResolveClass(LArch);
    if LModelCls = nil then Exit;

    // Step 4 — instantiate + wire + load.
    LModel := LModelCls.Create();
    LModel.SetStatusCallback(AStatusCallback, AStatusUserData);
    LModel.FGGUFPath     := AGGUFPath;
    LModel.FArchitecture := LArch;

    // LoadModelConfig adopts the reader on success — ownership
    // transfers to the model. LOwnsReader flag prevents the finally
    // block from double-freeing it.
    if not LModel.LoadModelConfig(LReader, AMaxContext) then
    begin
      // Base LoadModelConfig always adopts the reader via FReader :=
      // AReader before the descendant's validation runs, so ownership
      // has already transferred even when the descendant's override
      // returns False. Clear LOwnsReader so the finally block doesn't
      // double-free, then return the partial model with errors intact.
      LOwnsReader := False;
      Result := LModel;
      LModel := nil;
      Exit;
    end;
    LOwnsReader := False;

    if not LModel.InitSubsystems() then
    begin
      Result := LModel;
      LModel := nil;
      Exit;
    end;
    if not LModel.LoadWeights() then
    begin
      Result := LModel;
      LModel := nil;
      Exit;
    end;

    // Success — hand the fully-loaded model to the caller.
    Result := LModel;
    LModel := nil;
  finally
    // Any path that didn't reach the success assignment above leaves
    // LModel non-nil here; free it (which runs FreeWeights and then
    // frees the adopted reader). If config adoption never happened,
    // we also need to free LReader ourselves.
    LModel.Free();
    if LOwnsReader then
      LReader.Free();
  end;
end;

end.
