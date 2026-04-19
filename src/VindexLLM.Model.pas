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
  VindexLLM.GGUFReader;

type
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
    property Reader: TVdxGGUFReader read FReader;
  end;

implementation

uses
  System.SysUtils,
  VindexLLM.Model.Registry;

{ TVdxModel }

constructor TVdxModel.Create();
begin
  inherited;
  FReader     := nil;
  FArchitecture := '';
  FGGUFPath   := '';
  FMaxContext := 0;
end;

destructor TVdxModel.Destroy();
begin
  // Descendants release VRAM in FreeWeights. Call unconditionally —
  // implementations must be safe against partial or absent
  // initialization.
  FreeWeights();

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
  Result := True;
end;

function TVdxModel.LoadWeights(): Boolean;
begin
  Result := True;
end;

procedure TVdxModel.FreeWeights();
begin
  // No-op. Descendants override to release Vulkan buffers.
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
    if not LModel.LoadModelConfig(LReader, AMaxContext) then Exit;
    LOwnsReader := False;

    if not LModel.InitSubsystems() then Exit;
    if not LModel.LoadWeights()    then Exit;

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
