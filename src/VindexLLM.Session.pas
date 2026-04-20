{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Session;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.IOUtils,
  VindexLLM.Utils,
  VindexLLM.Inference,
  VindexLLM.Memory,
  VindexLLM.Embeddings,
  VindexLLM.Sampler;


const
  // BM25 hits on rebuild retrieval
  CVdxSessionFTS5TopK = 5;

  // Cosine hits on rebuild retrieval (if embedder attached)
  CVdxSessionVectorTopK = 5;

  // Recent turns preserved on rebuild for conversational flow
  CVdxSessionRecentTurns = 3;

  // Max unique retrieved turns after dedup (FTS5 + vector merged)
  CVdxSessionMergeTopK = 5;

type

  { TVdxRetrievalConfig }
  TVdxRetrievalConfig = record
    Enabled: Boolean;       // master switch (default True)
    TopK: Integer;          // max retrieved items per Chat() call (default 3)
    MinScore: Single;       // minimum cosine similarity to include (default 0.5)
  end;

  { TVdxSession }
  TVdxSession = class(TVdxBaseObject)
  private
    // Owned subsystems — created/destroyed by TVdxSession
    FInference: TVdxInference;
    FMemory: TVdxMemory;
    FEmbedder: TVdxEmbeddings;     // nil when no embedder path supplied

    // Configuration
    FSystemPrompt: string;
    FTurnIndex: Integer;           // tracks turn count for prompt assembly
    FLastUserMessage: string;      // raw user text for rebuild search queries
    FRetrievalConfig: TVdxRetrievalConfig;


    // Caller-facing callbacks — stored here, forwarded to FInference
    FTokenCallback: TVdxCallback<TVdxTokenCallback>;
    FCancelCallback: TVdxCallback<TVdxCancelCallback>;

    // Shared retrieval helper — searches FTS5 + optional vector,
    // deduplicates by TurnId, returns up to ATopK merged results.
    function RetrieveContext(const AQuery: string;
      const ATopK: Integer): TArray<TVdxMemoryTurn>;

    // Formats retrieved turns into a labeled context block for prompt
    // injection. Chunks and facts become "Reference information".
    function FormatRetrievedContext(
      const ATurns: TArray<TVdxMemoryTurn>): string;

    // Prompt formatting — builds Gemma 3 chat template strings.
    // AContext is a pre-formatted context block (or '' for none).
    function FormatPrompt(const AUserMessage: string;
      const AContext: string): string;

    // Rebuild handler — installed on FInference.SetRebuildCallback.
    // Queries FMemory for relevant context and assembles a replacement
    // prompt from system + retrieved + recent + current user message.
    function HandleRebuild(const APosition: UInt32;
      const AMaxContext: UInt32; const APrompt: string): string;

  public
    constructor Create(); override;
    destructor Destroy(); override;
    procedure SetErrors(const AErrors: TVdxErrors); override;

    // --- Lifecycle ---
    function LoadModel(
      const AModelPath: string;
      const AMemoryDbPath: string;
      const AEmbedderPath: string;
      const AMaxContext: Integer = 2048;
      const ARebuildAt: Integer = -1
    ): Boolean;

    procedure UnloadModel();
    function IsLoaded(): Boolean;


    // --- Configuration ---
    procedure SetSystemPrompt(const APrompt: string);
    procedure SetSamplerConfig(const AConfig: TVdxSamplerConfig);
    procedure SetRetrievalConfig(const AConfig: TVdxRetrievalConfig);
    class function DefaultRetrievalConfig(): TVdxRetrievalConfig; static;
    procedure SetTokenCallback(const ACallback: TVdxTokenCallback;
      const AUserData: Pointer);
    procedure SetCancelCallback(const ACallback: TVdxCancelCallback;
      const AUserData: Pointer);

    // --- Conversation ---
    function Chat(const AUserMessage: string;
      const AMaxTokens: Integer = 256): string;

    // Reset conversation state — clears the KV cache, purges all turns
    // from memory, and resets the turn index. The next Chat() call
    // behaves as a fresh session (BOS, system prompt injected again).
    // Memory DB schema stays intact; only data is cleared.
    procedure ClearHistory();

    // --- Knowledge ---
    function AddDocument(const ASource: string; const ATitle: string;
      const AText: string; const AChunkTokens: Integer = 512;
      const AOverlapTokens: Integer = 64;
      const APinned: Boolean = False): Int64;
    function AddFact(const AText: string;
      const APinned: Boolean = True): Int64;

    // --- Info ---
    function GetStats(): PVdxInferenceStats;
    function GetTurnCount(): Integer;
  end;

implementation


{ TVdxSession }

constructor TVdxSession.Create();
begin
  inherited Create();
  FInference := nil;
  FMemory := nil;
  FEmbedder := nil;
  FSystemPrompt := '';
  FTurnIndex := 0;
  FLastUserMessage := '';
  FRetrievalConfig := DefaultRetrievalConfig();
  FTokenCallback := Default(TVdxCallback<TVdxTokenCallback>);
  FCancelCallback := Default(TVdxCallback<TVdxCancelCallback>);
end;

destructor TVdxSession.Destroy();
begin
  UnloadModel();
  inherited Destroy();
end;

procedure TVdxSession.SetErrors(const AErrors: TVdxErrors);
begin
  inherited SetErrors(AErrors);
  if Assigned(FInference) then FInference.SetErrors(AErrors);
  if Assigned(FMemory) then FMemory.SetErrors(AErrors);
  if Assigned(FEmbedder) then FEmbedder.SetErrors(AErrors);
end;

function TVdxSession.LoadModel(const AModelPath: string;
  const AMemoryDbPath: string; const AEmbedderPath: string;
  const AMaxContext: Integer; const ARebuildAt: Integer): Boolean;
var
  LLoaded: Boolean;
  LOpened: Boolean;
begin
  Result := False;

  // Reset if already loaded
  if IsLoaded() then
    UnloadModel();

  FErrors.Clear();

  // Validate inputs
  if not TFile.Exists(AModelPath) then
  begin
    FErrors.Add(esError, 'SESSION', 'Model file not found: %s',
      [AModelPath]);
    Exit;
  end;

  if AMemoryDbPath.Trim().IsEmpty() then
  begin
    FErrors.Add(esError, 'SESSION', 'Memory DB path must not be empty');
    Exit;
  end;

  // --- Create and load inference engine ---
  FInference := TVdxInference.Create();
  FInference.SetErrors(FErrors);
  LLoaded := FInference.LoadModel(AModelPath, AMaxContext, ARebuildAt);

  if not LLoaded then
  begin
    FreeAndNil(FInference);
    Exit;
  end;

  // --- Create and open memory DB ---
  FMemory := TVdxMemory.Create();
  FMemory.SetErrors(FErrors);
  LOpened := FMemory.OpenSession(AMemoryDbPath);
  if not LOpened then
  begin
    FErrors.Add(esError, 'SESSION', 'Failed to open memory DB: %s',
      [AMemoryDbPath]);
    FInference.UnloadModel();
    FreeAndNil(FInference);
    FreeAndNil(FMemory);
    Exit;
  end;

  // --- Optionally create and load embedder ---
  if AEmbedderPath.Trim() <> '' then
  begin
    if not TFile.Exists(AEmbedderPath) then
    begin
      FErrors.Add(esWarning, 'SESSION',
        'Embedder file not found, continuing without vector search: %s',
        [AEmbedderPath]);
    end
    else
    begin
      FEmbedder := TVdxEmbeddings.Create();
      FEmbedder.SetErrors(FErrors);
      LLoaded := FEmbedder.LoadModel(AEmbedderPath);

      if not LLoaded then
      begin
        // Non-fatal — continue without vector search
        FErrors.Add(esWarning, 'SESSION',
          'Embedder failed to load, continuing without vector search');
        FreeAndNil(FEmbedder);
      end
      else
      begin
        FMemory.AttachEmbeddings(FEmbedder);
      end;
    end;
  end;

  // --- Forward pre-set callbacks ---
  if FTokenCallback.IsAssigned() then
    FInference.SetTokenCallback(FTokenCallback.Callback,
      FTokenCallback.UserData);

  if FCancelCallback.IsAssigned() then
    FInference.SetCancelCallback(FCancelCallback.Callback,
      FCancelCallback.UserData);

  // --- Install rebuild callback ---
  FInference.SetRebuildCallback(
    function(const APosition: UInt32; const AMaxCtx: UInt32;
      const APrompt: string; const AUserData: Pointer): string
    begin
      Result := HandleRebuild(APosition, AMaxCtx, APrompt);
    end,
    nil);

  FTurnIndex := 0;
  Result := True;
end;


procedure TVdxSession.UnloadModel();
begin
  if Assigned(FEmbedder) then
  begin
    if Assigned(FMemory) then
      FMemory.DetachEmbeddings();
    FEmbedder.UnloadModel();
    FreeAndNil(FEmbedder);
  end;

  if Assigned(FMemory) then
  begin
    FMemory.CloseSession();
    FreeAndNil(FMemory);
  end;

  if Assigned(FInference) then
  begin
    FInference.UnloadModel();
    FreeAndNil(FInference);
  end;

  FTurnIndex := 0;
end;

function TVdxSession.IsLoaded(): Boolean;
begin
  Result := Assigned(FInference) and Assigned(FMemory);
end;

procedure TVdxSession.SetSystemPrompt(const APrompt: string);
begin
  FSystemPrompt := APrompt;
end;

procedure TVdxSession.SetSamplerConfig(const AConfig: TVdxSamplerConfig);
begin
  if Assigned(FInference) then
    FInference.SetSamplerConfig(AConfig);
end;

procedure TVdxSession.SetRetrievalConfig(const AConfig: TVdxRetrievalConfig);
begin
  FRetrievalConfig := AConfig;
end;

class function TVdxSession.DefaultRetrievalConfig(): TVdxRetrievalConfig;
begin
  Result.Enabled := True;
  Result.TopK := 3;
  Result.MinScore := 0.5;
end;

procedure TVdxSession.SetTokenCallback(const ACallback: TVdxTokenCallback;
  const AUserData: Pointer);
begin
  FTokenCallback.Callback := ACallback;
  FTokenCallback.UserData := AUserData;
  if Assigned(FInference) then
    FInference.SetTokenCallback(ACallback, AUserData);
end;

procedure TVdxSession.SetCancelCallback(const ACallback: TVdxCancelCallback;
  const AUserData: Pointer);
begin
  FCancelCallback.Callback := ACallback;
  FCancelCallback.UserData := AUserData;
  if Assigned(FInference) then
    FInference.SetCancelCallback(ACallback, AUserData);
end;

function TVdxSession.Chat(const AUserMessage: string;
  const AMaxTokens: Integer): string;
var
  LPrompt: string;
  LResponse: string;
  LContext: string;
  LRetrieved: TArray<TVdxMemoryTurn>;
begin
  Result := '';

  if not IsLoaded() then
    Exit;

  // 1. Store raw user text for rebuild search queries
  FLastUserMessage := AUserMessage;

  // 2. Per-turn RAG retrieval — run BEFORE logging the user turn so
  // the just-asked question doesn't self-match in FTS5/vector search
  // and burn a retrieval slot on itself.
  LContext := '';
  if FRetrievalConfig.Enabled and (FRetrievalConfig.TopK > 0) then
  begin
    LRetrieved := RetrieveContext(AUserMessage, FRetrievalConfig.TopK);

    if Length(LRetrieved) > 0 then
      LContext := FormatRetrievedContext(LRetrieved);
  end;

  // 3. Log user turn to memory (after retrieval — see comment above)
  FMemory.AppendTurn(CVdxMemRoleUser, AUserMessage, 0);

  // 4. Assemble prompt (system + context + user)
  LPrompt := FormatPrompt(AUserMessage, LContext);

  // 5. Generate assistant response
  LResponse := FInference.Generate(LPrompt, AMaxTokens, False);

  // 6. Assistant turn is NOT logged to memory. Freely-generated model
  // output varies between runs and never deduplicates, so storing it
  // pollutes semantic retrieval with noise (especially failure-mode
  // responses like "I don't know your name" cosine-matching future
  // "what is my name" queries). The assistant side still flows through
  // the KV cache for in-session continuity; only long-term memory skips
  // it. User facts (the signal worth retrieving) come from user turns.

  // 7. Track turn count — one user turn logged per Chat call.
  Inc(FTurnIndex, 1);

  Result := LResponse;
end;

procedure TVdxSession.ClearHistory();
begin
  if not IsLoaded() then
    Exit;

  FInference.ResetKVCache();
  FMemory.PurgeAll();
  FTurnIndex := 0;
  FLastUserMessage := '';
end;

function TVdxSession.FormatPrompt(const AUserMessage: string;
  const AContext: string): string;
var
  LContent: string;
begin
  // Build the content to place inside the user turn.
  // Layout: [system prompt] [context block] [user message]
  LContent := '';

  // System prompt on first turn only — prefixed with 'System:' so the
  // instruction-tuned model treats it as a directive, not user chat.
  if (FInference.GetKVCachePosition() = 0) and (FSystemPrompt <> '') then
    LContent := 'System: ' + FSystemPrompt + #10 + #10;

  // Injected RAG context (if any)
  if AContext <> '' then
    LContent := LContent + AContext + #10 + #10;

  // User message
  LContent := LContent + AUserMessage;

  // Wrap in model-specific chat template via TVdxModel.FormatPrompt
  Result := FInference.Model.FormatPrompt(LContent);

  // For continuation turns, prepend newline to bridge after previous
  // <end_of_turn> token that is already in the KV cache.
  if FInference.GetKVCachePosition() > 0 then
    Result := #10 + Result;
end;

function TVdxSession.RetrieveContext(const AQuery: string;
  const ATopK: Integer): TArray<TVdxMemoryTurn>;
var
  LRaw: TArray<TVdxMemoryTurn>;
  LCount: Integer;
  LI: Integer;
begin
  // Semantic retrieval only. FTS5 keyword hits were flooding the top-K
  // and squeezing out the vector results — making the embedder
  // essentially dead weight. Cosine similarity against the embedder is
  // the whole point of having one, so defer to it exclusively.
  SetLength(Result, 0);
  if (FEmbedder = nil) or (not FEmbedder.IsLoaded()) then
    Exit;

  try
    LRaw := FMemory.SearchVector(AQuery, ATopK);
  except
    SetLength(Result, 0);
    Exit;
  end;

  // Filter by minimum cosine similarity threshold. SearchVector returns
  // results sorted by cosine DESC, so once we hit a score below the
  // threshold, all subsequent entries are also below — early exit.
  LCount := 0;
  SetLength(Result, Length(LRaw));
  for LI := 0 to High(LRaw) do
  begin
    if LRaw[LI].CosineScore < FRetrievalConfig.MinScore then
      Break;
    Result[LCount] := LRaw[LI];
    Inc(LCount);
  end;
  SetLength(Result, LCount);
end;

function TVdxSession.FormatRetrievedContext(
  const ATurns: TArray<TVdxMemoryTurn>): string;
var
  LI: Integer;
  LRole: string;
  LLabel: string;
begin
  if Length(ATurns) = 0 then
  begin
    Result := '';
    Exit;
  end;

  // Label each retrieved turn with its source so the instruction-tuned
  // model understands whether a past statement came from the user or
  // from the assistant. Facts and document chunks use 'reference'
  // since they're neither speaker side.
  Result := 'Relevant prior context:';
  for LI := 0 to High(ATurns) do
  begin
    LRole := ATurns[LI].Role;
    if LRole = CVdxMemRoleUser then
      LLabel := 'user'
    else if LRole = CVdxMemRoleAssistant then
      LLabel := 'assistant'
    else
      LLabel := 'reference';

    Result := Result + #10 + '[' + LLabel + '] ' + ATurns[LI].Text;
  end;
end;

function TVdxSession.HandleRebuild(const APosition: UInt32;
  const AMaxContext: UInt32; const APrompt: string): string;
var
  LResult: string;
begin
  // Start the new cache like a fresh session: system prompt + current
  // user message. Generate() wraps this in the chat template and
  // prefills it against the freshly zeroed KV. No RAG rebuild — per-turn
  // RAG already fired in Chat() before this callback, and subsequent
  // turns will RAG naturally from the DB.
  LResult := '';

  if FSystemPrompt <> '' then
    LResult := 'System: ' + FSystemPrompt + #10 + #10;

  LResult := LResult + FLastUserMessage;

  Result := LResult;
end;

function TVdxSession.AddDocument(const ASource: string;
  const ATitle: string; const AText: string;
  const AChunkTokens: Integer; const AOverlapTokens: Integer;
  const APinned: Boolean): Int64;
begin
  if Assigned(FMemory) then
    Result := FMemory.AddDocument(ASource, ATitle, AText, AChunkTokens,
      AOverlapTokens, APinned)
  else
    Result := -1;
end;

function TVdxSession.AddFact(const AText: string;
  const APinned: Boolean): Int64;
begin
  if Assigned(FMemory) then
    Result := FMemory.AddFact(AText, APinned)
  else
    Result := -1;
end;

function TVdxSession.GetStats(): PVdxInferenceStats;
begin
  if Assigned(FInference) then
    Result := FInference.GetStats()
  else
    Result := nil;
end;

function TVdxSession.GetTurnCount(): Integer;
begin
  if Assigned(FMemory) then
    Result := FMemory.GetTurnCount()
  else
    Result := 0;
end;

end.
