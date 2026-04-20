{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Chat;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.IOUtils,
  System.Generics.Collections,
  VindexLLM.Utils,
  VindexLLM.Inference,
  VindexLLM.Session,
  VindexLLM.Sampler;

type
  { TVdxChat }
  TVdxChat = class(TVdxBaseObject)
  private
    FSession: TVdxSession;
    FModelPath: string;
    FEmbedderPath: string;
    FMemoryDbPath: string;
    FMaxContext: Integer;
    FRebuildAt: Integer;
    FMaxTokens: Integer;
    FSystemPrompt: string;
    FSamplerConfig: TVdxSamplerConfig;
    FRetrievalConfig: TVdxRetrievalConfig;
    FRunning: Boolean;

    // Parse and dispatch a /command
    procedure ProcessCommand(const AInput: string);

    // Process a regular chat message via Session.Chat()
    procedure ProcessChat(const AInput: string);

    // Print built-in help text via DoInfo
    procedure PrintHelp();

    // Print inference stats via DoInfo
    procedure PrintStats();

    // Print session errors/warnings via DoInfo
    procedure PrintErrors();
  protected
    // --- Abstract I/O (must override) ---

    // Read one line of input from the user. Blocks until available.
    function  DoGetInput(): string; virtual; abstract;

    // Display a complete output line (command result, info, etc.)
    procedure DoOutput(const AText: string); virtual; abstract;

    // Stream a single token during generation
    procedure DoToken(const AToken: string); virtual; abstract;

    // Check if the user has requested cancellation
    function  DoCancel(): Boolean; virtual; abstract;

    // --- Virtual hooks (override to customize) ---

    // Status messages from model loading
    procedure DoStatus(const AText: string); virtual;

    // Error messages
    procedure DoError(const AText: string); virtual;

    // Informational messages
    procedure DoInfo(const AText: string); virtual;

    // Called once before model load
    procedure DoStartup(); virtual;

    // Called once after session teardown
    procedure DoShutdown(); virtual;
    // Called after each generation completes
    procedure DoGenerationComplete(); virtual;

    // Extension point for custom commands. Return True if handled.
    function  DoCommand(const ACmd: string; const AArgs: string): Boolean; virtual;

    // Access to the owned session (for derived class customization)
    function  GetSession(): TVdxSession;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Main loop — template method. Creates session, loads model,
    // runs input loop, unloads on exit.
    procedure Run();

    // Properties — configure before calling Run()
    property ModelPath: string read FModelPath write FModelPath;
    property EmbedderPath: string read FEmbedderPath write FEmbedderPath;
    property MemoryDbPath: string read FMemoryDbPath write FMemoryDbPath;
    property MaxContext: Integer read FMaxContext write FMaxContext;
    property RebuildAt: Integer read FRebuildAt write FRebuildAt;
    property MaxTokens: Integer read FMaxTokens write FMaxTokens;
    property SystemPrompt: string read FSystemPrompt write FSystemPrompt;
    property SamplerConfig: TVdxSamplerConfig read FSamplerConfig write FSamplerConfig;
    property RetrievalConfig: TVdxRetrievalConfig read FRetrievalConfig write FRetrievalConfig;
  end;

implementation

{ TVdxChat }

constructor TVdxChat.Create();
begin
  inherited Create();
  FSession := nil;
  FModelPath := '';
  FEmbedderPath := '';
  FMemoryDbPath := '';
  FMaxContext := 2048;
  FRebuildAt := -1;
  FMaxTokens := 1024;
  FSystemPrompt := '';
  FSamplerConfig := TVdxSampler.DefaultConfig();
  FRetrievalConfig := TVdxSession.DefaultRetrievalConfig();
  FRunning := False;
end;

destructor TVdxChat.Destroy();
begin
  inherited Destroy();
end;

procedure TVdxChat.DoStatus(const AText: string);
begin
  DoOutput(AText);
end;

procedure TVdxChat.DoError(const AText: string);
begin
  DoOutput('ERROR: ' + AText);
end;

procedure TVdxChat.DoInfo(const AText: string);
begin
  DoOutput(AText);
end;

procedure TVdxChat.DoStartup();
begin
  // no-op — override in derived class
end;

procedure TVdxChat.DoShutdown();
begin
  // no-op — override in derived class
end;
procedure TVdxChat.DoGenerationComplete();
begin
  // no-op — override to flush token writer, etc.
end;

function TVdxChat.DoCommand(const ACmd: string;
  const AArgs: string): Boolean;
begin
  Result := False;
end;

function TVdxChat.GetSession(): TVdxSession;
begin
  Result := FSession;
end;

procedure TVdxChat.PrintErrors();
var
  LErrors: TVdxErrors;
  LItems: TList<TVdxError>;
  LI: Integer;
  LErr: TVdxError;
begin
  if FSession = nil then
    Exit;

  LErrors := FSession.GetErrors();
  if (LErrors = nil) or (LErrors.GetItems().Count = 0) then
    Exit;

  LItems := LErrors.GetItems();
  for LI := 0 to LItems.Count - 1 do
  begin
    LErr := LItems[LI];
    DoInfo(Format('[%s] %s: %s',
      [LErr.GetSeverityString(), LErr.Code, LErr.Message]));
  end;
end;

procedure TVdxChat.Run();
var
  LLoaded: Boolean;
  LInput: string;
begin
  DoStartup();

  // Create session
  FSession := TVdxSession.Create();
  try
    // Wire callbacks via anonymous methods that bridge to Do* virtuals
    FSession.SetStatusCallback(
      procedure(const AText: string; const AUserData: Pointer)
      begin
        Self.DoStatus(AText);
      end, nil);

    FSession.SetTokenCallback(
      procedure(const AToken: string; const AUserData: Pointer)
      begin
        Self.DoToken(AToken);
      end, nil);

    FSession.SetCancelCallback(
      function(const AUserData: Pointer): Boolean
      begin
        Result := Self.DoCancel();
      end, nil);
    // Load model
    LLoaded := FSession.LoadModel(FModelPath, FMemoryDbPath,
      FEmbedderPath, FMaxContext, FRebuildAt);
    PrintErrors();

    if not LLoaded then
    begin
      DoError('Failed to load model.');
      FreeAndNil(FSession);
      DoShutdown();
      Exit;
    end;

    // Configure session
    if FSystemPrompt <> '' then
      FSession.SetSystemPrompt(FSystemPrompt);
    FSession.SetSamplerConfig(FSamplerConfig);
    FSession.SetRetrievalConfig(FRetrievalConfig);

    // Main input loop
    FRunning := True;
    while FRunning do
    begin
      LInput := DoGetInput();
      if LInput = '' then
        Continue;

      if LInput.StartsWith('/') then
        ProcessCommand(LInput)
      else
        ProcessChat(LInput);
    end;
    // Unload and cleanup
    FSession.UnloadModel();
  finally
    FreeAndNil(FSession);
  end;

  DoShutdown();
end;

procedure TVdxChat.ProcessCommand(const AInput: string);
var
  LSpacePos: Integer;
  LCmd: string;
  LArgs: string;
  LText: string;
  LNewMax: Integer;
begin
  // Split at first space: /command args
  LSpacePos := Pos(' ', AInput);
  if LSpacePos > 0 then
  begin
    LCmd := LowerCase(Copy(AInput, 1, LSpacePos - 1));
    LArgs := Trim(Copy(AInput, LSpacePos + 1, MaxInt));
  end
  else
  begin
    LCmd := LowerCase(AInput);
    LArgs := '';
  end;
  if LCmd = '/quit' then
  begin
    FRunning := False;
  end
  else if LCmd = '/clear' then
  begin
    FSession.ClearHistory();
    DoInfo('Conversation cleared.');
  end
  else if LCmd = '/system' then
  begin
    if LArgs <> '' then
    begin
      FSession.SetSystemPrompt(LArgs);
      DoInfo('System prompt updated.');
    end
    else
      DoInfo('Usage: /system <prompt text>');
  end
  else if LCmd = '/addfact' then
  begin
    if LArgs <> '' then
    begin
      FSession.AddFact(LArgs);
      DoInfo('Fact added.');
    end
    else
      DoInfo('Usage: /addfact <fact text>');
  end
  else if LCmd = '/addfile' then
  begin
    if LArgs = '' then
      DoInfo('Usage: /addfile <path>')
    else if not TFile.Exists(LArgs) then
      DoInfo('File not found: ' + LArgs)
    else
    begin
      try
        LText := TFile.ReadAllText(LArgs);
        FSession.AddDocument(LArgs, LArgs, LText);
        DoInfo('Document added: ' + LArgs);
      except
        on E: Exception do
          DoError('Failed to read file: ' + E.Message);
      end;
    end;
  end
  else if LCmd = '/stats' then
  begin
    PrintStats();
  end
  else if LCmd = '/turns' then
  begin
    DoInfo(Format('Turn count: %d', [FSession.GetTurnCount()]));
  end
  else if LCmd = '/tokens' then
  begin
    LNewMax := StrToIntDef(LArgs, FMaxTokens);
    FMaxTokens := LNewMax;
    DoInfo(Format('Max tokens set to %d', [FMaxTokens]));
  end
  else if LCmd = '/help' then
  begin
    PrintHelp();
  end
  else
  begin
    // Fall through to derived class custom commands
    if not DoCommand(LCmd, LArgs) then
      DoInfo('Unknown command: ' + LCmd);
  end;
end;

procedure TVdxChat.ProcessChat(const AInput: string);
var
  LResponse: string;
  LStats: PVdxInferenceStats;
  LStopStr: string;
begin
  // Token output is streamed via the DoToken callback wired in Run()
  LResponse := FSession.Chat(AInput, FMaxTokens);
  DoGenerationComplete();

  // Diagnostic: report if generation produced nothing
  if LResponse = '' then
  begin
    LStats := FSession.GetStats();
    if LStats <> nil then
    begin
      case LStats^.StopReason of
        srNone:        LStopStr := 'None';
        srEOS:         LStopStr := 'EOS';
        srStopToken:   LStopStr := 'Stop token';
        srMaxTokens:   LStopStr := 'Max tokens';
        srContextFull: LStopStr := 'Context full';
        srCancelled:   LStopStr := 'Cancelled';
      else
        LStopStr := 'Unknown';
      end;
      DoInfo(Format('[DEBUG] Empty response. Stop: %s, Prefill: %d, Gen: %d, KVPos after: ???',
        [LStopStr, LStats^.PrefillTokens, LStats^.GeneratedTokens]));
    end
    else
      DoInfo('[DEBUG] Empty response. No stats available.');
  end;
end;

procedure TVdxChat.PrintHelp();
begin
  DoInfo('Available commands:');
  DoInfo('  /quit              Exit the chat');
  DoInfo('  /clear             Clear conversation history');
  DoInfo('  /system <text>     Set system prompt');
  DoInfo('  /addfact <text>    Add a fact to memory');
  DoInfo('  /addfile <path>    Add a document from file');
  DoInfo('  /stats             Show inference statistics');
  DoInfo('  /turns             Show turn count');
  DoInfo('  /tokens <n>        Set max generation tokens');
  DoInfo('  /help              Show this help');
end;

procedure TVdxChat.PrintStats();
var
  LStats: PVdxInferenceStats;
  LStopStr: string;
begin
  LStats := FSession.GetStats();
  if LStats = nil then
  begin
    DoInfo('No stats available yet.');
    Exit;
  end;

  case LStats^.StopReason of
    srNone:        LStopStr := 'None';
    srEOS:         LStopStr := 'EOS';
    srStopToken:   LStopStr := 'Stop token';
    srMaxTokens:   LStopStr := 'Max tokens';
    srContextFull: LStopStr := 'Context full';
    srCancelled:   LStopStr := 'Cancelled';
  else
    LStopStr := 'Unknown';
  end;

  DoInfo(Format('Prefill: %d tok, %.1f tok/s',
    [LStats^.PrefillTokens, LStats^.PrefillTokPerSec]));
  DoInfo(Format('Generation: %d tok, %.1f tok/s',
    [LStats^.GeneratedTokens, LStats^.GenerationTokPerSec]));
  DoInfo(Format('Time to first token: %.1f ms', [LStats^.TimeToFirstTokenMs]));
  DoInfo(Format('Stop reason: %s', [LStopStr]));
  DoInfo(Format('VRAM: %.1f MB weights, %.1f MB cache, %.1f MB total', [
    LStats^.VRAMUsage.WeightsBytes / (1024 * 1024),
    LStats^.VRAMUsage.CacheBytes / (1024 * 1024),
    LStats^.VRAMUsage.TotalBytes / (1024 * 1024)]));
end;

end.