{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.Model.Gemma3;

interface

uses
  VindexLLM.TestCase;

type

  { TModelGemma3Test }
  // End-to-end coverage of TVdxGemma3Model via the TVdxModel factory.
  // Loads the real Gemma 3 4B IT F16 GGUF from the hardcoded model
  // path — if that file is absent the positive sections fail, which
  // is the intended signal.
  TModelGemma3Test = class(TVdxTestCase)
  private
    procedure SecSupportedArchitectures();
    procedure SecLoadModelMissingPath();
    procedure SecLoadModelConfig();
    procedure SecLoadModelSubsystems();
  protected
    procedure Run(); override;
  public
    constructor Create(); override;
  end;

implementation

uses
  System.SysUtils,
  VindexLLM.Utils,
  VindexLLM.Model,
  VindexLLM.Model.Gemma3,
  UTest.Common;

const
  CMaxContext = 1024;  // keep batch matrices small for the test

{ TModelGemma3Test }

constructor TModelGemma3Test.Create();
begin
  inherited;
  Title := 'Test_Model.Gemma3';
end;

procedure TModelGemma3Test.Run();
begin
  SecSupportedArchitectures();
  SecLoadModelMissingPath();
  SecLoadModelConfig();
  SecLoadModelSubsystems();
end;

// ---------------------------------------------------------------------------
// 1. SupportedArchitectures — class-level check, no file needed.
// Confirms the registry sees both strings Gemma3 claims.
// ---------------------------------------------------------------------------
procedure TModelGemma3Test.SecSupportedArchitectures();
var
  LArches: TArray<string>;
  LFoundGemma3, LFoundEmbedding: Boolean;
  LI: Integer;
begin
  Section('SupportedArchitectures returns [gemma3, gemma-embedding]');
  LArches := TVdxGemma3Model.SupportedArchitectures();

  Check(Length(LArches) = 2,
    'Two architectures claimed');

  LFoundGemma3    := False;
  LFoundEmbedding := False;
  for LI := 0 to Length(LArches) - 1 do
  begin
    if LArches[LI] = 'gemma3' then LFoundGemma3 := True;
    if LArches[LI] = 'gemma-embedding' then LFoundEmbedding := True;
  end;
  Check(LFoundGemma3,
    '"gemma3" present in claim list');
  Check(LFoundEmbedding,
    '"gemma-embedding" present in claim list');
end;

// ---------------------------------------------------------------------------
// 2. Missing path — factory returns nil without raising. Can't check
// error details (no self instance to carry them on a nil return),
// but the nil result itself is the contract.
// ---------------------------------------------------------------------------
procedure TModelGemma3Test.SecLoadModelMissingPath();
var
  LModel: TVdxModel;
begin
  Section('LoadModel on missing path returns nil');
  LModel := TVdxModel.LoadModel('does_not_exist.gguf', 1024);
  try
    Check(LModel = nil,
      'LoadModel returns nil for missing path');
  finally
    LModel.Free();  // nil-safe
  end;
end;

// ---------------------------------------------------------------------------
// 3. Config load — architecture resolves to 'gemma3', dimensions
// match the Gemma 3 4B spec. Task spec pins two values: NumLayers=34,
// HeadDim=256. Other dimensions checked loosely (> 0) since they
// vary by parameter count (4B vs 270M vs 12B etc.).
// ---------------------------------------------------------------------------
procedure TModelGemma3Test.SecLoadModelConfig();
var
  LModel: TVdxModel;
  LStatusCb: TVdxStatusCallback;
begin
  Section('LoadModel populates Gemma 3 4B config correctly');

  // Temporary status pump — the factory frees its partial instance
  // on failure, so errors vanish with it. Status callbacks survive
  // and show us exactly how far LoadModelConfig / InitSubsystems /
  // LoadWeights got before bailing.
  LStatusCb := procedure(const AText: string; const AUserData: Pointer)
  begin
    TVdxUtils.PrintLn('  [status] ' + AText);
  end;

  LModel := TVdxModel.LoadModel(CModelPath, CMaxContext, LStatusCb, nil);
  try
    Check(LModel <> nil,
      'LoadModel returns non-nil instance');
    if LModel = nil then Exit;

    // Factory now returns the partial instance on load-time failure
    // so the caller can inspect FErrors. If we're here, LoadModel
    // came back non-nil but may still have fatal errors populated —
    // print them before asserting no-fatal so the test output is
    // diagnostic rather than just "FAIL".
    if LModel.GetErrors().HasFatal() then
      PrintErrors(LModel.GetErrors());
    Check(not LModel.GetErrors().HasFatal(),
      'No fatal errors after successful load');

    Check(LModel is TVdxGemma3Model,
      'Factory resolved to TVdxGemma3Model');
    Check(LModel.Architecture = 'gemma3',
      'Architecture = "gemma3"');
    Check(LModel.NumLayers = 34,
      'NumLayers = 34 (Gemma 3 4B)');
    Check(LModel.HeadDim = 256,
      'HeadDim = 256 (Gemma 3 4B)');
    Check(LModel.HiddenDim > 0,
      'HiddenDim populated');
    Check(LModel.FFNWidth > 0,
      'FFNWidth populated');
    Check(LModel.NumQHeads > 0,
      'NumQHeads populated');
    Check(LModel.NumKVHeads > 0,
      'NumKVHeads populated');
    Check(LModel.VocabSize > 0,
      'VocabSize populated via tokenizer');
    Check(LModel.MaxSeqLen <= UInt32(CMaxContext),
      'MaxSeqLen clamped to requested max');
    Check(LModel.MaxSeqLen > 0,
      'MaxSeqLen non-zero');

    FlushErrors(LModel.GetErrors());
  finally
    LModel.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 4. Subsystems live — all five owned subsystems exist, factory
// clean teardown works without fatal. Spot-checks Gemma-3-specific
// virtuals (RoPE mod-6, chat template, embedding support).
// ---------------------------------------------------------------------------
procedure TModelGemma3Test.SecLoadModelSubsystems();
var
  LModel: TVdxModel;
  LStopTokens: TArray<string>;
  LPrompt: string;
begin
  Section('LoadModel wires subsystems and Gemma 3 template overrides');
  LModel := TVdxModel.LoadModel(CModelPath, CMaxContext);
  try
    Check(LModel <> nil,
      'LoadModel returns non-nil instance');
    if LModel = nil then Exit;

    // Subsystems — all created + initialized by InitSubsystems,
    // reachable through the accessors.
    Check(LModel.Compute <> nil,
      'Compute subsystem exposed');
    Check(LModel.Norm <> nil,
      'Norm subsystem exposed');
    Check(LModel.Attn <> nil,
      'Attn subsystem exposed');
    Check(LModel.FFN <> nil,
      'FFN subsystem exposed');
    Check(LModel.Tokenizer <> nil,
      'Tokenizer subsystem exposed');

    // Gemma 3 virtual overrides.
    Check(LModel.SupportsEmbedding(),
      'SupportsEmbedding returns True');
    Check(LModel.GetRoPETheta(0) = 10000.0,
      'RoPE theta = 10K at layer 0 (sliding)');
    Check(LModel.GetRoPETheta(5) = 1000000.0,
      'RoPE theta = 1M at layer 5 (global, mod-6 = 5)');
    Check(LModel.GetRoPETheta(11) = 1000000.0,
      'RoPE theta = 1M at layer 11 (global, mod-6 = 5)');
    Check(LModel.GetRoPETheta(12) = 10000.0,
      'RoPE theta = 10K at layer 12 (sliding, mod-6 = 0)');

    LStopTokens := LModel.GetStopTokenStrings();
    Check(Length(LStopTokens) = 1,
      'Exactly one stop token string');
    if Length(LStopTokens) >= 1 then
      Check(LStopTokens[0] = '<end_of_turn>',
        'Stop token is <end_of_turn>');

    // Chat template wraps the prompt with the expected markers.
    LPrompt := LModel.FormatPrompt('hello');
    Check(Pos('<start_of_turn>user', LPrompt) > 0,
      'FormatPrompt emits user turn marker');
    Check(Pos('<start_of_turn>model', LPrompt) > 0,
      'FormatPrompt leaves model turn open');
    Check(Pos('hello', LPrompt) > 0,
      'FormatPrompt embeds the prompt text');

    // Embedding prefixes differ between query and document side.
    Check(Pos('query:', LModel.FormatEmbedding('x', True)) > 0,
      'FormatEmbedding(IsQuery=True) includes query prefix');
    Check(Pos('text:', LModel.FormatEmbedding('x', False)) > 0,
      'FormatEmbedding(IsQuery=False) includes text prefix');

    FlushErrors(LModel.GetErrors());
  finally
    LModel.Free();
  end;

  // Destructor order + nil-safety is implicit — if anything leaked
  // GPU handles or double-freed, the next Section or the testbed exit
  // would surface it. Nothing more to assert here.
end;

end.
