{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.Tokenizer;

interface

uses
  VindexLLM.TestCase;

type

  { TTokenizerTest }
  // Section-based coverage of TVdxTokenizer. Positive sections depend
  // on the Gemma 3 4B IT F16 GGUF at the hardcoded model path — same
  // fixture UTest.GGUFReader uses. If that file is absent the positive
  // sections fail with TK02/TK03, which is the intended signal.
  //
  // Correctness strategy follows "Option C" from the Phase 5 plan:
  // structural assertions plus three semantic checks (special-token
  // greedy match, BPE actually merges, BOS/EOS strings look right).
  // No hardcoded token IDs — so the test survives any Gemma vocab
  // revision without becoming a liability.
  TTokenizerTest = class(TVdxTestCase)
  private
    procedure SecInitialState();
    procedure SecLoadNilReader();
    procedure SecLoadReaderNotOpen();
    procedure SecLoadValidModel();
    procedure SecBosEosAccessors();
    procedure SecEncodeVsEncodeWithBos();
    procedure SecRoundTrip();
    procedure SecGetTokenStrBounds();
    procedure SecSpecialTokenGreedy();
    procedure SecBpeActuallyMerges();
    procedure SecBosEosStringsLookRight();
  protected
    procedure Run(); override;
  public
    constructor Create(); override;
  end;

implementation

uses
  System.SysUtils,
  System.Generics.Collections,
  VindexLLM.Utils,
  VindexLLM.GGUFReader,
  VindexLLM.Tokenizer;

const
  CModelPath = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';

{ TTokenizerTest }

constructor TTokenizerTest.Create();
begin
  inherited;
  Title := 'Test_Tokenizer';
end;

procedure TTokenizerTest.Run();
begin
  SecInitialState();
  SecLoadNilReader();
  SecLoadReaderNotOpen();
  SecLoadValidModel();
  SecBosEosAccessors();
  SecEncodeVsEncodeWithBos();
  SecRoundTrip();
  SecGetTokenStrBounds();
  SecSpecialTokenGreedy();
  SecBpeActuallyMerges();
  SecBosEosStringsLookRight();
end;


// ---------------------------------------------------------------------------
// 1. Create() initial state — no errors, vocab empty, not loaded.
// ---------------------------------------------------------------------------
procedure TTokenizerTest.SecInitialState();
var
  LTok: TVdxTokenizer;
begin
  Section('Create() initial state');
  LTok := TVdxTokenizer.Create();
  try
    Check(not LTok.GetErrors().HasFatal(),
      'No fatal errors after Create');
    Check(LTok.GetVocabSize() = 0,
      'VocabSize is 0 before LoadFromReader');
    Check(not LTok.IsLoaded(),
      'IsLoaded is False before LoadFromReader');
    FlushErrors(LTok.GetErrors());
  finally
    LTok.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 2. LoadFromReader(nil) — guard 1 fires, logs TK01 fatal.
// ---------------------------------------------------------------------------
procedure TTokenizerTest.SecLoadNilReader();
var
  LTok: TVdxTokenizer;
  LItems: TList<TVdxError>;
  LFoundCode: Boolean;
  LI: Integer;
begin
  Section('LoadFromReader(nil) - logs TK01 fatal');
  LTok := TVdxTokenizer.Create();
  try
    Check(not LTok.LoadFromReader(nil),
      'LoadFromReader(nil) returns False');
    Check(LTok.GetErrors().HasFatal(),
      'HasFatal is True');
    Check(not LTok.IsLoaded(),
      'IsLoaded stays False on failure');

    LItems := LTok.GetErrors().GetItems();
    LFoundCode := False;
    for LI := 0 to LItems.Count - 1 do
      if LItems[LI].Code = VDX_ERROR_TK_NIL_READER then
      begin
        LFoundCode := True;
        Break;
      end;
    Check(LFoundCode, 'Error list contains TK01 (NIL_READER)');
    FlushErrors(LTok.GetErrors());
  finally
    LTok.Free();
  end;
end;


// ---------------------------------------------------------------------------
// 3. LoadFromReader on a reader that was never Open'd — guard 2 fires,
// logs TK02 fatal. Confirms IsOpen()-based detection works before the
// metadata lookups.
// ---------------------------------------------------------------------------
procedure TTokenizerTest.SecLoadReaderNotOpen();
var
  LReader: TVdxGGUFReader;
  LTok: TVdxTokenizer;
  LItems: TList<TVdxError>;
  LFoundCode: Boolean;
  LI: Integer;
begin
  Section('LoadFromReader on unopened reader - logs TK02 fatal');
  LReader := TVdxGGUFReader.Create();
  try
    // Deliberately do NOT call Open — reader is live but memory-map
    // is nil, so IsOpen() returns False.
    LTok := TVdxTokenizer.Create();
    try
      Check(not LReader.IsOpen(),
        'Precondition: reader.IsOpen() is False');
      Check(not LTok.LoadFromReader(LReader),
        'LoadFromReader on unopened reader returns False');
      Check(LTok.GetErrors().HasFatal(),
        'HasFatal is True');

      LItems := LTok.GetErrors().GetItems();
      LFoundCode := False;
      for LI := 0 to LItems.Count - 1 do
        if LItems[LI].Code = VDX_ERROR_TK_READER_NOT_OPEN then
        begin
          LFoundCode := True;
          Break;
        end;
      Check(LFoundCode, 'Error list contains TK02 (READER_NOT_OPEN)');
      FlushErrors(LTok.GetErrors());
    finally
      LTok.Free();
    end;
  finally
    LReader.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 4. LoadFromReader against the real Gemma 3 4B GGUF. If the model is
// absent on disk this fails — same intended signal as the GGUFReader
// test. Success drives every subsequent positive section.
// ---------------------------------------------------------------------------
procedure TTokenizerTest.SecLoadValidModel();
var
  LReader: TVdxGGUFReader;
  LTok: TVdxTokenizer;
begin
  Section('LoadFromReader on valid Gemma 3 4B F16 GGUF');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      LTok := TVdxTokenizer.Create();
      try
        Check(LTok.LoadFromReader(LReader),
          'LoadFromReader returns True on real GGUF');
        Check(not LTok.GetErrors().HasFatal(),
          'No fatal errors after successful load');
        Check(LTok.IsLoaded(),
          'IsLoaded flipped to True after success');
        Check(LTok.GetVocabSize() > 0,
          'VocabSize is > 0 after load');
        FlushErrors(LTok.GetErrors());
      finally
        LTok.Free();
      end;
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;


// ---------------------------------------------------------------------------
// 5. BOS / EOS accessor sanity — both IDs in range, distinct. No
// hardcoded expected values: Gemma 3 ships with BOS=2 / EOS=1 today
// but we only assert structural correctness so a future vocab swap
// doesn't break the test.
// ---------------------------------------------------------------------------
procedure TTokenizerTest.SecBosEosAccessors();
var
  LReader: TVdxGGUFReader;
  LTok: TVdxTokenizer;
  LBos: Integer;
  LEos: Integer;
  LVocab: Integer;
begin
  Section('BOS / EOS accessors are structurally sane');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      LTok := TVdxTokenizer.Create();
      try
        if LTok.LoadFromReader(LReader) then
        begin
          LBos := LTok.GetBosId();
          LEos := LTok.GetEosId();
          LVocab := LTok.GetVocabSize();

          Check(LBos >= 0, 'BosId >= 0');
          Check(LEos >= 0, 'EosId >= 0');
          Check(LBos < LVocab, 'BosId < VocabSize');
          Check(LEos < LVocab, 'EosId < VocabSize');
          Check(LBos <> LEos, 'BosId <> EosId');
        end;
        FlushErrors(LTok.GetErrors());
      finally
        LTok.Free();
      end;
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 6. Encode vs EncodeWithBos — same input, the BOS variant has
// exactly one more token at index 0 equal to BosId.
// ---------------------------------------------------------------------------
procedure TTokenizerTest.SecEncodeVsEncodeWithBos();
var
  LReader: TVdxGGUFReader;
  LTok: TVdxTokenizer;
  LIds: TArray<Integer>;
  LIdsBos: TArray<Integer>;
  LI: Integer;
  LTailMatches: Boolean;
begin
  Section('Encode vs EncodeWithBos - BOS prepended correctly');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      LTok := TVdxTokenizer.Create();
      try
        if LTok.LoadFromReader(LReader) then
        begin
          LIds    := LTok.Encode('Hello world');
          LIdsBos := LTok.EncodeWithBos('Hello world');

          Check(Length(LIdsBos) = Length(LIds) + 1,
            'EncodeWithBos returns exactly one extra token');
          Check((Length(LIdsBos) > 0) and (LIdsBos[0] = LTok.GetBosId()),
            'EncodeWithBos[0] equals BosId');

          // Tail of EncodeWithBos should match Encode element-for-element.
          LTailMatches := True;
          if Length(LIds) = Length(LIdsBos) - 1 then
          begin
            for LI := 0 to Length(LIds) - 1 do
              if LIds[LI] <> LIdsBos[LI + 1] then
              begin
                LTailMatches := False;
                Break;
              end;
          end
          else
            LTailMatches := False;
          Check(LTailMatches,
            'EncodeWithBos tail matches Encode element-for-element');
        end;
        FlushErrors(LTok.GetErrors());
      finally
        LTok.Free();
      end;
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;


// ---------------------------------------------------------------------------
// 7. Round-trip Decode(Encode(S)) = S. Exercises U+2581 whitespace
// normalization restoration and confirms the vocab table is coherent.
// Two strings — one with an ASCII-only payload, one with punctuation.
// ---------------------------------------------------------------------------
procedure TTokenizerTest.SecRoundTrip();
var
  LReader: TVdxGGUFReader;
  LTok: TVdxTokenizer;
  LInput1: string;
  LInput2: string;
  LRound1: string;
  LRound2: string;
begin
  Section('Round-trip Decode(Encode(S)) = S');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      LTok := TVdxTokenizer.Create();
      try
        if LTok.LoadFromReader(LReader) then
        begin
          LInput1 := 'Hello world';
          LInput2 := 'What is the capital of France?';

          LRound1 := LTok.Decode(LTok.Encode(LInput1));
          LRound2 := LTok.Decode(LTok.Encode(LInput2));

          Check(LRound1 = LInput1,
            'Round-trip preserves "Hello world"');
          Check(LRound2 = LInput2,
            'Round-trip preserves "What is the capital of France?"');
        end;
        FlushErrors(LTok.GetErrors());
      finally
        LTok.Free();
      end;
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 8. GetTokenStr bounds — valid IDs return a populated string, out-of-
// range IDs return the '<invalid>' sentinel (never crash, never throw).
// ---------------------------------------------------------------------------
procedure TTokenizerTest.SecGetTokenStrBounds();
var
  LReader: TVdxGGUFReader;
  LTok: TVdxTokenizer;
  LStr: string;
begin
  Section('GetTokenStr bounds - sentinel for out-of-range IDs');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      LTok := TVdxTokenizer.Create();
      try
        if LTok.LoadFromReader(LReader) then
        begin
          // Valid ID — BosId is guaranteed to be in range by
          // SecBosEosAccessors, reuse it here.
          LStr := LTok.GetTokenStr(LTok.GetBosId());
          Check(LStr <> '',
            'GetTokenStr(BosId) returns a non-empty string');
          Check(LStr <> '<invalid>',
            'GetTokenStr(BosId) is not the invalid sentinel');

          Check(LTok.GetTokenStr(-1) = '<invalid>',
            'GetTokenStr(-1) returns <invalid>');
          Check(LTok.GetTokenStr(LTok.GetVocabSize() + 100)
              = '<invalid>',
            'GetTokenStr(VocabSize + 100) returns <invalid>');
        end;
        FlushErrors(LTok.GetErrors());
      finally
        LTok.Free();
      end;
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;


// ---------------------------------------------------------------------------
// 9. Option C semantic check #1 — special-token greedy match.
// Encoding '<start_of_turn>' (a single Gemma control token) must
// produce exactly one token whose string is '<start_of_turn>'. A
// broken special-token table would BPE-split it character-by-character
// into 15 or more tokens.
// ---------------------------------------------------------------------------
procedure TTokenizerTest.SecSpecialTokenGreedy();
var
  LReader: TVdxGGUFReader;
  LTok: TVdxTokenizer;
  LIds: TArray<Integer>;
begin
  Section('Special-token greedy match - <start_of_turn> = 1 token');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      LTok := TVdxTokenizer.Create();
      try
        if LTok.LoadFromReader(LReader) then
        begin
          LIds := LTok.Encode('<start_of_turn>');
          Check(Length(LIds) = 1,
            'Encode(<start_of_turn>) returns exactly 1 token');
          if Length(LIds) = 1 then
            Check(LTok.GetTokenStr(LIds[0]) = '<start_of_turn>',
              'That token''s string round-trips to <start_of_turn>');
        end;
        FlushErrors(LTok.GetErrors());
      finally
        LTok.Free();
      end;
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 10. Option C semantic check #2 — BPE actually merges. 'Hello world'
// has 11 characters; a working BPE merger produces far fewer tokens.
// If FindBestMerge silently returned -1 every call, encoding would
// degenerate to one-token-per-character (12 tokens counting the
// U+2581 space marker). Asserting < 11 catches that degenerate case
// without hardcoding an exact count that would couple to vocab.
// ---------------------------------------------------------------------------
procedure TTokenizerTest.SecBpeActuallyMerges();
var
  LReader: TVdxGGUFReader;
  LTok: TVdxTokenizer;
  LIds: TArray<Integer>;
begin
  Section('BPE actually merges - Encode("Hello world") count < 11');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      LTok := TVdxTokenizer.Create();
      try
        if LTok.LoadFromReader(LReader) then
        begin
          LIds := LTok.Encode('Hello world');
          Check(Length(LIds) > 0,
            'Encode produced at least one token');
          Check(Length(LIds) < 11,
            'Encode produced fewer tokens than characters (BPE ran)');
        end;
        FlushErrors(LTok.GetErrors());
      finally
        LTok.Free();
      end;
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 11. Option C semantic check #3 — BOS / EOS strings look right.
// Gemma 3 uses '<bos>' and '<eos>' literally. Case-insensitive
// substring check is robust enough to survive a vocab that decorated
// them (e.g. '<|bos|>') without false positives on arbitrary tokens.
// ---------------------------------------------------------------------------
procedure TTokenizerTest.SecBosEosStringsLookRight();
var
  LReader: TVdxGGUFReader;
  LTok: TVdxTokenizer;
  LBosStr: string;
  LEosStr: string;
begin
  Section('BOS / EOS token strings look right');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      LTok := TVdxTokenizer.Create();
      try
        if LTok.LoadFromReader(LReader) then
        begin
          LBosStr := LowerCase(LTok.GetTokenStr(LTok.GetBosId()));
          LEosStr := LowerCase(LTok.GetTokenStr(LTok.GetEosId()));

          Check(Pos('bos', LBosStr) > 0,
            'GetTokenStr(BosId) contains "bos" (case-insensitive)');
          Check(Pos('eos', LEosStr) > 0,
            'GetTokenStr(EosId) contains "eos" (case-insensitive)');
        end;
        FlushErrors(LTok.GetErrors());
      finally
        LTok.Free();
      end;
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

end.
