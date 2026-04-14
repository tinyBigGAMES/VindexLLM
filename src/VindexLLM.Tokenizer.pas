{===============================================================================
  VindexLLM - Graph-Walk LLM Inference Engine

  Copyright (c) 2026-present tinyBigGAMES LLC
  All Rights Reserved.

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Tokenizer;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Generics.Collections,
  System.Generics.Defaults,
  VindexLLM.Utils,
  VindexLLM.GGUFReader;

type
  // Token type flags from GGUF (matches llama.cpp token types)
  TVdxTokenType = (
    ttNormal = 1,
    ttUnknown = 2,
    ttControl = 3,
    ttUserDefined = 4,
    ttUnused = 5,
    ttByte = 6
  );

  { TVdxTokenizer }
  TVdxTokenizer = class(TVdxErrorsObject)
  private
    FTokens: TArray<string>;         // Token strings indexed by ID
    FScores: TArray<Single>;         // BPE merge scores indexed by ID
    FTypes: TArray<Integer>;         // Token type flags indexed by ID
    FVocabSize: Integer;
    FBosId: Integer;
    FEosId: Integer;

    // Lookup: token string -> ID (for encoding)
    FTokenToId: TDictionary<string, Integer>;

    // Special/control tokens sorted by length descending (for greedy match)
    FSpecialTokens: TArray<TPair<string, Integer>>;

    // BPE merge: find best pair to merge in a token list
    function FindBestMerge(const APieces: TList<Integer>): Integer;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Load vocabulary from GGUF reader
    function LoadFromGGUF(const AReader: TVdxGGUFReader): Boolean;

    // Encode text to token IDs (adds BOS automatically)
    function Encode(const AText: string; const AAddBos: Boolean = True): TArray<Integer>;

    // Decode token IDs back to text
    function Decode(const AIds: TArray<Integer>): string;

    // Accessors
    function GetVocabSize(): Integer;
    function GetBosId(): Integer;
    function GetEosId(): Integer;
    function GetTokenStr(const AId: Integer): string;
  end;

implementation

uses
  System.Math;

// ============================================================================
//  TVdxTokenizer — Construction / Destruction
// ============================================================================

constructor TVdxTokenizer.Create();
begin
  inherited;
  FTokenToId := TDictionary<string, Integer>.Create();
  FVocabSize := 0;
  FBosId := 2;
  FEosId := 1;
end;

destructor TVdxTokenizer.Destroy();
begin
  FTokenToId.Free();
  inherited;
end;

// ============================================================================
//  TVdxTokenizer — Load vocabulary from GGUF
// ============================================================================

function TVdxTokenizer.LoadFromGGUF(const AReader: TVdxGGUFReader): Boolean;
var
  LVocab: TVdxGGUFMetaValue;
  LScoresVal: TVdxGGUFMetaValue;
  LTypesVal: TVdxGGUFMetaValue;
  LI: Integer;
  LSpecialList: TList<TPair<string, Integer>>;
  LTokenType: Integer;
begin
  Result := False;

  if not AReader.HasMetadata('tokenizer.ggml.tokens') then
    Exit;

  // Read token strings
  LVocab := AReader.GetMetadata('tokenizer.ggml.tokens');
  FVocabSize := Length(LVocab.ArrayItems);
  SetLength(FTokens, FVocabSize);
  for LI := 0 to FVocabSize - 1 do
    FTokens[LI] := LVocab.ArrayItems[LI].AsString;

  // Read scores
  SetLength(FScores, FVocabSize);
  if AReader.HasMetadata('tokenizer.ggml.scores') then
  begin
    LScoresVal := AReader.GetMetadata('tokenizer.ggml.scores');
    for LI := 0 to FVocabSize - 1 do
      FScores[LI] := LScoresVal.ArrayItems[LI].AsFloat64;
  end;

  // Read token types
  SetLength(FTypes, FVocabSize);
  if AReader.HasMetadata('tokenizer.ggml.token_type') then
  begin
    LTypesVal := AReader.GetMetadata('tokenizer.ggml.token_type');
    for LI := 0 to FVocabSize - 1 do
      FTypes[LI] := Integer(LTypesVal.ArrayItems[LI].AsInt64);
  end;

  // Build token-to-id lookup
  FTokenToId.Clear();
  for LI := 0 to FVocabSize - 1 do
    FTokenToId.AddOrSetValue(FTokens[LI], LI);

  // Build special token list (control + user-defined), sorted by length desc
  // so longer matches are tried first
  LSpecialList := TList<TPair<string, Integer>>.Create();
  try
    for LI := 0 to FVocabSize - 1 do
    begin
      LTokenType := FTypes[LI];
      if (LTokenType = Ord(ttControl)) or (LTokenType = Ord(ttUserDefined)) then
      begin
        if FTokens[LI] <> '' then
          LSpecialList.Add(TPair<string, Integer>.Create(FTokens[LI], LI));
      end;
    end;

    // Sort by string length descending (greedy match longest first)
    LSpecialList.Sort(TComparer<TPair<string, Integer>>.Construct(
      function(const ALeft: TPair<string, Integer>;
        const ARight: TPair<string, Integer>): Integer
      begin
        Result := Length(ARight.Key) - Length(ALeft.Key);
      end
    ));

    FSpecialTokens := LSpecialList.ToArray();
  finally
    LSpecialList.Free();
  end;

  // Read BOS/EOS IDs from metadata
  FBosId := Integer(AReader.GetMetadataUInt32('tokenizer.ggml.bos_token_id', 2));
  FEosId := Integer(AReader.GetMetadataUInt32('tokenizer.ggml.eos_token_id', 1));

  Result := True;
end;

// ============================================================================
//  TVdxTokenizer — BPE Merge: find best adjacent pair to merge
//  Returns index of the left element in the best pair, or -1 if no merge.
// ============================================================================

function TVdxTokenizer.FindBestMerge(const APieces: TList<Integer>): Integer;
var
  LI: Integer;
  LMergedStr: string;
  LMergedId: Integer;
  LBestIdx: Integer;
  LBestScore: Single;
  LScore: Single;
begin
  LBestIdx := -1;
  LBestScore := -1e30;

  for LI := 0 to APieces.Count - 2 do
  begin
    LMergedStr := FTokens[APieces[LI]] + FTokens[APieces[LI + 1]];
    if FTokenToId.TryGetValue(LMergedStr, LMergedId) then
    begin
      LScore := FScores[LMergedId];
      if LScore > LBestScore then
      begin
        LBestScore := LScore;
        LBestIdx := LI;
      end;
    end;
  end;

  Result := LBestIdx;
end;

// ============================================================================
//  TVdxTokenizer — Encode: text -> token IDs using BPE
// ============================================================================

function TVdxTokenizer.Encode(const AText: string;
  const AAddBos: Boolean): TArray<Integer>;
var
  LResult: TList<Integer>;
  LPos: Integer;
  LTextLen: Integer;
  LMatched: Boolean;
  LI: Integer;
  LSpecStr: string;
  LSpecLen: Integer;
  LSegEnd: Integer;
  LSegment: string;
  LNormalized: string;
  LCharStr: string;
  LCharId: Integer;
  LPieces: TList<Integer>;
  LMergeIdx: Integer;
  LMergedStr: string;
  LMergedId: Integer;
  LCharIdx: Integer;
  LBytes: TBytes;
  LByte: Byte;
  LByteToken: string;
begin
  LResult := TList<Integer>.Create();
  try
    if AAddBos then
      LResult.Add(FBosId);

    LPos := 1;  // Delphi strings are 1-based
    LTextLen := Length(AText);

    while LPos <= LTextLen do
    begin
      // Try to match a special token at current position
      LMatched := False;
      for LI := 0 to Length(FSpecialTokens) - 1 do
      begin
        LSpecStr := FSpecialTokens[LI].Key;
        LSpecLen := Length(LSpecStr);
        if (LPos + LSpecLen - 1 <= LTextLen) and
           (Copy(AText, LPos, LSpecLen) = LSpecStr) then
        begin
          LResult.Add(FSpecialTokens[LI].Value);
          LPos := LPos + LSpecLen;
          LMatched := True;
          Break;
        end;
      end;

      if LMatched then
        Continue;

      // Find extent of regular text (until next special token or end)
      LSegEnd := LPos + 1;
      while LSegEnd <= LTextLen do
      begin
        LMatched := False;
        for LI := 0 to Length(FSpecialTokens) - 1 do
        begin
          LSpecStr := FSpecialTokens[LI].Key;
          LSpecLen := Length(LSpecStr);
          if (LSegEnd + LSpecLen - 1 <= LTextLen) and
             (Copy(AText, LSegEnd, LSpecLen) = LSpecStr) then
          begin
            LMatched := True;
            Break;
          end;
        end;
        if LMatched then
          Break;
        Inc(LSegEnd);
      end;

      // Extract regular text segment
      LSegment := Copy(AText, LPos, LSegEnd - LPos);
      LPos := LSegEnd;

      // Normalize: replace spaces with ▁ (U+2581)
      LNormalized := LSegment.Replace(' ', #$2581);

      // Split into individual characters, map each to a token ID
      LPieces := TList<Integer>.Create();
      try
        LCharIdx := 1;
        while LCharIdx <= Length(LNormalized) do
        begin
          // Get one character (handle surrogate pairs for chars > U+FFFF)
          if (LCharIdx < Length(LNormalized)) and
             (Ord(LNormalized[LCharIdx]) >= $D800) and
             (Ord(LNormalized[LCharIdx]) <= $DBFF) and
             (Ord(LNormalized[LCharIdx + 1]) >= $DC00) and
             (Ord(LNormalized[LCharIdx + 1]) <= $DFFF) then
          begin
            LCharStr := Copy(LNormalized, LCharIdx, 2);
            Inc(LCharIdx, 2);
          end
          else
          begin
            LCharStr := LNormalized[LCharIdx];
            Inc(LCharIdx);
          end;

          // Look up single character in vocab
          if FTokenToId.TryGetValue(LCharStr, LCharId) then
          begin
            LPieces.Add(LCharId);
          end
          else
          begin
            // Byte fallback: encode character as UTF-8 bytes
            LBytes := TEncoding.UTF8.GetBytes(LCharStr);
            for LByte in LBytes do
            begin
              // Byte tokens in GGUF are stored as <0xHH>
              LByteToken := Format('<0x%s>', [IntToHex(LByte, 2)]);
              if FTokenToId.TryGetValue(LByteToken, LCharId) then
                LPieces.Add(LCharId);
            end;
          end;
        end;

        // BPE merge loop: repeatedly merge the best scoring pair
        LMergeIdx := FindBestMerge(LPieces);
        while LMergeIdx >= 0 do
        begin
          LMergedStr := FTokens[LPieces[LMergeIdx]] +
                        FTokens[LPieces[LMergeIdx + 1]];
          if FTokenToId.TryGetValue(LMergedStr, LMergedId) then
          begin
            LPieces[LMergeIdx] := LMergedId;
            LPieces.Delete(LMergeIdx + 1);
          end
          else
            Break;  // Should not happen, but safety exit
          LMergeIdx := FindBestMerge(LPieces);
        end;

        // Add merged pieces to result
        for LI := 0 to LPieces.Count - 1 do
          LResult.Add(LPieces[LI]);
      finally
        LPieces.Free();
      end;
    end;

    Result := LResult.ToArray();
  finally
    LResult.Free();
  end;
end;

// ============================================================================
//  TVdxTokenizer — Decode: token IDs -> text
// ============================================================================

function TVdxTokenizer.Decode(const AIds: TArray<Integer>): string;
var
  LI: Integer;
  LId: Integer;
  LToken: string;
begin
  Result := '';
  for LI := 0 to Length(AIds) - 1 do
  begin
    LId := AIds[LI];
    if (LId >= 0) and (LId < FVocabSize) then
    begin
      LToken := FTokens[LId];
      // Replace ▁ back to space for display
      LToken := LToken.Replace(#$2581, ' ');
      Result := Result + LToken;
    end;
  end;
end;

// ============================================================================
//  TVdxTokenizer — Accessors
// ============================================================================

function TVdxTokenizer.GetVocabSize(): Integer;
begin
  Result := FVocabSize;
end;

function TVdxTokenizer.GetBosId(): Integer;
begin
  Result := FBosId;
end;

function TVdxTokenizer.GetEosId(): Integer;
begin
  Result := FEosId;
end;

function TVdxTokenizer.GetTokenStr(const AId: Integer): string;
begin
  if (AId >= 0) and (AId < FVocabSize) then
    Result := FTokens[AId]
  else
    Result := '<invalid>';
end;

end.
