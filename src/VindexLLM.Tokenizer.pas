{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

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

const
  // Error codes — user-facing messages live in VindexLLM.Resources.
  VDX_ERROR_TK_NIL_READER       = 'TK01';
  VDX_ERROR_TK_READER_NOT_OPEN  = 'TK02';
  VDX_ERROR_TK_MISSING_TOKENS   = 'TK03';
  VDX_ERROR_TK_TOKENS_NOT_ARRAY = 'TK04';
  VDX_ERROR_TK_MISSING_SCORES   = 'TK05';
  VDX_ERROR_TK_MISSING_TYPES    = 'TK06';

type

  { TVdxTokenType }
  // GGUF token-type tag. Values follow the llama.cpp / GGUF convention;
  // `ttControl` and `ttUserDefined` are the two categories the tokenizer
  // treats as "special tokens" (greedy-matched before BPE runs).
  TVdxTokenType = (
    ttNormal      = 1,
    ttUnknown     = 2,
    ttControl     = 3,
    ttUserDefined = 4,
    ttUnused      = 5,
    ttByte        = 6
  );

  { TVdxTokenizer }
  // BPE tokenizer backed by a GGUF vocabulary. LoadFromReader parses
  // vocab / scores / token_type from an opened TVdxGGUFReader and
  // builds the reverse token-to-id dictionary plus the special-token
  // greedy-match table. Encode / EncodeWithBos produce token-ID
  // sequences; Decode is the inverse. Failures populate FErrors via
  // the TVdxBaseObject pattern — no raises propagate out.
  TVdxTokenizer = class(TVdxBaseObject)
  private
    FTokens: TArray<string>;      // Token strings indexed by ID
    FScores: TArray<Single>;      // BPE merge scores indexed by ID
    FTypes:  TArray<Integer>;     // TVdxTokenType-ordinal per ID
    FVocabSize: Integer;
    FBosId: Integer;
    FEosId: Integer;
    FLoaded: Boolean;

    // Token-string -> ID for encode-time lookup.
    FTokenToId: TDictionary<string, Integer>;

    // Control + user-defined tokens, sorted by string length descending
    // so Encode's greedy match tries the longest match first.
    FSpecialTokens: TArray<TPair<string, Integer>>;

    // Returns the index in APieces of the highest-scoring adjacent
    // mergeable pair, or -1 if no adjacent pair forms a known token.
    function FindBestMerge(const APieces: TList<Integer>): Integer;

    // Shared body for Encode / EncodeWithBos — AAddBos controls
    // whether FBosId is prepended to the result.
    function EncodeInternal(const AText: string;
      const AAddBos: Boolean): TArray<Integer>;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Parses the GGUF tokenizer metadata out of an already-opened
    // reader. Returns False on any fatal condition (nil reader,
    // reader not open, missing/malformed tokens / scores / types);
    // errors land in FErrors. Success populates the full vocab and
    // flips IsLoaded to True.
    function LoadFromReader(const AReader: TVdxGGUFReader): Boolean;

    // Tokenizes AText to IDs. Encode omits BOS; EncodeWithBos
    // prepends FBosId as the first element. Splitting the BOS case
    // into its own method keeps call sites self-documenting and
    // leaves room for future variants (EncodeWithEos, EncodePair,
    // etc.) without boolean-flag soup.
    function Encode(const AText: string): TArray<Integer>;
    function EncodeWithBos(const AText: string): TArray<Integer>;

    // Concatenates token strings for AIds, restoring U+2581 back to
    // ASCII space. Out-of-range IDs are skipped silently — Decode
    // is diagnostic, not validating.
    function Decode(const AIds: TArray<Integer>): string;

    function GetVocabSize(): Integer;
    function GetBosId(): Integer;
    function GetEosId(): Integer;

    // Returns FTokens[AId] for valid IDs, '<invalid>' otherwise.
    function GetTokenStr(const AId: Integer): string;

    // True after a successful LoadFromReader. Stays False if any
    // fatal condition fired during load — callers can gate Encode
    // / Decode on this.
    function IsLoaded(): Boolean;
  end;

implementation

uses
  VindexLLM.Resources;


{ TVdxTokenizer }

constructor TVdxTokenizer.Create();
begin
  inherited Create();
  FTokenToId := TDictionary<string, Integer>.Create();
  FVocabSize := 0;
  // Common Gemma-family defaults — overwritten from GGUF metadata in
  // LoadFromReader if the model file specifies anything different.
  FBosId := 2;
  FEosId := 1;
  FLoaded := False;
end;

destructor TVdxTokenizer.Destroy();
begin
  FreeAndNil(FTokenToId);
  inherited Destroy();
end;

function TVdxTokenizer.LoadFromReader(
  const AReader: TVdxGGUFReader): Boolean;
var
  LVocab: TVdxGGUFMetaValue;
  LScoresVal: TVdxGGUFMetaValue;
  LTypesVal: TVdxGGUFMetaValue;
  LI: Integer;
  LSpecialList: TList<TPair<string, Integer>>;
  LTokenType: Integer;
begin
  Result := False;
  FLoaded := False;

  // Guard 1: reader object itself.
  if AReader = nil then
  begin
    FErrors.Add(esFatal, VDX_ERROR_TK_NIL_READER, RSTKNilReader);
    Exit;
  end;

  // Guard 2: reader's memory-map must be live. Without this the
  // GetMetadata calls below would return False on every key and
  // produce a misleading "missing tokens" error instead of the
  // actual "you forgot to call Open" diagnosis.
  if not AReader.IsOpen() then
  begin
    FErrors.Add(esFatal, VDX_ERROR_TK_READER_NOT_OPEN,
      RSTKReaderNotOpen);
    Exit;
  end;

  // Guard 3: vocab array must be present.
  if not AReader.GetMetadata('tokenizer.ggml.tokens', LVocab) then
  begin
    FErrors.Add(esFatal, VDX_ERROR_TK_MISSING_TOKENS,
      RSTKMissingTokens);
    Exit;
  end;

  // Guard 4: vocab value must actually be an array. A corrupted /
  // non-standard file could have the key present but typed as
  // anything.
  if LVocab.ValueType <> gmtArray then
  begin
    FErrors.Add(esFatal, VDX_ERROR_TK_TOKENS_NOT_ARRAY,
      RSTKTokensNotArray, [Ord(LVocab.ValueType)]);
    Exit;
  end;

  FVocabSize := Length(LVocab.ArrayItems);
  SetLength(FTokens, FVocabSize);
  for LI := 0 to FVocabSize - 1 do
    FTokens[LI] := LVocab.ArrayItems[LI].AsString;

  // Guard 5: scores are mandatory — see design discussion.
  // Without scores FindBestMerge picks index 0 every iteration and
  // the tokenizer produces semantically wrong output silently.
  if not AReader.GetMetadata('tokenizer.ggml.scores', LScoresVal) then
  begin
    FErrors.Add(esFatal, VDX_ERROR_TK_MISSING_SCORES,
      RSTKMissingScores);
    Exit;
  end;

  SetLength(FScores, FVocabSize);
  for LI := 0 to FVocabSize - 1 do
    FScores[LI] := Single(LScoresVal.ArrayItems[LI].AsFloat64);

  // Guard 6: token types are mandatory — without them we can't
  // identify which entries are control / user-defined special
  // tokens, so chat templates ('<start_of_turn>' etc.) get BPE-split
  // character-by-character and prompt formatting breaks.
  if not AReader.GetMetadata(
    'tokenizer.ggml.token_type', LTypesVal) then
  begin
    FErrors.Add(esFatal, VDX_ERROR_TK_MISSING_TYPES,
      RSTKMissingTypes);
    Exit;
  end;

  SetLength(FTypes, FVocabSize);
  for LI := 0 to FVocabSize - 1 do
    FTypes[LI] := Integer(LTypesVal.ArrayItems[LI].AsInt64);

  // Build token-string -> ID reverse lookup for Encode.
  FTokenToId.Clear();
  for LI := 0 to FVocabSize - 1 do
    FTokenToId.AddOrSetValue(FTokens[LI], LI);

  // Build the special-token greedy-match table: every control or
  // user-defined token, sorted by string length descending so the
  // Encode loop matches longer sequences first (e.g.
  // '<start_of_turn>' before any single-character prefix of it).
  LSpecialList := TList<TPair<string, Integer>>.Create();
  try
    for LI := 0 to FVocabSize - 1 do
    begin
      LTokenType := FTypes[LI];
      if (LTokenType = Ord(ttControl)) or
         (LTokenType = Ord(ttUserDefined)) then
      begin
        if FTokens[LI] <> '' then
          LSpecialList.Add(
            TPair<string, Integer>.Create(FTokens[LI], LI));
      end;
    end;

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

  // BOS / EOS IDs from metadata, with sane Gemma defaults if absent.
  // GetMetadataUInt32 returns the default silently on miss — missing
  // BOS/EOS is not fatal because the defaults are correct for Gemma
  // and most Llama-family GGUFs.
  FBosId := Integer(AReader.GetMetadataUInt32(
    'tokenizer.ggml.bos_token_id', 2));
  FEosId := Integer(AReader.GetMetadataUInt32(
    'tokenizer.ggml.eos_token_id', 1));

  Status('Tokenizer loaded: %d tokens, BOS=%d, EOS=%d, %d specials',
    [FVocabSize, FBosId, FEosId, Length(FSpecialTokens)]);

  FLoaded := True;
  Result := True;
end;


function TVdxTokenizer.FindBestMerge(
  const APieces: TList<Integer>): Integer;
var
  LI: Integer;
  LMergedStr: string;
  LMergedId: Integer;
  LBestIdx: Integer;
  LBestScore: Single;
  LScore: Single;
begin
  // Scan every adjacent pair and pick the one whose concatenation
  // matches a known vocab entry with the highest merge score.
  // -1e30 is the sentinel for "no candidate seen yet"; any real
  // score exceeds it.
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

function TVdxTokenizer.EncodeInternal(const AText: string;
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

    // Delphi strings are 1-based — LPos walks AText from 1..Length.
    LPos := 1;
    LTextLen := Length(AText);

    while LPos <= LTextLen do
    begin
      // --- Phase 1: greedy special-token match at current position.
      // FSpecialTokens is pre-sorted longest-first so the first hit
      // wins.
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

      // --- Phase 2: find the extent of the next non-special run.
      // Walk forward until the next position where a special token
      // would match. Everything up to that point is BPE territory.
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

      LSegment := Copy(AText, LPos, LSegEnd - LPos);
      LPos := LSegEnd;

      // Sentencepiece / GGUF convention: spaces are encoded as the
      // U+2581 "▁" marker before vocab lookup. Decode reverses this.
      LNormalized := LSegment.Replace(' ', #$2581);

      // --- Phase 3: character-level seed, then BPE merge loop.
      // Split LNormalized into per-character token IDs (with byte
      // fallback for unknown characters), then repeatedly merge the
      // best-scoring adjacent pair until none remain.
      LPieces := TList<Integer>.Create();
      try
        LCharIdx := 1;
        while LCharIdx <= Length(LNormalized) do
        begin
          // Detect UTF-16 surrogate pair so emoji / supplementary-plane
          // characters survive as single logical characters.
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

          if FTokenToId.TryGetValue(LCharStr, LCharId) then
          begin
            LPieces.Add(LCharId);
          end
          else
          begin
            // Byte fallback: characters missing from the vocab are
            // emitted as a run of <0xHH> byte tokens covering their
            // UTF-8 bytes.
            LBytes := TEncoding.UTF8.GetBytes(LCharStr);
            for LByte in LBytes do
            begin
              LByteToken := Format('<0x%s>', [IntToHex(LByte, 2)]);
              if FTokenToId.TryGetValue(LByteToken, LCharId) then
                LPieces.Add(LCharId);
            end;
          end;
        end;

        // Greedy BPE: keep merging the highest-scoring adjacent pair
        // until no adjacent pair is in the vocab.
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
            Break;  // defensive — FindBestMerge only returns hits
          LMergeIdx := FindBestMerge(LPieces);
        end;

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


function TVdxTokenizer.Encode(const AText: string): TArray<Integer>;
begin
  Result := EncodeInternal(AText, False);
end;

function TVdxTokenizer.EncodeWithBos(
  const AText: string): TArray<Integer>;
begin
  Result := EncodeInternal(AText, True);
end;

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
      // Reverse the U+2581 substitution Encode applied.
      LToken := LToken.Replace(#$2581, ' ');
      Result := Result + LToken;
    end;
  end;
end;

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

function TVdxTokenizer.IsLoaded(): Boolean;
begin
  Result := FLoaded;
end;

end.
