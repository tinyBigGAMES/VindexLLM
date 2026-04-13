{===============================================================================
  VindexLLM™ - Graph-Walk LLM Inference Engine

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vipervm.org

  See LICENSE for license information
===============================================================================}

unit VindexLLM.TOML;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Classes,
  System.IOUtils,
  System.DateUtils,
  System.Math,
  System.Generics.Collections;

type
  TVdxToml = class;
  TVdxTomlArray = class;

  { TVdxTomlValueKind }
  TVdxTomlValueKind = (
    tvkNone,
    tvkString,
    tvkInteger,
    tvkFloat,
    tvkBoolean,
    tvkDateTime,
    tvkDate,
    tvkTime,
    tvkArray,
    tvkTable
  );

  { TVdxTomlValue }
  TVdxTomlValue = record
  private
    FKind:          TVdxTomlValueKind;
    FStringValue:   string;
    FIntegerValue:  Int64;
    FFloatValue:    Double;
    FBooleanValue:  Boolean;
    FDateTimeValue: TDateTime;
    FArrayValue:    TVdxTomlArray;
    FTableValue:    TVdxToml;
  public
    class function CreateNone(): TVdxTomlValue; static;
    class function CreateString(const AValue: string): TVdxTomlValue; static;
    class function CreateInteger(const AValue: Int64): TVdxTomlValue; static;
    class function CreateFloat(const AValue: Double): TVdxTomlValue; static;
    class function CreateBoolean(const AValue: Boolean): TVdxTomlValue; static;
    class function CreateDateTime(const AValue: TDateTime): TVdxTomlValue; static;
    class function CreateDate(const AValue: TDateTime): TVdxTomlValue; static;
    class function CreateTime(const AValue: TDateTime): TVdxTomlValue; static;
    class function CreateArray(const AValue: TVdxTomlArray): TVdxTomlValue; static;
    class function CreateTable(const AValue: TVdxToml): TVdxTomlValue; static;

    property Kind: TVdxTomlValueKind read FKind;

    function AsString(): string;
    function AsInteger(): Int64;
    function AsFloat(): Double;
    function AsBoolean(): Boolean;
    function AsDateTime(): TDateTime;
    function AsArray(): TVdxTomlArray;
    function AsTable(): TVdxToml;

    function IsNone(): Boolean;
  end;

  { TVdxTomlArray }
  TVdxTomlArray = class
  private
    FItems: TList<TVdxTomlValue>;
    function GetCount(): Integer;
    function GetItem(const AIndex: Integer): TVdxTomlValue;
  public
    constructor Create();
    destructor Destroy(); override;

    procedure Add(const AValue: TVdxTomlValue);
    procedure AddString(const AValue: string);
    procedure AddInteger(const AValue: Int64);
    procedure AddFloat(const AValue: Double);
    procedure AddBoolean(const AValue: Boolean);
    procedure Clear();

    property Count: Integer read GetCount;
    property Items[const AIndex: Integer]: TVdxTomlValue read GetItem; default;
  end;

  { TVdxTomlParseError }
  TVdxTomlParseError = class(Exception)
  private
    FLine:   Integer;
    FColumn: Integer;
  public
    constructor Create(const AMessage: string; const ALine: Integer;
      const AColumn: Integer);
    property Line:   Integer read FLine;
    property Column: Integer read FColumn;
  end;

  { TVdxToml }
  TVdxToml = class
  private
    FValues:         TDictionary<string, TVdxTomlValue>;
    FOwnedTables:    TObjectList<TVdxToml>;
    FOwnedArrays:    TObjectList<TVdxTomlArray>;
    FKeyOrder:       TList<string>;

    // Parser state
    FSource:         string;
    FPos:            Integer;
    FLine:           Integer;
    FColumn:         Integer;
    FLength:         Integer;
    FLastError:      string;
    FCurrentTable:   TVdxToml;
    FImplicitTables: TDictionary<string, TVdxToml>;
    FDefinedTables:  TDictionary<string, Boolean>;
    FArrayTables:    TDictionary<string, TVdxTomlArray>;

    // Comment preservation during parsing
    FComment:        string;
    FKeyComments:    TDictionary<string, string>;
    FPendingComment: string;

    // Lexer helpers
    function  IsEOF(): Boolean;
    function  Peek(const AOffset: Integer = 0): Char;
    procedure Advance(const ACount: Integer = 1);
    procedure SkipWhitespace();
    procedure SkipWhitespaceAndNewlines();
    procedure SkipToEndOfLine();
    function  IsNewline(const ACh: Char): Boolean;
    function  IsWhitespace(const ACh: Char): Boolean;
    function  IsBareKeyChar(const ACh: Char): Boolean;
    function  IsDigit(const ACh: Char): Boolean;
    function  IsHexDigit(const ACh: Char): Boolean;
    function  IsOctalDigit(const ACh: Char): Boolean;
    function  IsBinaryDigit(const ACh: Char): Boolean;
    function  HexValue(const ACh: Char): Integer;

    // Error handling
    procedure Error(const AMessage: string); overload;
    procedure Error(const AMessage: string;
      const AArgs: array of const); overload;

    // Parser methods
    function  ParseDocument(): Boolean;
    procedure ParseExpression();
    procedure ParseKeyValue(const ATable: TVdxToml);
    function  ParseKey(): TArray<string>;
    function  ParseSimpleKey(): string;
    function  ParseBasicString(): string;
    function  ParseMultiLineBasicString(): string;
    function  ParseLiteralString(): string;
    function  ParseMultiLineLiteralString(): string;
    function  ParseEscapeSequence(): string;
    function  ParseValue(): TVdxTomlValue;
    function  ParseNumberOrDateTime(): TVdxTomlValue;
    function  ParseInteger(const AText: string): Int64;
    function  ParseFloat(const AText: string): Double;
    function  ParseDateTime(const AText: string): TVdxTomlValue;
    function  ParseArray(): TVdxTomlArray;
    function  ParseInlineTable(): TVdxToml;
    procedure ParseTableHeader();
    procedure ParseArrayTableHeader();
    procedure CaptureComment();

    // Table navigation
    function  GetOrCreateTablePath(const APath: TArray<string>;
      const AImplicit: Boolean): TVdxToml;
    function  GetOrCreateArrayTablePath(
      const APath: TArray<string>): TVdxToml;
    procedure SetValueAtPath(const ATable: TVdxToml;
      const APath: TArray<string>; const AValue: TVdxTomlValue);
    function  PathToString(const APath: TArray<string>): string;

  public
    // Ownership
    function CreateOwnedTable(): TVdxToml;
    function CreateOwnedArray(): TVdxTomlArray;

    constructor Create();
    destructor Destroy(); override;

    // Parsing
    class function FromFile(const AFilename: string): TVdxToml;
    class function FromString(const ASource: string): TVdxToml;
    function Parse(const ASource: string): Boolean;
    function GetLastError(): string;

    // Value access
    function  TryGetValue(const AKey: string;
      out AValue: TVdxTomlValue): Boolean;
    function  ContainsKey(const AKey: string): Boolean;
    procedure RemoveKey(const AKey: string);

    // Typed getters with defaults
    function GetString(const AKey: string;
      const ADefault: string = ''): string;
    function GetInteger(const AKey: string;
      const ADefault: Int64 = 0): Int64;
    function GetFloat(const AKey: string;
      const ADefault: Double = 0): Double;
    function GetBoolean(const AKey: string;
      const ADefault: Boolean = False): Boolean;
    function GetDateTime(const AKey: string;
      const ADefault: TDateTime = 0): TDateTime;

    // Typed setters
    procedure SetString(const AKey: string; const AValue: string);
    procedure SetInteger(const AKey: string; const AValue: Int64);
    procedure SetFloat(const AKey: string; const AValue: Double);
    procedure SetBoolean(const AKey: string; const AValue: Boolean);
    procedure SetDateTime(const AKey: string; const AValue: TDateTime);

    // Table/Array creation and access
    function GetOrCreateTable(const AKey: string): TVdxToml;
    function GetOrCreateArray(const AKey: string): TVdxTomlArray;
    function GetTable(const AKey: string): TVdxToml;
    function GetArray(const AKey: string): TVdxTomlArray;

    // Enumeration
    function GetKeys(): TArray<string>;
    function GetCount(): Integer;

    // Generic value setter
    procedure SetValue(const AKey: string; const AValue: TVdxTomlValue);

    property Keys: TArray<string> read GetKeys;

    // Comment preservation
    property Comment: string read FComment write FComment;
    function GetKeyComment(const AKey: string): string;
    procedure SetKeyComment(const AKey: string; const AValue: string);
    property KeyComments: TDictionary<string, string> read FKeyComments;
  end;

implementation

{ TVdxTomlValue }

class function TVdxTomlValue.CreateNone(): TVdxTomlValue;
begin
  Result       := Default(TVdxTomlValue);
  Result.FKind := tvkNone;
end;

class function TVdxTomlValue.CreateString(
  const AValue: string): TVdxTomlValue;
begin
  Result              := Default(TVdxTomlValue);
  Result.FKind        := tvkString;
  Result.FStringValue := AValue;
end;

class function TVdxTomlValue.CreateInteger(
  const AValue: Int64): TVdxTomlValue;
begin
  Result               := Default(TVdxTomlValue);
  Result.FKind         := tvkInteger;
  Result.FIntegerValue := AValue;
end;

class function TVdxTomlValue.CreateFloat(
  const AValue: Double): TVdxTomlValue;
begin
  Result             := Default(TVdxTomlValue);
  Result.FKind       := tvkFloat;
  Result.FFloatValue := AValue;
end;

class function TVdxTomlValue.CreateBoolean(
  const AValue: Boolean): TVdxTomlValue;
begin
  Result               := Default(TVdxTomlValue);
  Result.FKind         := tvkBoolean;
  Result.FBooleanValue := AValue;
end;

class function TVdxTomlValue.CreateDateTime(
  const AValue: TDateTime): TVdxTomlValue;
begin
  Result                := Default(TVdxTomlValue);
  Result.FKind          := tvkDateTime;
  Result.FDateTimeValue := AValue;
end;

class function TVdxTomlValue.CreateDate(
  const AValue: TDateTime): TVdxTomlValue;
begin
  Result                := Default(TVdxTomlValue);
  Result.FKind          := tvkDate;
  Result.FDateTimeValue := AValue;
end;

class function TVdxTomlValue.CreateTime(
  const AValue: TDateTime): TVdxTomlValue;
begin
  Result                := Default(TVdxTomlValue);
  Result.FKind          := tvkTime;
  Result.FDateTimeValue := AValue;
end;

class function TVdxTomlValue.CreateArray(
  const AValue: TVdxTomlArray): TVdxTomlValue;
begin
  Result             := Default(TVdxTomlValue);
  Result.FKind       := tvkArray;
  Result.FArrayValue := AValue;
end;

class function TVdxTomlValue.CreateTable(
  const AValue: TVdxToml): TVdxTomlValue;
begin
  Result             := Default(TVdxTomlValue);
  Result.FKind       := tvkTable;
  Result.FTableValue := AValue;
end;

function TVdxTomlValue.AsString(): string;
begin
  if FKind = tvkString then
    Result := FStringValue
  else
    Result := '';
end;

function TVdxTomlValue.AsInteger(): Int64;
begin
  if FKind = tvkInteger then
    Result := FIntegerValue
  else
    Result := 0;
end;

function TVdxTomlValue.AsFloat(): Double;
begin
  if FKind = tvkFloat then
    Result := FFloatValue
  else if FKind = tvkInteger then
    Result := FIntegerValue
  else
    Result := 0;
end;

function TVdxTomlValue.AsBoolean(): Boolean;
begin
  if FKind = tvkBoolean then
    Result := FBooleanValue
  else
    Result := False;
end;

function TVdxTomlValue.AsDateTime(): TDateTime;
begin
  if FKind in [tvkDateTime, tvkDate, tvkTime] then
    Result := FDateTimeValue
  else
    Result := 0;
end;

function TVdxTomlValue.AsArray(): TVdxTomlArray;
begin
  if FKind = tvkArray then
    Result := FArrayValue
  else
    Result := nil;
end;

function TVdxTomlValue.AsTable(): TVdxToml;
begin
  if FKind = tvkTable then
    Result := FTableValue
  else
    Result := nil;
end;

function TVdxTomlValue.IsNone(): Boolean;
begin
  Result := FKind = tvkNone;
end;

{ TVdxTomlArray }

constructor TVdxTomlArray.Create();
begin
  inherited;
  FItems := TList<TVdxTomlValue>.Create();
end;

destructor TVdxTomlArray.Destroy();
begin
  FItems.Free();
  inherited;
end;

function TVdxTomlArray.GetCount(): Integer;
begin
  Result := FItems.Count;
end;

function TVdxTomlArray.GetItem(const AIndex: Integer): TVdxTomlValue;
begin
  Result := FItems[AIndex];
end;

procedure TVdxTomlArray.Add(const AValue: TVdxTomlValue);
begin
  FItems.Add(AValue);
end;

procedure TVdxTomlArray.AddString(const AValue: string);
begin
  FItems.Add(TVdxTomlValue.CreateString(AValue));
end;

procedure TVdxTomlArray.AddInteger(const AValue: Int64);
begin
  FItems.Add(TVdxTomlValue.CreateInteger(AValue));
end;

procedure TVdxTomlArray.AddFloat(const AValue: Double);
begin
  FItems.Add(TVdxTomlValue.CreateFloat(AValue));
end;

procedure TVdxTomlArray.AddBoolean(const AValue: Boolean);
begin
  FItems.Add(TVdxTomlValue.CreateBoolean(AValue));
end;

procedure TVdxTomlArray.Clear();
begin
  FItems.Clear();
end;

{ TVdxTomlParseError }

constructor TVdxTomlParseError.Create(const AMessage: string;
  const ALine: Integer; const AColumn: Integer);
begin
  inherited CreateFmt('%s at line %d, column %d', [AMessage, ALine, AColumn]);
  FLine   := ALine;
  FColumn := AColumn;
end;

{ TVdxToml }

constructor TVdxToml.Create();
begin
  inherited;
  FValues         := TDictionary<string, TVdxTomlValue>.Create();
  FOwnedTables    := TObjectList<TVdxToml>.Create(True);
  FOwnedArrays    := TObjectList<TVdxTomlArray>.Create(True);
  FKeyOrder       := TList<string>.Create();
  FImplicitTables := nil;
  FDefinedTables  := nil;
  FArrayTables    := nil;
  FCurrentTable   := nil;
  FLastError      := '';
  FKeyComments    := TDictionary<string, string>.Create();
  FComment        := '';
  FPendingComment := '';
end;

destructor TVdxToml.Destroy();
begin
  FKeyOrder.Free();
  FKeyComments.Free();
  FOwnedArrays.Free();
  FOwnedTables.Free();
  FValues.Free();
  FImplicitTables.Free();
  FDefinedTables.Free();
  FArrayTables.Free();
  inherited;
end;

class function TVdxToml.FromFile(const AFilename: string): TVdxToml;
var
  LSource: string;
begin
  if not TFile.Exists(AFilename) then
    raise TVdxTomlParseError.Create('File not found: ' + AFilename, 0, 0);

  LSource := TFile.ReadAllText(AFilename, TEncoding.UTF8);
  Result  := FromString(LSource);
end;

class function TVdxToml.FromString(const ASource: string): TVdxToml;
begin
  Result := TVdxToml.Create();
  try
    if not Result.Parse(ASource) then
      raise TVdxTomlParseError.Create(Result.GetLastError(),
        Result.FLine, Result.FColumn);
  except
    Result.Free();
    raise;
  end;
end;

function TVdxToml.Parse(const ASource: string): Boolean;
begin
  FSource       := ASource;
  FPos          := 1;
  FLine         := 1;
  FColumn       := 1;
  FLength       := Length(FSource);
  FLastError    := '';
  FCurrentTable := Self;
  FPendingComment := '';

  // Initialize tracking dictionaries
  FreeAndNil(FImplicitTables);
  FreeAndNil(FDefinedTables);
  FreeAndNil(FArrayTables);
  FImplicitTables := TDictionary<string, TVdxToml>.Create();
  FDefinedTables  := TDictionary<string, Boolean>.Create();
  FArrayTables    := TDictionary<string, TVdxTomlArray>.Create();

  try
    Result := ParseDocument();
  except
    on E: TVdxTomlParseError do
    begin
      FLastError := E.Message;
      FLine      := E.Line;
      FColumn    := E.Column;
      Result     := False;
    end;
    on E: Exception do
    begin
      FLastError := E.Message;
      Result     := False;
    end;
  end;
end;

function TVdxToml.GetLastError(): string;
begin
  Result := FLastError;
end;

// -- Lexer helpers ------------------------------------------------------------

function TVdxToml.IsEOF(): Boolean;
begin
  Result := FPos > FLength;
end;

function TVdxToml.Peek(const AOffset: Integer): Char;
var
  LIndex: Integer;
begin
  LIndex := FPos + AOffset;
  if (LIndex >= 1) and (LIndex <= FLength) then
    Result := FSource[LIndex]
  else
    Result := #0;
end;

procedure TVdxToml.Advance(const ACount: Integer);
var
  LI: Integer;
begin
  for LI := 1 to ACount do
  begin
    if FPos <= FLength then
    begin
      if FSource[FPos] = #10 then
      begin
        Inc(FLine);
        FColumn := 1;
      end
      else if FSource[FPos] <> #13 then
        Inc(FColumn);
      Inc(FPos);
    end;
  end;
end;

procedure TVdxToml.SkipWhitespace();
begin
  while not IsEOF() and IsWhitespace(Peek()) do
    Advance();
end;

procedure TVdxToml.SkipWhitespaceAndNewlines();
begin
  while not IsEOF() do
  begin
    if IsWhitespace(Peek()) or IsNewline(Peek()) then
      Advance()
    else if Peek() = '#' then
      CaptureComment()
    else
      Break;
  end;
end;

procedure TVdxToml.SkipToEndOfLine();
begin
  while not IsEOF() and not IsNewline(Peek()) do
    Advance();
end;

function TVdxToml.IsNewline(const ACh: Char): Boolean;
begin
  Result := (ACh = #10) or (ACh = #13);
end;

function TVdxToml.IsWhitespace(const ACh: Char): Boolean;
begin
  Result := (ACh = ' ') or (ACh = #9);
end;

function TVdxToml.IsBareKeyChar(const ACh: Char): Boolean;
begin
  Result := CharInSet(ACh, ['A'..'Z', 'a'..'z', '0'..'9', '_', '-']);
end;

function TVdxToml.IsDigit(const ACh: Char): Boolean;
begin
  Result := CharInSet(ACh, ['0'..'9']);
end;

function TVdxToml.IsHexDigit(const ACh: Char): Boolean;
begin
  Result := CharInSet(ACh, ['0'..'9', 'A'..'F', 'a'..'f']);
end;

function TVdxToml.IsOctalDigit(const ACh: Char): Boolean;
begin
  Result := CharInSet(ACh, ['0'..'7']);
end;

function TVdxToml.IsBinaryDigit(const ACh: Char): Boolean;
begin
  Result := CharInSet(ACh, ['0', '1']);
end;

function TVdxToml.HexValue(const ACh: Char): Integer;
begin
  if CharInSet(ACh, ['0'..'9']) then
    Result := Ord(ACh) - Ord('0')
  else if CharInSet(ACh, ['A'..'F']) then
    Result := Ord(ACh) - Ord('A') + 10
  else if CharInSet(ACh, ['a'..'f']) then
    Result := Ord(ACh) - Ord('a') + 10
  else
    Result := 0;
end;

// -- Error handling -----------------------------------------------------------

procedure TVdxToml.Error(const AMessage: string);
begin
  raise TVdxTomlParseError.Create(AMessage, FLine, FColumn);
end;

procedure TVdxToml.Error(const AMessage: string;
  const AArgs: array of const);
begin
  Error(Format(AMessage, AArgs));
end;

// -- Ownership helpers --------------------------------------------------------

function TVdxToml.CreateOwnedTable(): TVdxToml;
begin
  Result := TVdxToml.Create();
  FOwnedTables.Add(Result);
end;

function TVdxToml.CreateOwnedArray(): TVdxTomlArray;
begin
  Result := TVdxTomlArray.Create();
  FOwnedArrays.Add(Result);
end;

// -- Parser methods -----------------------------------------------------------

procedure TVdxToml.CaptureComment();
var
  LStart: Integer;
  LLine:  string;
begin
  // FPos points to '#'; capture everything to end of line
  LStart := FPos;
  SkipToEndOfLine();
  LLine := Copy(FSource, LStart, FPos - LStart);
  if FPendingComment <> '' then
    FPendingComment := FPendingComment + #10 + LLine
  else
    FPendingComment := LLine;
end;

function TVdxToml.ParseDocument(): Boolean;
begin
  Result := True;

  while not IsEOF() do
  begin
    SkipWhitespaceAndNewlines();

    if IsEOF() then
      Break;

    ParseExpression();
  end;

  // Any trailing comments stored on root
  if FPendingComment <> '' then
  begin
    Self.FComment := FPendingComment;
    FPendingComment := '';
  end;
end;

procedure TVdxToml.ParseExpression();
begin
  SkipWhitespace();

  if IsEOF() or IsNewline(Peek()) then
    Exit;

  // Comment
  if Peek() = '#' then
  begin
    CaptureComment();
    Exit;
  end;

  // Table header
  if Peek() = '[' then
  begin
    if Peek(1) = '[' then
      ParseArrayTableHeader()
    else
      ParseTableHeader();
    Exit;
  end;

  // Key-value pair
  ParseKeyValue(FCurrentTable);

  // Must be followed by newline, comment, or EOF
  SkipWhitespace();
  if not IsEOF() and not IsNewline(Peek()) and (Peek() <> '#') then
    Error('Expected newline or end of file after key-value pair');
end;

procedure TVdxToml.ParseKeyValue(const ATable: TVdxToml);
var
  LKey:   TArray<string>;
  LValue: TVdxTomlValue;
begin
  LKey := ParseKey();

  // Attach any pending comment to this key
  if FPendingComment <> '' then
  begin
    ATable.FKeyComments.AddOrSetValue(LKey[0], FPendingComment);
    FPendingComment := '';
  end;

  SkipWhitespace();
  if Peek() <> '=' then
    Error('Expected ''='' after key');
  Advance(); // skip =

  SkipWhitespace();
  LValue := ParseValue();

  SetValueAtPath(ATable, LKey, LValue);
end;

function TVdxToml.ParseKey(): TArray<string>;
var
  LParts: TList<string>;
  LKey:   string;
begin
  LParts := TList<string>.Create();
  try
    LKey := ParseSimpleKey();
    LParts.Add(LKey);

    // Check for dotted key
    SkipWhitespace();
    while Peek() = '.' do
    begin
      Advance(); // skip .
      SkipWhitespace();
      LKey := ParseSimpleKey();
      LParts.Add(LKey);
      SkipWhitespace();
    end;

    Result := LParts.ToArray();
  finally
    LParts.Free();
  end;
end;

function TVdxToml.ParseSimpleKey(): string;
begin
  if Peek() = '"' then
    Result := ParseBasicString()
  else if Peek() = '''' then
    Result := ParseLiteralString()
  else if IsBareKeyChar(Peek()) then
  begin
    Result := '';
    while not IsEOF() and IsBareKeyChar(Peek()) do
    begin
      Result := Result + Peek();
      Advance();
    end;
  end
  else
    Error('Expected key');
end;

function TVdxToml.ParseBasicString(): string;
begin
  if Peek() <> '"' then
    Error('Expected ''"''');
  Advance();

  // Check for multi-line
  if (Peek() = '"') and (Peek(1) = '"') then
  begin
    Advance(2);
    Result := ParseMultiLineBasicString();
    Exit;
  end;

  Result := '';
  while not IsEOF() do
  begin
    if Peek() = '"' then
    begin
      Advance();
      Exit;
    end
    else if Peek() = '\' then
    begin
      Advance();
      Result := Result + ParseEscapeSequence();
    end
    else if IsNewline(Peek()) then
      Error('Newline in basic string')
    else if (Ord(Peek()) < 32) and (Peek() <> #9) then
      Error('Control character in string')
    else
    begin
      Result := Result + Peek();
      Advance();
    end;
  end;

  Error('Unterminated string');
end;

function TVdxToml.ParseMultiLineBasicString(): string;
var
  LSkipNewline: Boolean;
begin
  Result := '';

  // Skip immediate newline after opening quotes
  if Peek() = #13 then
    Advance();
  if Peek() = #10 then
    Advance();

  while not IsEOF() do
  begin
    // Check for closing quotes
    if (Peek() = '"') and (Peek(1) = '"') and (Peek(2) = '"') then
    begin
      // Handle up to 2 additional quotes before closing
      if Peek(3) = '"' then
      begin
        if Peek(4) = '"' then
        begin
          Result := Result + '""';
          Advance(5);
        end
        else
        begin
          Result := Result + '"';
          Advance(4);
        end;
      end
      else
        Advance(3);
      Exit;
    end
    else if Peek() = '\' then
    begin
      Advance();
      // Line-ending backslash trims following whitespace/newlines
      if IsNewline(Peek()) or IsWhitespace(Peek()) then
      begin
        LSkipNewline := False;
        while not IsEOF() and (IsWhitespace(Peek()) or IsNewline(Peek())) do
        begin
          if IsNewline(Peek()) then
            LSkipNewline := True;
          Advance();
        end;
        if not LSkipNewline then
          Error('Invalid escape sequence');
      end
      else
        Result := Result + ParseEscapeSequence();
    end
    else if (Ord(Peek()) < 32) and not CharInSet(Peek(), [#9, #10, #13]) then
      Error('Control character in string')
    else
    begin
      Result := Result + Peek();
      Advance();
    end;
  end;

  Error('Unterminated multi-line string');
end;

function TVdxToml.ParseLiteralString(): string;
begin
  if Peek() <> '''' then
    Error('Expected ''''');
  Advance();

  // Check for multi-line
  if (Peek() = '''') and (Peek(1) = '''') then
  begin
    Advance(2);
    Result := ParseMultiLineLiteralString();
    Exit;
  end;

  Result := '';
  while not IsEOF() do
  begin
    if Peek() = '''' then
    begin
      Advance();
      Exit;
    end
    else if IsNewline(Peek()) then
      Error('Newline in literal string')
    else if (Ord(Peek()) < 32) and (Peek() <> #9) then
      Error('Control character in string')
    else
    begin
      Result := Result + Peek();
      Advance();
    end;
  end;

  Error('Unterminated string');
end;

function TVdxToml.ParseMultiLineLiteralString(): string;
begin
  Result := '';

  // Skip immediate newline after opening quotes
  if Peek() = #13 then
    Advance();
  if Peek() = #10 then
    Advance();

  while not IsEOF() do
  begin
    // Check for closing quotes
    if (Peek() = '''') and (Peek(1) = '''') and (Peek(2) = '''') then
    begin
      // Handle up to 2 additional quotes before closing
      if Peek(3) = '''' then
      begin
        if Peek(4) = '''' then
        begin
          Result := Result + '''''';
          Advance(5);
        end
        else
        begin
          Result := Result + '''';
          Advance(4);
        end;
      end
      else
        Advance(3);
      Exit;
    end
    else if (Ord(Peek()) < 32) and not CharInSet(Peek(), [#9, #10, #13]) then
      Error('Control character in string')
    else
    begin
      Result := Result + Peek();
      Advance();
    end;
  end;

  Error('Unterminated multi-line string');
end;

function TVdxToml.ParseEscapeSequence(): string;
var
  LCodePoint: Integer;
  LI:         Integer;
begin
  case Peek() of
    'b':
      begin
        Advance();
        Result := #8;
      end;
    't':
      begin
        Advance();
        Result := #9;
      end;
    'n':
      begin
        Advance();
        Result := #10;
      end;
    'f':
      begin
        Advance();
        Result := #12;
      end;
    'r':
      begin
        Advance();
        Result := #13;
      end;
    '"':
      begin
        Advance();
        Result := '"';
      end;
    '\':
      begin
        Advance();
        Result := '\';
      end;
    'u':
      begin
        Advance();
        LCodePoint := 0;
        for LI := 1 to 4 do
        begin
          if not IsHexDigit(Peek()) then
            Error('Invalid unicode escape');
          LCodePoint := LCodePoint * 16 + HexValue(Peek());
          Advance();
        end;
        Result := Char(LCodePoint);
      end;
    'U':
      begin
        Advance();
        LCodePoint := 0;
        for LI := 1 to 8 do
        begin
          if not IsHexDigit(Peek()) then
            Error('Invalid unicode escape');
          LCodePoint := LCodePoint * 16 + HexValue(Peek());
          Advance();
        end;
        // Convert to UTF-16 surrogate pair if needed
        if LCodePoint > $FFFF then
        begin
          Dec(LCodePoint, $10000);
          Result := Char($D800 + (LCodePoint shr 10)) +
                    Char($DC00 + (LCodePoint and $3FF));
        end
        else
          Result := Char(LCodePoint);
      end;
  else
    Error('Invalid escape sequence: \%s', [Peek()]);
    Result := '';
  end;
end;

function TVdxToml.ParseValue(): TVdxTomlValue;
var
  LCh: Char;
begin
  LCh := Peek();

  if LCh = '"' then
    Result := TVdxTomlValue.CreateString(ParseBasicString())
  else if LCh = '''' then
    Result := TVdxTomlValue.CreateString(ParseLiteralString())
  else if LCh = '[' then
    Result := TVdxTomlValue.CreateArray(ParseArray())
  else if LCh = '{' then
    Result := TVdxTomlValue.CreateTable(ParseInlineTable())
  else if (LCh = 't') and (Peek(1) = 'r') and (Peek(2) = 'u') and
          (Peek(3) = 'e') and not IsBareKeyChar(Peek(4)) then
  begin
    Advance(4);
    Result := TVdxTomlValue.CreateBoolean(True);
  end
  else if (LCh = 'f') and (Peek(1) = 'a') and (Peek(2) = 'l') and
          (Peek(3) = 's') and (Peek(4) = 'e') and
          not IsBareKeyChar(Peek(5)) then
  begin
    Advance(5);
    Result := TVdxTomlValue.CreateBoolean(False);
  end
  else if CharInSet(LCh, ['+', '-', '0'..'9', 'i', 'n']) then
    Result := ParseNumberOrDateTime()
  else
  begin
    Error('Invalid value');
    Result := TVdxTomlValue.CreateNone();
  end;
end;

function TVdxToml.ParseNumberOrDateTime(): TVdxTomlValue;
var
  LStart:    Integer;
  LText:     string;
  LHasColon: Boolean;
  LHasDash:  Boolean;
  LHasT:     Boolean;
  LHasDot:   Boolean;
  LHasE:     Boolean;
  LIsHex:    Boolean;
  LIsOctal:  Boolean;
  LIsBinary: Boolean;
  LI:        Integer;
begin
  LStart    := FPos;
  LHasColon := False;
  LHasDash  := False;
  LHasT     := False;
  LHasDot   := False;
  LHasE     := False;
  LIsHex    := False;
  LIsOctal  := False;
  LIsBinary := False;

  // Check for inf/nan
  if CharInSet(Peek(), ['i', 'n']) then
  begin
    if (Peek() = 'i') and (Peek(1) = 'n') and (Peek(2) = 'f') then
    begin
      Advance(3);
      Result := TVdxTomlValue.CreateFloat(Infinity);
      Exit;
    end
    else if (Peek() = 'n') and (Peek(1) = 'a') and (Peek(2) = 'n') then
    begin
      Advance(3);
      Result := TVdxTomlValue.CreateFloat(NaN);
      Exit;
    end
    else
      Error('Invalid value');
  end;

  // Skip sign
  if CharInSet(Peek(), ['+', '-']) then
  begin
    if Peek() = '-' then
    begin
      // Check for -inf, -nan
      if (Peek(1) = 'i') and (Peek(2) = 'n') and (Peek(3) = 'f') then
      begin
        Advance(4);
        Result := TVdxTomlValue.CreateFloat(NegInfinity);
        Exit;
      end
      else if (Peek(1) = 'n') and (Peek(2) = 'a') and (Peek(3) = 'n') then
      begin
        Advance(4);
        Result := TVdxTomlValue.CreateFloat(NaN);
        Exit;
      end;
    end
    else
    begin
      // Check for +inf, +nan
      if (Peek(1) = 'i') and (Peek(2) = 'n') and (Peek(3) = 'f') then
      begin
        Advance(4);
        Result := TVdxTomlValue.CreateFloat(Infinity);
        Exit;
      end
      else if (Peek(1) = 'n') and (Peek(2) = 'a') and (Peek(3) = 'n') then
      begin
        Advance(4);
        Result := TVdxTomlValue.CreateFloat(NaN);
        Exit;
      end;
    end;
    Advance();
  end;

  // Check for hex/octal/binary prefix
  if (Peek() = '0') and CharInSet(Peek(1), ['x', 'X', 'o', 'O', 'b', 'B']) then
  begin
    if CharInSet(Peek(1), ['x', 'X']) then
      LIsHex := True
    else if CharInSet(Peek(1), ['o', 'O']) then
      LIsOctal := True
    else
      LIsBinary := True;
    Advance(2);
  end;

  // Scan the rest of the number/datetime
  while not IsEOF() do
  begin
    if IsDigit(Peek()) then
      Advance()
    else if Peek() = '_' then
      Advance()
    else if Peek() = ':' then
    begin
      LHasColon := True;
      Advance();
    end
    else if Peek() = '-' then
    begin
      LHasDash := True;
      Advance();
    end
    else if CharInSet(Peek(), ['T', 't', ' ']) and LHasDash then
    begin
      LHasT := True;
      Advance();
    end
    else if Peek() = '.' then
    begin
      LHasDot := True;
      Advance();
    end
    else if CharInSet(Peek(), ['e', 'E']) then
    begin
      LHasE := True;
      Advance();
      if CharInSet(Peek(), ['+', '-']) then
        Advance();
    end
    else if CharInSet(Peek(), ['Z', 'z']) then
      Advance()
    else if (Peek() = '+') and LHasT then
      Advance()
    else if LIsHex and IsHexDigit(Peek()) then
      Advance()
    else if LIsOctal and IsOctalDigit(Peek()) then
      Advance()
    else if LIsBinary and IsBinaryDigit(Peek()) then
      Advance()
    else
      Break;
  end;

  // Extract the text without underscores
  LText := '';
  for LI := LStart to FPos - 1 do
  begin
    if FSource[LI] <> '_' then
      LText := LText + FSource[LI];
  end;

  // Determine type
  if LHasColon or (LHasDash and (Length(LText) >= 10)) then
    Result := ParseDateTime(LText)
  else if LHasDot or LHasE then
    Result := TVdxTomlValue.CreateFloat(ParseFloat(LText))
  else
    Result := TVdxTomlValue.CreateInteger(ParseInteger(LText));
end;

function TVdxToml.ParseInteger(const AText: string): Int64;
var
  LText: string;
  LNeg:  Boolean;
  LI:    Integer;
begin
  LText := AText;
  LNeg  := False;

  if LText = '' then
    Error('Empty integer');

  if LText[1] = '+' then
    Delete(LText, 1, 1)
  else if LText[1] = '-' then
  begin
    LNeg := True;
    Delete(LText, 1, 1);
  end;

  if LText = '' then
    Error('Empty integer');

  // Hex
  if (Length(LText) >= 2) and (LText[1] = '0') and
     CharInSet(LText[2], ['x', 'X']) then
  begin
    Delete(LText, 1, 2);
    Result := 0;
    for LI := 1 to Length(LText) do
      Result := Result * 16 + HexValue(LText[LI]);
  end
  // Octal
  else if (Length(LText) >= 2) and (LText[1] = '0') and
          CharInSet(LText[2], ['o', 'O']) then
  begin
    Delete(LText, 1, 2);
    Result := 0;
    for LI := 1 to Length(LText) do
      Result := Result * 8 + (Ord(LText[LI]) - Ord('0'));
  end
  // Binary
  else if (Length(LText) >= 2) and (LText[1] = '0') and
          CharInSet(LText[2], ['b', 'B']) then
  begin
    Delete(LText, 1, 2);
    Result := 0;
    for LI := 1 to Length(LText) do
      Result := Result * 2 + (Ord(LText[LI]) - Ord('0'));
  end
  // Decimal
  else
  begin
    // Leading zeros are not allowed except for single 0
    if (Length(LText) > 1) and (LText[1] = '0') then
      Error('Leading zeros not allowed');
    Result := StrToInt64(LText);
  end;

  if LNeg then
    Result := -Result;
end;

function TVdxToml.ParseFloat(const AText: string): Double;
var
  LFormatSettings: TFormatSettings;
begin
  LFormatSettings                  := TFormatSettings.Create();
  LFormatSettings.DecimalSeparator := '.';

  if not TryStrToFloat(AText, Result, LFormatSettings) then
    Error('Invalid float: %s', [AText]);
end;

function TVdxToml.ParseDateTime(const AText: string): TVdxTomlValue;
var
  LText:        string;
  LYear:        Integer;
  LMonth:       Integer;
  LDay:         Integer;
  LHour:        Integer;
  LMinute:      Integer;
  LSecond:      Integer;
  LMillisecond: Integer;
  LDateTime:    TDateTime;
  LHasDate:     Boolean;
  LHasTime:     Boolean;
  LDatePart:    string;
  LTimePart:    string;
  LSepPos:      Integer;
  LOffsetPos:   Integer;
  LFracPos:     Integer;
  LFracStr:     string;
begin
  LText        := AText;
  LHasDate     := False;
  LHasTime     := False;
  LYear        := 0;
  LMonth       := 1;
  LDay         := 1;
  LHour        := 0;
  LMinute      := 0;
  LSecond      := 0;
  LMillisecond := 0;

  // Find T or space separator
  LSepPos := Pos('T', LText);
  if LSepPos = 0 then
    LSepPos := Pos('t', LText);
  if LSepPos = 0 then
    LSepPos := Pos(' ', LText);

  // Determine date and time parts
  if LSepPos > 0 then
  begin
    LDatePart := Copy(LText, 1, LSepPos - 1);
    LTimePart := Copy(LText, LSepPos + 1, Length(LText));
    LHasDate  := True;
    LHasTime  := True;
  end
  else if Pos(':', LText) > 0 then
  begin
    LTimePart := LText;
    LHasTime  := True;
  end
  else
  begin
    LDatePart := LText;
    LHasDate  := True;
  end;

  // Parse date
  if LHasDate and (Length(LDatePart) >= 10) then
  begin
    LYear  := StrToInt(Copy(LDatePart, 1, 4));
    LMonth := StrToInt(Copy(LDatePart, 6, 2));
    LDay   := StrToInt(Copy(LDatePart, 9, 2));
  end;

  // Parse time
  if LHasTime then
  begin
    // Remove timezone offset for parsing
    LOffsetPos := Pos('Z', LTimePart);
    if LOffsetPos = 0 then
      LOffsetPos := Pos('z', LTimePart);
    if LOffsetPos = 0 then
    begin
      LOffsetPos := LastDelimiter('+-', LTimePart);
      // Ensure it's not the leading sign
      if LOffsetPos <= 2 then
        LOffsetPos := 0;
    end;

    if LOffsetPos > 0 then
      LTimePart := Copy(LTimePart, 1, LOffsetPos - 1);

    // Parse fractional seconds
    LFracPos := Pos('.', LTimePart);
    if LFracPos > 0 then
    begin
      LFracStr := Copy(LTimePart, LFracPos + 1, Length(LTimePart));
      // Pad or truncate to 3 digits for milliseconds
      while Length(LFracStr) < 3 do
        LFracStr := LFracStr + '0';
      LMillisecond := StrToIntDef(Copy(LFracStr, 1, 3), 0);
      LTimePart    := Copy(LTimePart, 1, LFracPos - 1);
    end;

    if Length(LTimePart) >= 5 then
    begin
      LHour   := StrToInt(Copy(LTimePart, 1, 2));
      LMinute := StrToInt(Copy(LTimePart, 4, 2));
      if Length(LTimePart) >= 8 then
        LSecond := StrToInt(Copy(LTimePart, 7, 2));
    end;
  end;

  // Build TDateTime
  if LHasDate then
    LDateTime := EncodeDate(LYear, LMonth, LDay)
  else
    LDateTime := 0;

  if LHasTime then
    LDateTime := LDateTime + EncodeTime(LHour, LMinute, LSecond, LMillisecond);

  // Return appropriate type
  if LHasDate and LHasTime then
    Result := TVdxTomlValue.CreateDateTime(LDateTime)
  else if LHasDate then
    Result := TVdxTomlValue.CreateDate(LDateTime)
  else
    Result := TVdxTomlValue.CreateTime(LDateTime);
end;

function TVdxToml.ParseArray(): TVdxTomlArray;
var
  LValue: TVdxTomlValue;
begin
  if Peek() <> '[' then
    Error('Expected ''[''');
  Advance();

  Result := CreateOwnedArray();

  SkipWhitespaceAndNewlines();

  // Empty array
  if Peek() = ']' then
  begin
    Advance();
    Exit;
  end;

  // First value
  LValue := ParseValue();
  Result.Add(LValue);

  SkipWhitespaceAndNewlines();

  // Additional values
  while Peek() = ',' do
  begin
    Advance();
    SkipWhitespaceAndNewlines();

    // Trailing comma
    if Peek() = ']' then
      Break;

    LValue := ParseValue();
    Result.Add(LValue);

    SkipWhitespaceAndNewlines();
  end;

  if Peek() <> ']' then
    Error('Expected '']'' or '',''');
  Advance();
end;

function TVdxToml.ParseInlineTable(): TVdxToml;
var
  LKey:   TArray<string>;
  LValue: TVdxTomlValue;
begin
  if Peek() <> '{' then
    Error('Expected ''{''');
  Advance();

  Result := CreateOwnedTable();

  SkipWhitespace();

  // Empty table
  if Peek() = '}' then
  begin
    Advance();
    Exit;
  end;

  // First key-value
  LKey := ParseKey();
  SkipWhitespace();
  if Peek() <> '=' then
    Error('Expected ''=''');
  Advance();
  SkipWhitespace();
  LValue := ParseValue();
  SetValueAtPath(Result, LKey, LValue);

  SkipWhitespace();

  // Additional key-values
  while Peek() = ',' do
  begin
    Advance();
    SkipWhitespace();

    LKey := ParseKey();
    SkipWhitespace();
    if Peek() <> '=' then
      Error('Expected ''=''');
    Advance();
    SkipWhitespace();
    LValue := ParseValue();
    SetValueAtPath(Result, LKey, LValue);

    SkipWhitespace();
  end;

  if Peek() <> '}' then
    Error('Expected ''}'' or '',''');
  Advance();
end;

procedure TVdxToml.ParseTableHeader();
var
  LKey:  TArray<string>;
  LPath: string;
begin
  if Peek() <> '[' then
    Error('Expected ''[''');
  Advance();

  SkipWhitespace();
  LKey := ParseKey();
  SkipWhitespace();

  if Peek() <> ']' then
    Error('Expected '']''');
  Advance();

  LPath := PathToString(LKey);

  // Check if already defined as a regular table
  if FDefinedTables.ContainsKey(LPath) then
    Error('Table ''%s'' already defined', [LPath]);

  // Check if it was used as an array table
  if FArrayTables.ContainsKey(LPath) then
    Error('Cannot redefine array table ''%s'' as regular table', [LPath]);

  FCurrentTable := GetOrCreateTablePath(LKey, False);
  FDefinedTables.Add(LPath, True);

  // Attach pending comment to the new table
  if FPendingComment <> '' then
  begin
    FCurrentTable.FComment := FPendingComment;
    FPendingComment := '';
  end;
end;

procedure TVdxToml.ParseArrayTableHeader();
var
  LKey:  TArray<string>;
  LPath: string;
begin
  if (Peek() <> '[') or (Peek(1) <> '[') then
    Error('Expected ''[[''');
  Advance(2);

  SkipWhitespace();
  LKey := ParseKey();
  SkipWhitespace();

  if (Peek() <> ']') or (Peek(1) <> ']') then
    Error('Expected '']]''');
  Advance(2);

  LPath := PathToString(LKey);

  // Check if it was defined as a regular table
  if FDefinedTables.ContainsKey(LPath) and
     not FArrayTables.ContainsKey(LPath) then
    Error('Cannot redefine table ''%s'' as array table', [LPath]);

  FCurrentTable := GetOrCreateArrayTablePath(LKey);

  // Attach pending comment to the new array-of-table entry
  if FPendingComment <> '' then
  begin
    FCurrentTable.FComment := FPendingComment;
    FPendingComment := '';
  end;
end;

function TVdxToml.PathToString(const APath: TArray<string>): string;
var
  LI: Integer;
begin
  Result := '';
  for LI := 0 to High(APath) do
  begin
    if LI > 0 then
      Result := Result + '.';
    Result := Result + APath[LI];
  end;
end;

function TVdxToml.GetOrCreateTablePath(const APath: TArray<string>;
  const AImplicit: Boolean): TVdxToml;
var
  LCurrent:  TVdxToml;
  LI:        Integer;
  LKey:      string;
  LValue:    TVdxTomlValue;
  LSubPath:  string;
  LNewTable: TVdxToml;
  LLastItem: TVdxTomlValue;
begin
  LCurrent := Self;

  for LI := 0 to High(APath) do
  begin
    LKey := APath[LI];

    if LCurrent.FValues.TryGetValue(LKey, LValue) then
    begin
      if LValue.Kind = tvkTable then
        LCurrent := LValue.AsTable()
      else if LValue.Kind = tvkArray then
      begin
        // Get last table from array
        if LValue.AsArray().Count > 0 then
        begin
          LLastItem := LValue.AsArray()[LValue.AsArray().Count - 1];
          if LLastItem.Kind = tvkTable then
            LCurrent := LLastItem.AsTable()
          else
            Error('Expected table in array');
        end
        else
          Error('Empty array table');
      end
      else
        Error('Key ''%s'' is not a table', [LKey]);
    end
    else
    begin
      // Create new table
      LNewTable := CreateOwnedTable();
      LCurrent.FValues.Add(LKey, TVdxTomlValue.CreateTable(LNewTable));
      if not LCurrent.FKeyOrder.Contains(LKey) then
        LCurrent.FKeyOrder.Add(LKey);

      // Track implicit tables
      if AImplicit or (LI < High(APath)) then
      begin
        LSubPath := PathToString(Copy(APath, 0, LI + 1));
        if not FImplicitTables.ContainsKey(LSubPath) then
          FImplicitTables.Add(LSubPath, LNewTable);
      end;

      LCurrent := LNewTable;
    end;
  end;

  Result := LCurrent;
end;

function TVdxToml.GetOrCreateArrayTablePath(
  const APath: TArray<string>): TVdxToml;
var
  LCurrent:  TVdxToml;
  LI:        Integer;
  LKey:      string;
  LValue:    TVdxTomlValue;
  LPath:     string;
  LNewTable: TVdxToml;
  LArray:    TVdxTomlArray;
  LLastItem: TVdxTomlValue;
begin
  LCurrent := Self;

  for LI := 0 to High(APath) - 1 do
  begin
    LKey := APath[LI];

    if LCurrent.FValues.TryGetValue(LKey, LValue) then
    begin
      if LValue.Kind = tvkTable then
        LCurrent := LValue.AsTable()
      else if LValue.Kind = tvkArray then
      begin
        // Get last table from array
        if LValue.AsArray().Count > 0 then
        begin
          LLastItem := LValue.AsArray()[LValue.AsArray().Count - 1];
          if LLastItem.Kind = tvkTable then
            LCurrent := LLastItem.AsTable()
          else
            Error('Expected table in array');
        end
        else
          Error('Empty array table');
      end
      else
        Error('Key ''%s'' is not a table', [LKey]);
    end
    else
    begin
      // Create implicit table
      LNewTable := CreateOwnedTable();
      LCurrent.FValues.Add(LKey, TVdxTomlValue.CreateTable(LNewTable));
      if not LCurrent.FKeyOrder.Contains(LKey) then
        LCurrent.FKeyOrder.Add(LKey);
      LCurrent := LNewTable;
    end;
  end;

  // Handle final key as array
  LKey  := APath[High(APath)];
  LPath := PathToString(APath);

  if LCurrent.FValues.TryGetValue(LKey, LValue) then
  begin
    if LValue.Kind <> tvkArray then
      Error('Key ''%s'' is not an array', [LKey]);
    LArray := LValue.AsArray();
  end
  else
  begin
    LArray := CreateOwnedArray();
    LCurrent.FValues.Add(LKey, TVdxTomlValue.CreateArray(LArray));
    if not LCurrent.FKeyOrder.Contains(LKey) then
      LCurrent.FKeyOrder.Add(LKey);
    FArrayTables.Add(LPath, LArray);
  end;

  // Create new table entry in array
  LNewTable := CreateOwnedTable();
  LArray.Add(TVdxTomlValue.CreateTable(LNewTable));

  Result := LNewTable;
end;

procedure TVdxToml.SetValueAtPath(const ATable: TVdxToml;
  const APath: TArray<string>; const AValue: TVdxTomlValue);
var
  LCurrent:  TVdxToml;
  LI:        Integer;
  LKey:      string;
  LValue:    TVdxTomlValue;
  LNewTable: TVdxToml;
begin
  LCurrent := ATable;

  // Navigate to parent table
  for LI := 0 to High(APath) - 1 do
  begin
    LKey := APath[LI];

    if LCurrent.FValues.TryGetValue(LKey, LValue) then
    begin
      if LValue.Kind = tvkTable then
        LCurrent := LValue.AsTable()
      else
        Error('Key ''%s'' is not a table', [LKey]);
    end
    else
    begin
      // Create implicit table
      LNewTable := CreateOwnedTable();
      LCurrent.FValues.Add(LKey, TVdxTomlValue.CreateTable(LNewTable));
      if not LCurrent.FKeyOrder.Contains(LKey) then
        LCurrent.FKeyOrder.Add(LKey);
      LCurrent := LNewTable;
    end;
  end;

  // Set final value
  LKey := APath[High(APath)];

  if LCurrent.FValues.ContainsKey(LKey) then
    Error('Key ''%s'' already defined', [LKey]);

  LCurrent.FValues.Add(LKey, AValue);
  if not LCurrent.FKeyOrder.Contains(LKey) then
    LCurrent.FKeyOrder.Add(LKey);
end;

// -- Comment preservation -----------------------------------------------------

function TVdxToml.GetKeyComment(const AKey: string): string;
begin
  if not FKeyComments.TryGetValue(AKey, Result) then
    Result := '';
end;

procedure TVdxToml.SetKeyComment(const AKey: string; const AValue: string);
begin
  FKeyComments.AddOrSetValue(AKey, AValue);
end;

// -- Public value access ------------------------------------------------------

function TVdxToml.TryGetValue(const AKey: string;
  out AValue: TVdxTomlValue): Boolean;
begin
  Result := FValues.TryGetValue(AKey, AValue);
end;

function TVdxToml.ContainsKey(const AKey: string): Boolean;
begin
  Result := FValues.ContainsKey(AKey);
end;

procedure TVdxToml.RemoveKey(const AKey: string);
begin
  FValues.Remove(AKey);
  FKeyOrder.Remove(AKey);
end;

function TVdxToml.GetString(const AKey: string;
  const ADefault: string): string;
var
  LValue: TVdxTomlValue;
begin
  if FValues.TryGetValue(AKey, LValue) and (LValue.Kind = tvkString) then
    Result := LValue.AsString()
  else
    Result := ADefault;
end;

function TVdxToml.GetInteger(const AKey: string;
  const ADefault: Int64): Int64;
var
  LValue: TVdxTomlValue;
begin
  if FValues.TryGetValue(AKey, LValue) and (LValue.Kind = tvkInteger) then
    Result := LValue.AsInteger()
  else
    Result := ADefault;
end;

function TVdxToml.GetFloat(const AKey: string;
  const ADefault: Double): Double;
var
  LValue: TVdxTomlValue;
begin
  if FValues.TryGetValue(AKey, LValue) then
  begin
    if LValue.Kind = tvkFloat then
      Result := LValue.AsFloat()
    else if LValue.Kind = tvkInteger then
      Result := LValue.AsInteger()
    else
      Result := ADefault;
  end
  else
    Result := ADefault;
end;

function TVdxToml.GetBoolean(const AKey: string;
  const ADefault: Boolean): Boolean;
var
  LValue: TVdxTomlValue;
begin
  if FValues.TryGetValue(AKey, LValue) and (LValue.Kind = tvkBoolean) then
    Result := LValue.AsBoolean()
  else
    Result := ADefault;
end;

function TVdxToml.GetDateTime(const AKey: string;
  const ADefault: TDateTime): TDateTime;
var
  LValue: TVdxTomlValue;
begin
  if FValues.TryGetValue(AKey, LValue) and
     (LValue.Kind in [tvkDateTime, tvkDate, tvkTime]) then
    Result := LValue.AsDateTime()
  else
    Result := ADefault;
end;

procedure TVdxToml.SetString(const AKey: string; const AValue: string);
begin
  FValues.AddOrSetValue(AKey, TVdxTomlValue.CreateString(AValue));
  if not FKeyOrder.Contains(AKey) then
    FKeyOrder.Add(AKey);
end;

procedure TVdxToml.SetInteger(const AKey: string; const AValue: Int64);
begin
  FValues.AddOrSetValue(AKey, TVdxTomlValue.CreateInteger(AValue));
  if not FKeyOrder.Contains(AKey) then
    FKeyOrder.Add(AKey);
end;

procedure TVdxToml.SetFloat(const AKey: string; const AValue: Double);
begin
  FValues.AddOrSetValue(AKey, TVdxTomlValue.CreateFloat(AValue));
  if not FKeyOrder.Contains(AKey) then
    FKeyOrder.Add(AKey);
end;

procedure TVdxToml.SetBoolean(const AKey: string; const AValue: Boolean);
begin
  FValues.AddOrSetValue(AKey, TVdxTomlValue.CreateBoolean(AValue));
  if not FKeyOrder.Contains(AKey) then
    FKeyOrder.Add(AKey);
end;

procedure TVdxToml.SetDateTime(const AKey: string; const AValue: TDateTime);
begin
  FValues.AddOrSetValue(AKey, TVdxTomlValue.CreateDateTime(AValue));
  if not FKeyOrder.Contains(AKey) then
    FKeyOrder.Add(AKey);
end;

function TVdxToml.GetOrCreateTable(const AKey: string): TVdxToml;
var
  LValue:    TVdxTomlValue;
  LNewTable: TVdxToml;
begin
  if FValues.TryGetValue(AKey, LValue) then
  begin
    if LValue.Kind = tvkTable then
      Result := LValue.AsTable()
    else
      raise Exception.CreateFmt('Key ''%s'' is not a table', [AKey]);
  end
  else
  begin
    LNewTable := CreateOwnedTable();
    FValues.Add(AKey, TVdxTomlValue.CreateTable(LNewTable));
    if not FKeyOrder.Contains(AKey) then
      FKeyOrder.Add(AKey);
    Result := LNewTable;
  end;
end;

function TVdxToml.GetOrCreateArray(const AKey: string): TVdxTomlArray;
var
  LValue:    TVdxTomlValue;
  LNewArray: TVdxTomlArray;
begin
  if FValues.TryGetValue(AKey, LValue) then
  begin
    if LValue.Kind = tvkArray then
      Result := LValue.AsArray()
    else
      raise Exception.CreateFmt('Key ''%s'' is not an array', [AKey]);
  end
  else
  begin
    LNewArray := CreateOwnedArray();
    FValues.Add(AKey, TVdxTomlValue.CreateArray(LNewArray));
    if not FKeyOrder.Contains(AKey) then
      FKeyOrder.Add(AKey);
    Result := LNewArray;
  end;
end;

function TVdxToml.GetTable(const AKey: string): TVdxToml;
var
  LValue: TVdxTomlValue;
begin
  if FValues.TryGetValue(AKey, LValue) and (LValue.Kind = tvkTable) then
    Result := LValue.AsTable()
  else
    Result := nil;
end;

function TVdxToml.GetArray(const AKey: string): TVdxTomlArray;
var
  LValue: TVdxTomlValue;
begin
  if FValues.TryGetValue(AKey, LValue) and (LValue.Kind = tvkArray) then
    Result := LValue.AsArray()
  else
    Result := nil;
end;

function TVdxToml.GetKeys(): TArray<string>;
begin
  Result := FKeyOrder.ToArray();
end;

function TVdxToml.GetCount(): Integer;
begin
  Result := FValues.Count;
end;

procedure TVdxToml.SetValue(const AKey: string;
  const AValue: TVdxTomlValue);
begin
  FValues.AddOrSetValue(AKey, AValue);
  if not FKeyOrder.Contains(AKey) then
    FKeyOrder.Add(AKey);
end;

end.
