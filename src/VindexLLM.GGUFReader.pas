{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.GGUFReader;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Generics.Collections,
  VindexLLM.Utils,
  VindexLLM.VirtualFile;

const
  // Error codes — user-facing messages live in VindexLLM.Resources.
  VDX_ERROR_GG_READ_PAST_EOF       = 'GG01';
  VDX_ERROR_GG_BAD_MAGIC           = 'GG02';
  VDX_ERROR_GG_UNSUPPORTED_VERSION = 'GG03';
  VDX_ERROR_GG_UNKNOWN_META_TYPE   = 'GG04';
  VDX_ERROR_GG_NO_DATA_BASE        = 'GG05';
  VDX_ERROR_GG_PARSE_EXCEPTION     = 'GG06';

type

  { TVdxGGMLType }
  // GGML tensor element types. Values from the GGUF spec — do not
  // renumber. Gaps (e.g. 4, 5, 31..33) exist in the upstream spec
  // and are preserved here.
  TVdxGGMLType = (
    gtF32     = 0,
    gtF16     = 1,
    gtQ4_0    = 2,
    gtQ4_1    = 3,
    gtQ5_0    = 6,
    gtQ5_1    = 7,
    gtQ8_0    = 8,
    gtQ8_1    = 9,
    gtQ2_K    = 10,
    gtQ3_K    = 11,
    gtQ4_K    = 12,
    gtQ5_K    = 13,
    gtQ6_K    = 14,
    gtQ8_K    = 15,
    gtIQ2_XXS = 16,
    gtIQ2_XS  = 17,
    gtIQ3_XXS = 18,
    gtIQ1_S   = 19,
    gtIQ4_NL  = 20,
    gtIQ3_S   = 21,
    gtIQ2_S   = 22,
    gtIQ4_XS  = 23,
    gtI8      = 24,
    gtI16     = 25,
    gtI32     = 26,
    gtI64     = 27,
    gtF64     = 28,
    gtIQ1_M   = 29,
    gtBF16    = 30,
    gtTQ1_0   = 34,
    gtTQ2_0   = 35,
    gtMXFP4   = 39
  );

  { TVdxGGUFMetaType }
  // Metadata value type tags from the GGUF spec. Stored as UInt32
  // in the file, mapped to this enum on parse.
  TVdxGGUFMetaType = (
    gmtUInt8   = 0,
    gmtInt8    = 1,
    gmtUInt16  = 2,
    gmtInt16   = 3,
    gmtUInt32  = 4,
    gmtInt32   = 5,
    gmtFloat32 = 6,
    gmtBool    = 7,
    gmtString  = 8,
    gmtArray   = 9,
    gmtUInt64  = 10,
    gmtInt64   = 11,
    gmtFloat64 = 12
  );

  { TVdxGGUFMetaValue }
  // Normalized metadata value. Only the fields relevant to ValueType
  // carry meaningful data; the rest are zero. Arrays carry their
  // element type separately so nested typing is preserved.
  TVdxGGUFMetaValue = record
    ValueType: TVdxGGUFMetaType;
    AsUInt64: UInt64;
    AsInt64: Int64;
    AsFloat64: Double;
    AsBool: Boolean;
    AsString: string;
    ArrayType: TVdxGGUFMetaType;
    ArrayItems: TArray<TVdxGGUFMetaValue>;
  end;

  { TVdxGGUFTensorInfo }
  // Per-tensor header parsed from the tensor-info block. DataOffset
  // is relative to the tensor-data base (computed in ComputeTensorDataBase),
  // not to file start.
  TVdxGGUFTensorInfo = record
    TensorName: string;
    NumDimensions: UInt32;
    Dimensions: TArray<UInt64>;
    TensorType: TVdxGGMLType;
    DataOffset: UInt64;
  end;

  { TVdxGGUFReader }
  // Parses a GGUF file into indexed metadata + tensor info. The file
  // is memory-mapped via a contained TVdxVirtualFile<Byte>; tensor
  // data is accessed as zero-copy pointers into that mapping.
  // Consumers call Open, inspect metadata / tensor info, call
  // GetTensorDataPtr for raw tensor bytes, then Close (or Free).
  // Pointers returned by GetTensorDataPtr are valid only between
  // Open and Close of the same instance.
  TVdxGGUFReader = class(TVdxBaseObject)
  private
    FFile: TVdxVirtualFile<Byte>;
    FBasePtr: PByte;
    FFileSize: UInt64;

    FVersion: UInt32;
    FTensorCount: UInt64;
    FMetadataKVCount: UInt64;
    FAlignment: UInt32;

    FMetadata: TDictionary<string, TVdxGGUFMetaValue>;
    FTensors: TDictionary<string, TVdxGGUFTensorInfo>;
    FTensorList: TList<TVdxGGUFTensorInfo>;

    FTensorDataBase: PByte;
    FCursor: PByte;
    FParseError: Boolean;

    // Cursor bounds check. Sets FParseError and logs GG01 on EOF.
    // Does not raise — read helpers check FParseError after calling.
    procedure CheckCursor(const ASize: NativeUInt);

    // Typed reads. Each advances FCursor by its element size on
    // success; on FParseError already set, each returns a zero-valued
    // result without advancing further.
    function ReadUInt8(): UInt8;
    function ReadInt8(): Int8;
    function ReadUInt16(): UInt16;
    function ReadInt16(): Int16;
    function ReadUInt32(): UInt32;
    function ReadInt32(): Int32;
    function ReadUInt64(): UInt64;
    function ReadInt64(): Int64;
    function ReadFloat32(): Single;
    function ReadFloat64(): Double;
    function ReadBool(): Boolean;
    function ReadGGUFString(): string;
    function ReadMetaValue(const AType: TVdxGGUFMetaType): TVdxGGUFMetaValue;

    // Parse stages. Each early-exits if FParseError is already set.
    procedure ParseHeader();
    procedure ParseMetadata();
    procedure ParseTensorInfos();
    procedure ComputeTensorDataBase();

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Opens a GGUF file, memory-maps it, and parses header +
    // metadata + tensor-info sections. Any failure logs to FErrors
    // and returns False — call HasFatal on GetErrors to distinguish.
    // Errors from the underlying TVdxVirtualFile<Byte> propagate
    // directly via the shared error buffer.
    function  Open(const AFilePath: string): Boolean;

    // Idempotent. Invalidates any pointers previously returned by
    // GetTensorDataPtr.
    procedure Close();

    // Header info.
    function GetVersion(): UInt32;
    function GetTensorCount(): UInt64;
    function GetMetadataCount(): UInt64;
    function GetAlignment(): UInt32;
    function GetFileSize(): UInt64;

    // True between a successful Open and the next Close (or Free).
    // Cheap — tests whether the underlying memory-map is still live.
    function IsOpen(): Boolean;

    // Metadata access. HasMetadata is a cheap pre-check. GetMetadata
    // returns True + populates AValue on hit, False + leaves AValue
    // zero-initialized on miss — missing keys are not errors, callers
    // decide significance. The typed convenience accessors return
    // ADefault silently on miss for the same reason.
    function HasMetadata(const AKey: string): Boolean;
    function GetMetadata(const AKey: string;
      out AValue: TVdxGGUFMetaValue): Boolean;
    function GetMetadataString(const AKey: string;
      const ADefault: string = ''): string;
    function GetMetadataUInt64(const AKey: string;
      const ADefault: UInt64 = 0): UInt64;
    function GetMetadataUInt32(const AKey: string;
      const ADefault: UInt32 = 0): UInt32;
    function GetMetadataFloat32(const AKey: string;
      const ADefault: Single = 0.0): Single;
    function GetMetadataKeys(): TArray<string>;

    // Tensor access. HasTensor is a cheap pre-check. GetTensorInfo
    // follows the same out-param + Boolean convention as GetMetadata.
    // GetTensorDataPtr returns nil + logs GG05 on miss or when the
    // reader is not open; the returned PByte is a pointer into the
    // memory-mapped file and is valid only while the reader stays
    // open.
    function HasTensor(const ATensorName: string): Boolean;
    function GetTensorInfo(const ATensorName: string;
      out AInfo: TVdxGGUFTensorInfo): Boolean;
    function GetTensorDataPtr(const ATensorName: string): PByte;

    // Tensors in the order they appear in the GGUF tensor-info
    // block. Iteration preserves file order — useful for diagnostics
    // and for consumers that don't have a predefined name list.
    function GetTensorList(): TList<TVdxGGUFTensorInfo>;
  end;

// Pure helpers — no error surface.
function VdxGGMLTypeName(const AType: TVdxGGMLType): string;
function VdxGGMLTypeSize(const AType: TVdxGGMLType): UInt64;
function VdxGGMLTensorBytes(const AType: TVdxGGMLType;
  const ADim0: UInt64; const ADim1: UInt64): UInt64;

implementation

uses
  System.Classes,
  VindexLLM.Resources;

const
  CGGUF_MAGIC          = $46554747;  // 'GGUF' as little-endian UInt32
  CGGUF_DEFAULT_ALIGN  = 32;

{ Pure helpers }

function VdxGGMLTypeName(const AType: TVdxGGMLType): string;
begin
  case Ord(AType) of
    0:  Result := 'F32';
    1:  Result := 'F16';
    2:  Result := 'Q4_0';
    3:  Result := 'Q4_1';
    6:  Result := 'Q5_0';
    7:  Result := 'Q5_1';
    8:  Result := 'Q8_0';
    9:  Result := 'Q8_1';
    10: Result := 'Q2_K';
    11: Result := 'Q3_K';
    12: Result := 'Q4_K';
    13: Result := 'Q5_K';
    14: Result := 'Q6_K';
    15: Result := 'Q8_K';
    16: Result := 'IQ2_XXS';
    17: Result := 'IQ2_XS';
    18: Result := 'IQ3_XXS';
    19: Result := 'IQ1_S';
    20: Result := 'IQ4_NL';
    21: Result := 'IQ3_S';
    22: Result := 'IQ2_S';
    23: Result := 'IQ4_XS';
    24: Result := 'I8';
    25: Result := 'I16';
    26: Result := 'I32';
    27: Result := 'I64';
    28: Result := 'F64';
    29: Result := 'IQ1_M';
    30: Result := 'BF16';
    34: Result := 'TQ1_0';
    35: Result := 'TQ2_0';
    39: Result := 'MXFP4';
  else
    Result := Format('Unknown(%d)', [Ord(AType)]);
  end;
end;

// Returns bytes per element for non-quantized types; returns 0 for
// quantized types (use VdxGGMLTensorBytes for those — block-based).
function VdxGGMLTypeSize(const AType: TVdxGGMLType): UInt64;
begin
  case Ord(AType) of
    0:  Result := 4;  // F32
    1:  Result := 2;  // F16
    24: Result := 1;  // I8
    25: Result := 2;  // I16
    26: Result := 4;  // I32
    27: Result := 8;  // I64
    28: Result := 8;  // F64
    30: Result := 2;  // BF16
  else
    Result := 0;  // quantized — block-based, not per-element
  end;
end;

// Total byte size of a 2D tensor. Non-quantized: elements * element
// size. Quantized: block count * block byte size per the GGML spec.
// Returns 0 for unsupported quantization types.
function VdxGGMLTensorBytes(const AType: TVdxGGMLType;
  const ADim0: UInt64; const ADim1: UInt64): UInt64;
var
  LTotalElements: UInt64;
  LNumBlocks: UInt64;
  LElemSize: UInt64;
begin
  LTotalElements := ADim0 * ADim1;

  case Ord(AType) of
    2: // Q4_0: block_size=32, block_bytes=18
    begin
      LNumBlocks := LTotalElements div 32;
      Result := LNumBlocks * 18;
    end;

    3: // Q4_1: block_size=32, block_bytes=20
    begin
      LNumBlocks := LTotalElements div 32;
      Result := LNumBlocks * 20;
    end;

    8: // Q8_0: block_size=32, block_bytes=34
    begin
      LNumBlocks := LTotalElements div 32;
      Result := LNumBlocks * 34;
    end;

    12: // Q4_K: block_size=256, block_bytes=144
    begin
      LNumBlocks := LTotalElements div 256;
      Result := LNumBlocks * 144;
    end;

    13: // Q5_K: block_size=256, block_bytes=176
    begin
      LNumBlocks := LTotalElements div 256;
      Result := LNumBlocks * 176;
    end;

    14: // Q6_K: block_size=256, block_bytes=210
    begin
      LNumBlocks := LTotalElements div 256;
      Result := LNumBlocks * 210;
    end;

  else
    // Non-quantized: per-element size
    LElemSize := VdxGGMLTypeSize(AType);
    if LElemSize > 0 then
      Result := LTotalElements * LElemSize
    else
      Result := 0;  // unsupported quantization type
  end;
end;

{ TVdxGGUFReader }

constructor TVdxGGUFReader.Create();
begin
  inherited Create();

  // Own the file-map component, and share our error buffer with it
  // so its errors land in our shared list (principle #10). FOwnsErrors
  // on the child becomes False as a result of SetErrors.
  FFile := TVdxVirtualFile<Byte>.Create();
  FFile.SetErrors(FErrors);

  FBasePtr         := nil;
  FFileSize        := 0;
  FVersion         := 0;
  FTensorCount     := 0;
  FMetadataKVCount := 0;
  FAlignment       := CGGUF_DEFAULT_ALIGN;
  FTensorDataBase  := nil;
  FCursor          := nil;
  FParseError      := False;

  FMetadata   := TDictionary<string, TVdxGGUFMetaValue>.Create();
  FTensors    := TDictionary<string, TVdxGGUFTensorInfo>.Create();
  FTensorList := TList<TVdxGGUFTensorInfo>.Create();
end;

destructor TVdxGGUFReader.Destroy();
begin
  Close();
  FreeAndNil(FTensorList);
  FreeAndNil(FTensors);
  FreeAndNil(FMetadata);
  FreeAndNil(FFile);
  inherited Destroy();
end;

procedure TVdxGGUFReader.Close();
begin
  // File-map release delegated to the contained TVdxVirtualFile —
  // it handles the Unmap / CloseHandle sequence safely and is
  // itself idempotent.
  if FFile <> nil then
    FFile.Close();

  FBasePtr         := nil;
  FFileSize        := 0;
  FVersion         := 0;
  FTensorCount     := 0;
  FMetadataKVCount := 0;
  FAlignment       := CGGUF_DEFAULT_ALIGN;
  FTensorDataBase  := nil;
  FCursor          := nil;
  FParseError      := False;

  FMetadata.Clear();
  FTensors.Clear();
  FTensorList.Clear();
end;

function TVdxGGUFReader.Open(const AFilePath: string): Boolean;
begin
  Result := False;

  // Idempotent re-open — release any prior state, including the
  // shared-errors path. Caller inherits whatever errors accrue.
  Close();

  Status('Opening GGUF file: %s', [AFilePath]);

  // Hand the path to the file-map component. On failure, its error
  // (VF02/VF04/VF06) lands directly in our shared buffer — no need
  // to log a separate GG-level error.
  if not FFile.Open(AFilePath) then
    Exit;

  FBasePtr  := PByte(FFile.Memory);
  FFileSize := FFile.Size;
  FCursor   := FBasePtr;

  // Parse stages. Each checks FParseError at entry and bails; the
  // try/except is a final safety net for any unexpected RTL raise
  // (dict allocation, UTF-8 decode, etc.).
  try
    ParseHeader();
    ParseMetadata();
    ParseTensorInfos();
    ComputeTensorDataBase();
  except
    on E: Exception do
    begin
      FErrors.Add(esFatal, VDX_ERROR_GG_PARSE_EXCEPTION,
        RSGGParseException, [E.Message]);
      FParseError := True;
    end;
  end;

  if FParseError then
  begin
    Close();
    Exit;
  end;

  Status('GGUF parse complete: %d metadata entries, %d tensors',
    [FMetadata.Count, FTensors.Count]);

  Result := True;
end;

{ Cursor bounds check — logs + sets flag instead of raising }

procedure TVdxGGUFReader.CheckCursor(const ASize: NativeUInt);
var
  LOffset: NativeUInt;
begin
  if FParseError then Exit;
  LOffset := NativeUInt(FCursor) - NativeUInt(FBasePtr);
  if LOffset + ASize > FFileSize then
  begin
    FErrors.Add(esFatal, VDX_ERROR_GG_READ_PAST_EOF,
      RSGGReadPastEOF, [LOffset, ASize, FFileSize]);
    FParseError := True;
  end;
end;

{ Typed read helpers — each advances FCursor on success; silently
  returns zero-valued result if FParseError is already set or if
  CheckCursor flips it. }

function TVdxGGUFReader.ReadUInt8(): UInt8;
begin
  Result := 0;
  if FParseError then Exit;
  CheckCursor(1);
  if FParseError then Exit;
  Result := PByte(FCursor)^;
  Inc(FCursor, 1);
end;

function TVdxGGUFReader.ReadInt8(): Int8;
begin
  Result := 0;
  if FParseError then Exit;
  CheckCursor(1);
  if FParseError then Exit;
  Result := PShortInt(FCursor)^;
  Inc(FCursor, 1);
end;

function TVdxGGUFReader.ReadUInt16(): UInt16;
begin
  Result := 0;
  if FParseError then Exit;
  CheckCursor(2);
  if FParseError then Exit;
  Result := PWord(FCursor)^;
  Inc(FCursor, 2);
end;

function TVdxGGUFReader.ReadInt16(): Int16;
begin
  Result := 0;
  if FParseError then Exit;
  CheckCursor(2);
  if FParseError then Exit;
  Result := PSmallInt(FCursor)^;
  Inc(FCursor, 2);
end;

function TVdxGGUFReader.ReadUInt32(): UInt32;
begin
  Result := 0;
  if FParseError then Exit;
  CheckCursor(4);
  if FParseError then Exit;
  Result := PCardinal(FCursor)^;
  Inc(FCursor, 4);
end;

function TVdxGGUFReader.ReadInt32(): Int32;
begin
  Result := 0;
  if FParseError then Exit;
  CheckCursor(4);
  if FParseError then Exit;
  Result := PInteger(FCursor)^;
  Inc(FCursor, 4);
end;

function TVdxGGUFReader.ReadUInt64(): UInt64;
begin
  Result := 0;
  if FParseError then Exit;
  CheckCursor(8);
  if FParseError then Exit;
  Result := PUInt64(FCursor)^;
  Inc(FCursor, 8);
end;

function TVdxGGUFReader.ReadInt64(): Int64;
begin
  Result := 0;
  if FParseError then Exit;
  CheckCursor(8);
  if FParseError then Exit;
  Result := PInt64(FCursor)^;
  Inc(FCursor, 8);
end;

function TVdxGGUFReader.ReadFloat32(): Single;
begin
  Result := 0.0;
  if FParseError then Exit;
  CheckCursor(4);
  if FParseError then Exit;
  Result := PSingle(FCursor)^;
  Inc(FCursor, 4);
end;

function TVdxGGUFReader.ReadFloat64(): Double;
begin
  Result := 0.0;
  if FParseError then Exit;
  CheckCursor(8);
  if FParseError then Exit;
  Result := PDouble(FCursor)^;
  Inc(FCursor, 8);
end;

function TVdxGGUFReader.ReadBool(): Boolean;
begin
  Result := False;
  if FParseError then Exit;
  CheckCursor(1);
  if FParseError then Exit;
  Result := PByte(FCursor)^ <> 0;
  Inc(FCursor, 1);
end;

function TVdxGGUFReader.ReadGGUFString(): string;
var
  LLen: UInt64;
  LBytes: TBytes;
begin
  Result := '';
  if FParseError then Exit;

  LLen := ReadUInt64();
  if FParseError then Exit;
  if LLen = 0 then Exit;

  // Bounds-check the payload before allocating. A corrupted file
  // could claim a huge length; CheckCursor flips FParseError here
  // instead of attempting the allocation.
  CheckCursor(NativeUInt(LLen));
  if FParseError then Exit;

  SetLength(LBytes, LLen);
  Move(FCursor^, LBytes[0], LLen);
  Inc(FCursor, LLen);

  Result := TEncoding.UTF8.GetString(LBytes);
end;

function TVdxGGUFReader.ReadMetaValue(
  const AType: TVdxGGUFMetaType): TVdxGGUFMetaValue;
var
  LArrayLen: UInt64;
  LI: UInt64;
begin
  Result := Default(TVdxGGUFMetaValue);
  Result.ValueType := AType;

  if FParseError then Exit;

  case AType of
    gmtUInt8:   Result.AsUInt64  := ReadUInt8();
    gmtInt8:    Result.AsInt64   := ReadInt8();
    gmtUInt16:  Result.AsUInt64  := ReadUInt16();
    gmtInt16:   Result.AsInt64   := ReadInt16();
    gmtUInt32:  Result.AsUInt64  := ReadUInt32();
    gmtInt32:   Result.AsInt64   := ReadInt32();
    gmtFloat32: Result.AsFloat64 := ReadFloat32();
    gmtBool:    Result.AsBool    := ReadBool();
    gmtString:  Result.AsString  := ReadGGUFString();
    gmtUInt64:  Result.AsUInt64  := ReadUInt64();
    gmtInt64:   Result.AsInt64   := ReadInt64();
    gmtFloat64: Result.AsFloat64 := ReadFloat64();

    gmtArray:
    begin
      Result.ArrayType := TVdxGGUFMetaType(ReadUInt32());
      if FParseError then Exit;

      LArrayLen := ReadUInt64();
      if FParseError then Exit;

      SetLength(Result.ArrayItems, LArrayLen);
      if LArrayLen > 0 then
      begin
        for LI := 0 to LArrayLen - 1 do
        begin
          if FParseError then Exit;
          Result.ArrayItems[LI] := ReadMetaValue(Result.ArrayType);
        end;
      end;
    end;
  else
    // Unknown type tag — file is corrupt or a newer GGUF version.
    FErrors.Add(esFatal, VDX_ERROR_GG_UNKNOWN_META_TYPE,
      RSGGUnknownMetaType, [Ord(AType)]);
    FParseError := True;
  end;
end;

{ Parse stages }

procedure TVdxGGUFReader.ParseHeader();
var
  LMagic: UInt32;
begin
  if FParseError then Exit;

  LMagic := ReadUInt32();
  if FParseError then Exit;

  if LMagic <> CGGUF_MAGIC then
  begin
    FErrors.Add(esFatal, VDX_ERROR_GG_BAD_MAGIC,
      RSGGBadMagic, [CGGUF_MAGIC, LMagic]);
    FParseError := True;
    Exit;
  end;

  FVersion := ReadUInt32();
  if FParseError then Exit;

  if FVersion < 2 then
  begin
    FErrors.Add(esFatal, VDX_ERROR_GG_UNSUPPORTED_VERSION,
      RSGGUnsupportedVersion, [FVersion]);
    FParseError := True;
    Exit;
  end;

  FTensorCount     := ReadUInt64();
  FMetadataKVCount := ReadUInt64();

  Status('GGUF v%d — %d tensors, %d metadata entries',
    [FVersion, FTensorCount, FMetadataKVCount]);
end;

procedure TVdxGGUFReader.ParseMetadata();
var
  LI: UInt64;
  LKey: string;
  LValueType: TVdxGGUFMetaType;
  LValue: TVdxGGUFMetaValue;
begin
  if FParseError then Exit;
  if FMetadataKVCount = 0 then Exit;

  for LI := 0 to FMetadataKVCount - 1 do
  begin
    if FParseError then Exit;

    LKey       := ReadGGUFString();
    LValueType := TVdxGGUFMetaType(ReadUInt32());
    LValue     := ReadMetaValue(LValueType);
    if FParseError then Exit;

    FMetadata.AddOrSetValue(LKey, LValue);

    // Pick up the tensor-data alignment override if present. Default
    // is 32 if the key is absent or zero.
    if SameText(LKey, 'general.alignment') then
    begin
      FAlignment := UInt32(LValue.AsUInt64);
      if FAlignment = 0 then
        FAlignment := CGGUF_DEFAULT_ALIGN;
    end;
  end;

  Status('Metadata parsed: %d entries', [FMetadata.Count]);
end;

procedure TVdxGGUFReader.ParseTensorInfos();
var
  LI: UInt64;
  LJ: UInt32;
  LInfo: TVdxGGUFTensorInfo;
begin
  if FParseError then Exit;
  if FTensorCount = 0 then Exit;

  for LI := 0 to FTensorCount - 1 do
  begin
    if FParseError then Exit;

    LInfo := Default(TVdxGGUFTensorInfo);

    LInfo.TensorName    := ReadGGUFString();
    LInfo.NumDimensions := ReadUInt32();
    if FParseError then Exit;

    SetLength(LInfo.Dimensions, LInfo.NumDimensions);
    if LInfo.NumDimensions > 0 then
    begin
      for LJ := 0 to LInfo.NumDimensions - 1 do
      begin
        LInfo.Dimensions[LJ] := ReadUInt64();
        if FParseError then Exit;
      end;
    end;

    LInfo.TensorType := TVdxGGMLType(ReadUInt32());
    LInfo.DataOffset := ReadUInt64();
    if FParseError then Exit;

    FTensors.AddOrSetValue(LInfo.TensorName, LInfo);
    FTensorList.Add(LInfo);
  end;

  Status('Tensor infos parsed: %d tensors', [FTensors.Count]);
end;

procedure TVdxGGUFReader.ComputeTensorDataBase();
var
  LOffset: UInt64;
  LAligned: UInt64;
begin
  if FParseError then Exit;

  // Cursor is sitting right after the last tensor-info record.
  // Tensor data starts at the next FAlignment boundary.
  LOffset := UInt64(FCursor) - UInt64(FBasePtr);
  LAligned := LOffset + (UInt64(FAlignment) -
    (LOffset mod UInt64(FAlignment))) mod UInt64(FAlignment);

  FTensorDataBase := FBasePtr + LAligned;

  Status('Tensor data base at file offset %d ($%x), alignment=%d',
    [LAligned, LAligned, FAlignment]);
end;

{ Public API — header info }

function TVdxGGUFReader.GetVersion(): UInt32;
begin
  Result := FVersion;
end;

function TVdxGGUFReader.GetTensorCount(): UInt64;
begin
  Result := FTensorCount;
end;

function TVdxGGUFReader.GetMetadataCount(): UInt64;
begin
  Result := FMetadataKVCount;
end;

function TVdxGGUFReader.GetAlignment(): UInt32;
begin
  Result := FAlignment;
end;

function TVdxGGUFReader.GetFileSize(): UInt64;
begin
  Result := FFileSize;
end;

function TVdxGGUFReader.IsOpen(): Boolean;
begin
  // FBasePtr is nil before Open and cleared by Close — non-nil
  // exactly when the memory-map is live.
  Result := FBasePtr <> nil;
end;

{ Public API — metadata access }

function TVdxGGUFReader.HasMetadata(const AKey: string): Boolean;
begin
  Result := FMetadata.ContainsKey(AKey);
end;

function TVdxGGUFReader.GetMetadata(const AKey: string;
  out AValue: TVdxGGUFMetaValue): Boolean;
begin
  AValue := Default(TVdxGGUFMetaValue);
  Result := FMetadata.TryGetValue(AKey, AValue);
end;

function TVdxGGUFReader.GetMetadataString(const AKey: string;
  const ADefault: string): string;
var
  LValue: TVdxGGUFMetaValue;
begin
  if FMetadata.TryGetValue(AKey, LValue) then
    Result := LValue.AsString
  else
    Result := ADefault;
end;

function TVdxGGUFReader.GetMetadataUInt64(const AKey: string;
  const ADefault: UInt64): UInt64;
var
  LValue: TVdxGGUFMetaValue;
begin
  if FMetadata.TryGetValue(AKey, LValue) then
    Result := LValue.AsUInt64
  else
    Result := ADefault;
end;

function TVdxGGUFReader.GetMetadataUInt32(const AKey: string;
  const ADefault: UInt32): UInt32;
var
  LValue: TVdxGGUFMetaValue;
begin
  if FMetadata.TryGetValue(AKey, LValue) then
    Result := UInt32(LValue.AsUInt64)
  else
    Result := ADefault;
end;

function TVdxGGUFReader.GetMetadataFloat32(const AKey: string;
  const ADefault: Single): Single;
var
  LValue: TVdxGGUFMetaValue;
begin
  if FMetadata.TryGetValue(AKey, LValue) then
    Result := Single(LValue.AsFloat64)
  else
    Result := ADefault;
end;

function TVdxGGUFReader.GetMetadataKeys(): TArray<string>;
var
  LKey: string;
  LIndex: Integer;
begin
  SetLength(Result, FMetadata.Count);
  LIndex := 0;
  for LKey in FMetadata.Keys do
  begin
    Result[LIndex] := LKey;
    Inc(LIndex);
  end;
end;

{ Public API — tensor access }

function TVdxGGUFReader.HasTensor(const ATensorName: string): Boolean;
begin
  Result := FTensors.ContainsKey(ATensorName);
end;

function TVdxGGUFReader.GetTensorInfo(const ATensorName: string;
  out AInfo: TVdxGGUFTensorInfo): Boolean;
begin
  AInfo := Default(TVdxGGUFTensorInfo);
  Result := FTensors.TryGetValue(ATensorName, AInfo);
end;

function TVdxGGUFReader.GetTensorDataPtr(
  const ATensorName: string): PByte;
var
  LInfo: TVdxGGUFTensorInfo;
begin
  Result := nil;

  if (FTensorDataBase = nil) or (FBasePtr = nil) then
  begin
    FErrors.Add(esError, VDX_ERROR_GG_NO_DATA_BASE, RSGGNoDataBase);
    Exit;
  end;

  // Missing tensor is not fatal at the reader layer — caller
  // decides whether it matters.
  if not FTensors.TryGetValue(ATensorName, LInfo) then Exit;

  Result := FTensorDataBase + LInfo.DataOffset;
end;

function TVdxGGUFReader.GetTensorList(): TList<TVdxGGUFTensorInfo>;
begin
  Result := FTensorList;
end;

end.
