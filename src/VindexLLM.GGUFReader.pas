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
  VDX_ERROR_GG_PARSE       = 'GG01';
  VDX_ERROR_GG_BAD_MAGIC   = 'GG02';
  VDX_ERROR_GG_BAD_VERSION = 'GG03';
  VDX_ERROR_GG_META_KEY    = 'GG04';
  VDX_ERROR_GG_TENSOR      = 'GG05';
  VDX_ERROR_GG_NOT_OPEN    = 'GG06';

type

  { TVdxGGMLType }
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
  TVdxGGUFTensorInfo = record
    TensorName: string;
    NumDimensions: UInt32;
    Dimensions: TArray<UInt64>;
    TensorType: TVdxGGMLType;
    DataOffset: UInt64;
  end;

  { TVdxGGUFReader }
  TVdxGGUFReader = class(TVdxBaseObject)
  private
    // Memory-mapped file access
    FVirtualFile: TVdxVirtualFile<Byte>;

    // Parsed header
    FVersion: UInt32;
    FTensorCount: UInt64;
    FMetadataKVCount: UInt64;
    FAlignment: UInt32;

    // Parsed data
    FMetadata: TDictionary<string, TVdxGGUFMetaValue>;
    FTensors: TDictionary<string, TVdxGGUFTensorInfo>;
    FTensorList: TList<TVdxGGUFTensorInfo>;

    // Computed offset where tensor data begins
    FTensorDataBase: PByte;

    // Cursor for sequential parsing
    FCursor: PByte;

    // Internal read helpers
    procedure CheckCursor(const ASize: NativeUInt);
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

    // Internal parsing stages
    procedure ParseHeader();
    procedure ParseMetadata();
    procedure ParseTensorInfos();
    procedure ComputeTensorDataBase();

  public
    constructor Create(); override;
    destructor Destroy(); override;

    procedure SetErrors(const AErrors: TVdxErrors); override;

    // Open and parse a GGUF file (memory-maps entire file)
    function Open(const AFilePath: string): Boolean;

    // Close and unmap
    procedure Close();

    // Header info
    function GetVersion(): UInt32;
    function GetTensorCount(): UInt64;
    function GetMetadataCount(): UInt64;
    function GetAlignment(): UInt32;
    function GetFileSize(): UInt64;

    // Metadata access
    function HasMetadata(const AKey: string): Boolean;
    function GetMetadata(const AKey: string; out AValue: TVdxGGUFMetaValue): Boolean;
    function GetMetadataString(const AKey: string; const ADefault: string = ''): string;
    function GetMetadataUInt64(const AKey: string; const ADefault: UInt64 = 0): UInt64;
    function GetMetadataUInt32(const AKey: string; const ADefault: UInt32 = 0): UInt32;
    function GetMetadataFloat32(const AKey: string; const ADefault: Single = 0.0): Single;
    function GetMetadataKeys(): TArray<string>;

    // Tensor access
    function HasTensor(const ATensorName: string): Boolean;
    function GetTensorInfo(const ATensorName: string; out AInfo: TVdxGGUFTensorInfo): Boolean;
    function GetTensorDataPtr(const ATensorName: string): Pointer;
    function GetTensorList(): TList<TVdxGGUFTensorInfo>;
  end;

function VdxGGMLTypeName(const AType: TVdxGGMLType): string;
function VdxGGMLTypeSize(const AType: TVdxGGMLType): UInt64;
function VdxGGMLTensorBytes(const AType: TVdxGGMLType;
  const ADim0: UInt64; const ADim1: UInt64): UInt64;

implementation

const
  CGGUF_MAGIC          = $46554747; // 'GGUF' as little-endian uint32
  CGGUF_DEFAULT_ALIGN  = 32;

{ Helper functions }

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

// Returns bytes per element for non-quantized types.
// For quantized types, returns 0 (use block-based calculation instead).
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

// Compute total byte size of a 2D tensor for any GGML type.
// For non-quantized types: dim0 * dim1 * element_size.
// For quantized types: block-based calculation.
// Q4_0: 18 bytes per block of 32 elements.
// Q8_0: 34 bytes per block of 32 elements.
function VdxGGMLTensorBytes(const AType: TVdxGGMLType;
  const ADim0: UInt64; const ADim1: UInt64): UInt64;
var
  LTotalElements: UInt64;
  LNumBlocks: UInt64;
  LElemSize: UInt64;
begin
  LTotalElements := ADim0 * ADim1;

  case Ord(AType) of
    2: // Q4_0: block_size=32, block_bytes=18 (2 scale + 16 qs)
    begin
      LNumBlocks := LTotalElements div 32;
      Result := LNumBlocks * 18;
    end;

    3: // Q4_1: block_size=32, block_bytes=20 (2 scale + 2 min + 16 qs)
    begin
      LNumBlocks := LTotalElements div 32;
      Result := LNumBlocks * 20;
    end;

    8: // Q8_0: block_size=32, block_bytes=34 (2 scale + 32 qs)
    begin
      LNumBlocks := LTotalElements div 32;
      Result := LNumBlocks * 34;
    end;

    12: // Q4_K: block_size=256, block_bytes=144 (2+2 scale/min + 12 sub-scales + 128 qs)
    begin
      LNumBlocks := LTotalElements div 256;
      Result := LNumBlocks * 144;
    end;

    13: // Q5_K: block_size=256, block_bytes=176 (2+2 + 12 + 32 high-bits + 128 qs)
    begin
      LNumBlocks := LTotalElements div 256;
      Result := LNumBlocks * 176;
    end;

    14: // Q6_K: block_size=256, block_bytes=210 (128 ql + 64 qh + 16 scales + 2 d)
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

  FVirtualFile := TVdxVirtualFile<Byte>.Create();
  FVirtualFile.SetErrors(FErrors);

  FVersion := 0;
  FTensorCount := 0;
  FMetadataKVCount := 0;
  FAlignment := CGGUF_DEFAULT_ALIGN;
  FTensorDataBase := nil;
  FCursor := nil;

  FMetadata := TDictionary<string, TVdxGGUFMetaValue>.Create();
  FTensors := TDictionary<string, TVdxGGUFTensorInfo>.Create();
  FTensorList := TList<TVdxGGUFTensorInfo>.Create();
end;

destructor TVdxGGUFReader.Destroy();
begin
  Close();

  FreeAndNil(FTensorList);
  FreeAndNil(FTensors);
  FreeAndNil(FMetadata);
  FreeAndNil(FVirtualFile);

  inherited Destroy();
end;

procedure TVdxGGUFReader.SetErrors(const AErrors: TVdxErrors);
begin
  inherited SetErrors(AErrors);
  if Assigned(FVirtualFile) then
    FVirtualFile.SetErrors(AErrors);
end;

procedure TVdxGGUFReader.Close();
begin
  FVirtualFile.Close();

  FVersion := 0;
  FTensorCount := 0;
  FMetadataKVCount := 0;
  FAlignment := CGGUF_DEFAULT_ALIGN;
  FTensorDataBase := nil;
  FCursor := nil;

  FMetadata.Clear();
  FTensors.Clear();
  FTensorList.Clear();
end;

function TVdxGGUFReader.Open(const AFilePath: string): Boolean;
begin
  Result := False;
  // Close any previously opened file
  Close();

  Status('Opening GGUF file: %s', [AFilePath]);

  // Memory-map the file via TVdxVirtualFile
  if not FVirtualFile.Open(AFilePath) then
  begin
    Status('Failed to open file: %s', [AFilePath]);
    Exit;
  end;

  Status('File size: %d bytes (%.2f GB)',
    [FVirtualFile.Size, FVirtualFile.Size / (1024.0 * 1024.0 * 1024.0)]);
  Status('File mapped into memory at $%p', [FVirtualFile.Memory]);

  // Initialize cursor to start of file
  FCursor := PByte(FVirtualFile.Memory);

  try
    // Parse the file sequentially
    ParseHeader();
    if FErrors.HasErrors() then
    begin
      Close();
      Exit;
    end;

    ParseMetadata();
    if FErrors.HasErrors() then
    begin
      Close();
      Exit;
    end;

    ParseTensorInfos();
    ComputeTensorDataBase();

    Status('GGUF parse complete: %d metadata entries, %d tensors',
      [FMetadata.Count, FTensors.Count]);

    Result := True;
  except
    on E: Exception do
    begin
      FErrors.Add(esFatal, VDX_ERROR_GG_PARSE, 'GGUF parse error: %s', [E.Message]);
      Close();
    end;
  end;
end;

{ Cursor bounds checking }

procedure TVdxGGUFReader.CheckCursor(const ASize: NativeUInt);
var
  LOffset: NativeUInt;
begin
  LOffset := NativeUInt(FCursor) - NativeUInt(FVirtualFile.Memory);
  if LOffset + ASize > FVirtualFile.Size then
    raise Exception.CreateFmt(
      'GGUF read past end of file: offset=%d, need=%d, filesize=%d',
      [LOffset, ASize, FVirtualFile.Size]);
end;

{ Cursor read helpers — all advance FCursor }

function TVdxGGUFReader.ReadUInt8(): UInt8;
begin
  CheckCursor(1);
  Result := PByte(FCursor)^;
  Inc(FCursor, 1);
end;

function TVdxGGUFReader.ReadInt8(): Int8;
begin
  CheckCursor(1);
  Result := PShortInt(FCursor)^;
  Inc(FCursor, 1);
end;

function TVdxGGUFReader.ReadUInt16(): UInt16;
begin
  CheckCursor(2);
  Result := PWord(FCursor)^;
  Inc(FCursor, 2);
end;

function TVdxGGUFReader.ReadInt16(): Int16;
begin
  CheckCursor(2);
  Result := PSmallInt(FCursor)^;
  Inc(FCursor, 2);
end;

function TVdxGGUFReader.ReadUInt32(): UInt32;
begin
  CheckCursor(4);
  Result := PCardinal(FCursor)^;
  Inc(FCursor, 4);
end;

function TVdxGGUFReader.ReadInt32(): Int32;
begin
  CheckCursor(4);
  Result := PInteger(FCursor)^;
  Inc(FCursor, 4);
end;

function TVdxGGUFReader.ReadUInt64(): UInt64;
begin
  CheckCursor(8);
  Result := PUInt64(FCursor)^;
  Inc(FCursor, 8);
end;

function TVdxGGUFReader.ReadInt64(): Int64;
begin
  CheckCursor(8);
  Result := PInt64(FCursor)^;
  Inc(FCursor, 8);
end;

function TVdxGGUFReader.ReadFloat32(): Single;
begin
  CheckCursor(4);
  Result := PSingle(FCursor)^;
  Inc(FCursor, 4);
end;

function TVdxGGUFReader.ReadFloat64(): Double;
begin
  CheckCursor(8);
  Result := PDouble(FCursor)^;
  Inc(FCursor, 8);
end;

function TVdxGGUFReader.ReadBool(): Boolean;
begin
  CheckCursor(1);
  Result := PByte(FCursor)^ <> 0;
  Inc(FCursor, 1);
end;

function TVdxGGUFReader.ReadGGUFString(): string;
var
  LLen: UInt64;
  LBytes: TBytes;
begin
  LLen := ReadUInt64();

  if LLen = 0 then
    Exit('');

  // Sanity check: string length should not exceed remaining file
  CheckCursor(NativeUInt(LLen));

  SetLength(LBytes, LLen);
  Move(FCursor^, LBytes[0], LLen);
  Inc(FCursor, LLen);

  Result := TEncoding.UTF8.GetString(LBytes);
end;

function TVdxGGUFReader.ReadMetaValue(const AType: TVdxGGUFMetaType): TVdxGGUFMetaValue;
var
  LArrayLen: UInt64;
  LI: UInt64;
begin
  Result := Default(TVdxGGUFMetaValue);
  Result.ValueType := AType;

  case AType of
    gmtUInt8:
      Result.AsUInt64 := ReadUInt8();

    gmtInt8:
      Result.AsInt64 := ReadInt8();

    gmtUInt16:
      Result.AsUInt64 := ReadUInt16();

    gmtInt16:
      Result.AsInt64 := ReadInt16();

    gmtUInt32:
      Result.AsUInt64 := ReadUInt32();

    gmtInt32:
      Result.AsInt64 := ReadInt32();

    gmtFloat32:
      Result.AsFloat64 := ReadFloat32();

    gmtBool:
      Result.AsBool := ReadBool();

    gmtString:
      Result.AsString := ReadGGUFString();

    gmtUInt64:
      Result.AsUInt64 := ReadUInt64();

    gmtInt64:
      Result.AsInt64 := ReadInt64();

    gmtFloat64:
      Result.AsFloat64 := ReadFloat64();

    gmtArray:
    begin
      // Read array element type and length
      Result.ArrayType := TVdxGGUFMetaType(ReadUInt32());
      LArrayLen := ReadUInt64();
      SetLength(Result.ArrayItems, LArrayLen);

      // Read each element
      for LI := 0 to LArrayLen - 1 do
        Result.ArrayItems[LI] := ReadMetaValue(Result.ArrayType);
    end;
  else
    raise Exception.CreateFmt('Unknown GGUF metadata type: %d', [Ord(AType)]);
  end;
end;

{ Parsing stages }

procedure TVdxGGUFReader.ParseHeader();
var
  LMagic: UInt32;
begin
  // Read and validate magic number
  LMagic := ReadUInt32();
  if LMagic <> CGGUF_MAGIC then
  begin
    FErrors.Add(esFatal, VDX_ERROR_GG_BAD_MAGIC,
      'Invalid GGUF magic: expected $%08X, got $%08X', [CGGUF_MAGIC, LMagic]);
    Exit;
  end;

  // Read version
  FVersion := ReadUInt32();
  if FVersion < 2 then
  begin
    FErrors.Add(esFatal, VDX_ERROR_GG_BAD_VERSION,
      'Unsupported GGUF version: %d (need >= 2)', [FVersion]);
    Exit;
  end;

  // Read counts
  FTensorCount := ReadUInt64();
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
  Status('Parsing %d metadata entries...', [FMetadataKVCount]);

  for LI := 0 to FMetadataKVCount - 1 do
  begin
    // Read key string
    LKey := ReadGGUFString();

    // Read value type
    LValueType := TVdxGGUFMetaType(ReadUInt32());

    // Read value
    LValue := ReadMetaValue(LValueType);

    // Store in dictionary
    FMetadata.AddOrSetValue(LKey, LValue);

    // Check for alignment override
    if SameText(LKey, 'general.alignment') then
    begin
      FAlignment := UInt32(LValue.AsUInt64);
      if FAlignment = 0 then
        FAlignment := CGGUF_DEFAULT_ALIGN;
      Status('  Alignment: %d', [FAlignment]);
    end;

    // Log selected important metadata
    if SameText(LKey, 'general.architecture') then
      Status('  Architecture: %s', [LValue.AsString])
    else if SameText(LKey, 'general.name') then
      Status('  Model name: %s', [LValue.AsString])
    else if SameText(LKey, 'general.file_type') then
      Status('  File type: %d', [LValue.AsUInt64]);
  end;

  Status('Metadata parsing complete (%d entries)', [FMetadata.Count]);
end;

procedure TVdxGGUFReader.ParseTensorInfos();
var
  LI: UInt64;
  LJ: UInt32;
  LInfo: TVdxGGUFTensorInfo;
  LDimsStr: string;
begin
  Status('Parsing %d tensor info entries...', [FTensorCount]);

  for LI := 0 to FTensorCount - 1 do
  begin
    // Read tensor name
    LInfo.TensorName := ReadGGUFString();

    // Read number of dimensions
    LInfo.NumDimensions := ReadUInt32();

    // Read each dimension
    SetLength(LInfo.Dimensions, LInfo.NumDimensions);
    for LJ := 0 to LInfo.NumDimensions - 1 do
      LInfo.Dimensions[LJ] := ReadUInt64();

    // Read tensor type
    LInfo.TensorType := TVdxGGMLType(ReadUInt32());

    // Read data offset (relative to tensor_data start)
    LInfo.DataOffset := ReadUInt64();

    // Store in both dictionary and ordered list
    FTensors.AddOrSetValue(LInfo.TensorName, LInfo);
    FTensorList.Add(LInfo);

    // Build dimensions string for status
    LDimsStr := '';
    for LJ := 0 to LInfo.NumDimensions - 1 do
    begin
      if LJ > 0 then
        LDimsStr := LDimsStr + ' x ';
      LDimsStr := LDimsStr + IntToStr(LInfo.Dimensions[LJ]);
    end;

    Status('  [%d] %s — %s [%s] offset=%d',
      [LI, LInfo.TensorName, VdxGGMLTypeName(LInfo.TensorType), LDimsStr, LInfo.DataOffset]);
  end;

  Status('Tensor info parsing complete (%d tensors)', [FTensors.Count]);
end;

procedure TVdxGGUFReader.ComputeTensorDataBase();
var
  LOffset: UInt64;
  LAligned: UInt64;
begin
  // Current cursor position is right after all tensor info entries.
  // Tensor data starts at the next ALIGNMENT boundary from here.
  LOffset := UInt64(FCursor) - UInt64(FVirtualFile.Memory);

  // Align: offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT
  LAligned := LOffset + (UInt64(FAlignment) - (LOffset mod UInt64(FAlignment))) mod UInt64(FAlignment);

  FTensorDataBase := PByte(FVirtualFile.Memory) + LAligned;

  Status('Tensor data starts at file offset %d ($%x), alignment=%d',
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
  Result := FVirtualFile.Size;
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

function TVdxGGUFReader.GetMetadataString(const AKey: string; const ADefault: string): string;
var
  LValue: TVdxGGUFMetaValue;
begin
  if FMetadata.TryGetValue(AKey, LValue) then
    Result := LValue.AsString
  else
    Result := ADefault;
end;

function TVdxGGUFReader.GetMetadataUInt64(const AKey: string; const ADefault: UInt64): UInt64;
var
  LValue: TVdxGGUFMetaValue;
begin
  if FMetadata.TryGetValue(AKey, LValue) then
    Result := LValue.AsUInt64
  else
    Result := ADefault;
end;

function TVdxGGUFReader.GetMetadataUInt32(const AKey: string; const ADefault: UInt32): UInt32;
var
  LValue: TVdxGGUFMetaValue;
begin
  if FMetadata.TryGetValue(AKey, LValue) then
    Result := UInt32(LValue.AsUInt64)
  else
    Result := ADefault;
end;

function TVdxGGUFReader.GetMetadataFloat32(const AKey: string; const ADefault: Single): Single;
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

function TVdxGGUFReader.GetTensorDataPtr(const ATensorName: string): Pointer;
var
  LInfo: TVdxGGUFTensorInfo;
begin
  Result := nil;

  if not GetTensorInfo(ATensorName, LInfo) then
  begin
    FErrors.Add(esError, VDX_ERROR_GG_TENSOR,
      'GGUF tensor not found: %s', [ATensorName]);
    Exit;
  end;

  if FTensorDataBase = nil then
  begin
    FErrors.Add(esError, VDX_ERROR_GG_NOT_OPEN,
      'GGUF file not open or tensor data base not computed');
    Exit;
  end;

  // DataOffset is relative to the start of tensor_data
  Result := FTensorDataBase + LInfo.DataOffset;
end;

function TVdxGGUFReader.GetTensorList(): TList<TVdxGGUFTensorInfo>;
begin
  Result := FTensorList;
end;

end.