{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.VirtualBuffer;

{$I VindexLLM.Defines.inc}

interface

uses
  WinApi.Windows,
  System.SysUtils,
  System.IOUtils,
  System.Classes,
  System.SyncObjs,
  VindexLLM.Utils;

const
  // Error codes — user-facing messages live in VindexLLM.Resources.
  VDX_ERROR_VB_SIZE_ZERO          = 'VB01';
  VDX_ERROR_VB_MAPPING_FAILED     = 'VB02';
  VDX_ERROR_VB_MAPVIEW_FAILED     = 'VB03';
  VDX_ERROR_VB_ALLOCATE_EXCEPTION = 'VB04';
  VDX_ERROR_VB_ALIGNMENT          = 'VB05';
  VDX_ERROR_VB_LOADFILE_EXCEPTION = 'VB06';

type

  { TVdxVirtualBuffer<T> }
  TVdxVirtualBuffer<T> = class(TVdxBaseObject)
  private
    FHandle: THandle;
    FName: string;
    FCriticalSection: TCriticalSection;
    FMemory: Pointer;
    FSize: UInt64;
    FPosition: UInt64;
    procedure Clear();
    function GetItem(AIndex: UInt64): T;
    procedure SetItem(AIndex: UInt64; AValue: T);
    function GetCapacity(): UInt64;
    procedure SetPosition(const AValue: UInt64);
    procedure Lock();
    procedure Unlock();
  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Allocate or re-allocate the backing memory-mapped region.
    // Returns False and populates FErrors on OS failure. Safe to
    // call multiple times — releases prior mapping first.
    function Allocate(const ASize: UInt64): Boolean;

    // Stream read/write
    function Write(const ABuffer; const ACount: UInt64): UInt64; overload;
    function Write(const ABuffer: TBytes; const AOffset, ACount: UInt64): UInt64; overload;
    function Read(var ABuffer; const ACount: UInt64): UInt64; overload;
    function Read(var ABuffer: TBytes; const AOffset, ACount: UInt64): UInt64; overload;

    // String serialization
    function ReadString(): string;
    procedure WriteString(const AValue: string);

    // File I/O
    procedure SaveToFile(const AFilename: string);
    class function LoadFromFile(const AFilename: string): TVdxVirtualBuffer<T>;

    // Buffer operations
    procedure ZeroMemory();
    procedure CopyFrom(const ASource: Pointer; const ASizeBytes: UInt64);

    // End of buffer check
    function Eob(): Boolean;

    // Typed indexed access
    property Item[AIndex: UInt64]: T read GetItem write SetItem; default;

    // Properties
    property Capacity: UInt64 read GetCapacity;
    property Memory: Pointer read FMemory;
    property Size: UInt64 read FSize;
    property Position: UInt64 read FPosition write SetPosition;
    property Name: string read FName;
  end;

implementation

uses
  VindexLLM.Resources;

{ TVdxVirtualBuffer }
procedure TVdxVirtualBuffer<T>.Lock();
begin
  FCriticalSection.Enter();
end;

procedure TVdxVirtualBuffer<T>.Unlock();
begin
  FCriticalSection.Leave();
end;

procedure TVdxVirtualBuffer<T>.Clear();
begin
  if FMemory <> nil then
    UnmapViewOfFile(FMemory);

  if FHandle <> 0 then
    CloseHandle(FHandle);

  FMemory := nil;
  FHandle := 0;
  FSize := 0;
  FPosition := 0;
end;

function TVdxVirtualBuffer<T>.GetItem(AIndex: UInt64): T;
begin
  Lock();
  try
    if AIndex >= Capacity then
      raise EArgumentOutOfRangeException.Create('Index out of bounds');
    CopyMemory(@Result, Pointer(UIntPtr(FMemory) + UIntPtr(AIndex * UInt64(SizeOf(T)))), SizeOf(T));
  finally
    Unlock();
  end;
end;

procedure TVdxVirtualBuffer<T>.SetItem(AIndex: UInt64; AValue: T);
begin
  Lock();
  try
    if AIndex >= Capacity then
      raise EArgumentOutOfRangeException.Create('Index out of bounds');
    CopyMemory(Pointer(UIntPtr(FMemory) + UIntPtr(AIndex * UInt64(SizeOf(T)))), @AValue, SizeOf(T));
  finally
    Unlock();
  end;
end;

function TVdxVirtualBuffer<T>.GetCapacity(): UInt64;
begin
  Result := FSize div UInt64(SizeOf(T));
end;

procedure TVdxVirtualBuffer<T>.SetPosition(const AValue: UInt64);
begin
  Lock();
  try
    if AValue > FSize then
      raise EArgumentOutOfRangeException.Create('Position out of bounds');
    FPosition := AValue;
  finally
    Unlock();
  end;
end;

constructor TVdxVirtualBuffer<T>.Create();
begin
  inherited Create();
  FCriticalSection := TCriticalSection.Create();
  FHandle := 0;
  FMemory := nil;
  FSize := 0;
  FPosition := 0;
  FName := '';
end;

function TVdxVirtualBuffer<T>.Allocate(const ASize: UInt64): Boolean;
var
  LSizeHigh: DWORD;
  LSizeLow: DWORD;
  LTotalBytes: UInt64;
begin
  Result := False;

  if ASize = 0 then
  begin
    FErrors.Add(esError, VDX_ERROR_VB_SIZE_ZERO, RSVBSizeZero);
    Exit;
  end;

  // Idempotent re-allocate: release any prior mapping first.
  Clear();

  LTotalBytes := UInt64(SizeOf(T)) * ASize;
  LSizeLow  := DWORD(LTotalBytes and $FFFFFFFF);
  LSizeHigh := DWORD(LTotalBytes shr 32);

  FName := TPath.GetGUIDFileName();

  try
    FHandle := CreateFileMapping(INVALID_HANDLE_VALUE, nil,
      PAGE_READWRITE, LSizeHigh, LSizeLow, PChar(FName));
    if FHandle = 0 then
    begin
      FErrors.Add(esFatal, VDX_ERROR_VB_MAPPING_FAILED,
        RSVBMappingFailed, [GetLastError()]);
      Exit;
    end;

    FMemory := MapViewOfFile(FHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if FMemory = nil then
    begin
      FErrors.Add(esFatal, VDX_ERROR_VB_MAPVIEW_FAILED,
        RSVBMapViewFailed, [GetLastError()]);
      CloseHandle(FHandle);
      FHandle := 0;
      Exit;
    end;
  except
    on E: Exception do
    begin
      FErrors.Add(esFatal, VDX_ERROR_VB_ALLOCATE_EXCEPTION,
        RSVBAllocateException, [E.Message]);
      Clear();
      Exit;
    end;
  end;

  FSize := LTotalBytes;
  FPosition := 0;
  Result := True;
end;

destructor TVdxVirtualBuffer<T>.Destroy();
begin
  Clear();
  FCriticalSection.Free();
  inherited;
end;

function TVdxVirtualBuffer<T>.Write(const ABuffer; const ACount: UInt64): UInt64;
begin
  Lock();
  try
    if FPosition + ACount > FSize then
      Exit(0);
    CopyMemory(Pointer(UIntPtr(FMemory) + UIntPtr(FPosition)), @ABuffer, ACount);
    Inc(FPosition, ACount);
    Result := ACount;
  finally
    Unlock();
  end;
end;

function TVdxVirtualBuffer<T>.Write(const ABuffer: TBytes; const AOffset, ACount: UInt64): UInt64;
begin
  Lock();
  try
    if FPosition + ACount > FSize then
      Exit(0);
    CopyMemory(Pointer(UIntPtr(FMemory) + UIntPtr(FPosition)), @ABuffer[AOffset], ACount);
    Inc(FPosition, ACount);
    Result := ACount;
  finally
    Unlock();
  end;
end;

function TVdxVirtualBuffer<T>.Read(var ABuffer; const ACount: UInt64): UInt64;
var
  LCount: UInt64;
begin
  Lock();
  try
    LCount := ACount;
    if FPosition + LCount > FSize then
      LCount := FSize - FPosition;
    CopyMemory(@ABuffer, Pointer(UIntPtr(FMemory) + UIntPtr(FPosition)), LCount);
    Inc(FPosition, LCount);
    Result := LCount;
  finally
    Unlock();
  end;
end;

function TVdxVirtualBuffer<T>.Read(var ABuffer: TBytes; const AOffset, ACount: UInt64): UInt64;
var
  LCount: UInt64;
begin
  Lock();
  try
    if (AOffset + ACount > UInt64(Length(ABuffer))) then
      raise EArgumentOutOfRangeException.Create('Buffer overflow in Read');

    LCount := ACount;
    if FPosition + LCount > FSize then
      LCount := FSize - FPosition;

    CopyMemory(@ABuffer[AOffset], Pointer(UIntPtr(FMemory) + UIntPtr(FPosition)), LCount);
    Inc(FPosition, LCount);
    Result := LCount;
  finally
    Unlock();
  end;
end;

function TVdxVirtualBuffer<T>.ReadString(): string;
var
  LLen: UInt64;
begin
  Read(LLen, SizeOf(LLen));
  SetLength(Result, LLen);
  if LLen > 0 then
    Read(Result[1], LLen * SizeOf(Char));
end;

procedure TVdxVirtualBuffer<T>.WriteString(const AValue: string);
var
  LLength: UInt64;
begin
  Lock();
  try
    LLength := Length(AValue);
    Write(LLength, SizeOf(LLength));
    if LLength > 0 then
      Write(PChar(AValue)^, LLength * SizeOf(Char));
  finally
    Unlock();
  end;
end;

procedure TVdxVirtualBuffer<T>.SaveToFile(const AFilename: string);
var
  LFileStream: TFileStream;
begin
  LFileStream := TFileStream.Create(AFilename, fmCreate);
  try
    LFileStream.WriteBuffer(FMemory^, FSize);
  finally
    LFileStream.Free();
  end;
end;

class function TVdxVirtualBuffer<T>.LoadFromFile(const AFilename: string): TVdxVirtualBuffer<T>;
var
  LFileStream: TFileStream;
  LFileSize: Int64;
  LElements: UInt64;
begin
  // Always returns a non-nil instance. Caller MUST check
  // Result.HasFatal() before using it — on failure, FErrors is
  // populated with the reason and the instance is empty.
  Result := TVdxVirtualBuffer<T>.Create();
  try
    LFileStream := TFileStream.Create(AFilename, fmOpenRead or fmShareDenyWrite);
    try
      LFileSize := LFileStream.Size;
      if LFileSize mod SizeOf(T) <> 0 then
      begin
        Result.FErrors.Add(esFatal, VDX_ERROR_VB_ALIGNMENT,
          RSVBAlignment, [LFileSize, SizeOf(T)]);
        Exit;
      end;

      LElements := LFileSize div SizeOf(T);
      if not Result.Allocate(LElements) then
        Exit;  // FErrors already populated by Allocate

      LFileStream.ReadBuffer(Result.FMemory^, LFileSize);
      Result.FPosition := 0;
    finally
      LFileStream.Free();
    end;
  except
    on E: Exception do
      Result.FErrors.Add(esFatal, VDX_ERROR_VB_LOADFILE_EXCEPTION,
        RSVBLoadFileException, [AFilename, E.Message]);
  end;
end;

procedure TVdxVirtualBuffer<T>.ZeroMemory();
begin
  Lock();
  try
    FillChar(FMemory^, FSize, 0);
  finally
    Unlock();
  end;
end;

procedure TVdxVirtualBuffer<T>.CopyFrom(const ASource: Pointer; const ASizeBytes: UInt64);
begin
  Lock();
  try
    if ASizeBytes > FSize then
      raise EArgumentOutOfRangeException.Create('Source size exceeds buffer capacity');
    CopyMemory(FMemory, ASource, ASizeBytes);
  finally
    Unlock();
  end;
end;

function TVdxVirtualBuffer<T>.Eob(): Boolean;
begin
  Result := FPosition >= FSize;
end;

end.
