{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.VirtualFile;

{$I VindexLLM.Defines.inc}

interface

uses
  WinApi.Windows,
  System.SysUtils,
  System.Classes,
  System.SyncObjs,
  VindexLLM.Utils;

const
  // Error codes — user-facing messages live in VindexLLM.Resources.
  VDX_ERROR_VF_NOT_OPEN       = 'VF01';
  VDX_ERROR_VF_OPEN_FAILED    = 'VF02';
  VDX_ERROR_VF_MAPPING_FAILED = 'VF03';
  VDX_ERROR_VF_MAPVIEW_FAILED = 'VF04';
  VDX_ERROR_VF_OPEN_EXCEPTION = 'VF05';
  VDX_ERROR_VF_EMPTY          = 'VF06';

type

  { TVdxVirtualFile<T> }
  // Read-only memory-mapped view of an existing file, exposed as a
  // typed array. Companion to TVdxVirtualBuffer<T>, which maps an
  // anonymous in-memory region. Thread-safe via internal critical
  // section. The entire file is mapped at Open time; on Win64 with
  // UInt64 offsets this works for multi-GB files.
  TVdxVirtualFile<T> = class(TVdxBaseObject)
  private
    FFilename: string;
    FFileHandle: THandle;
    FMappingHandle: THandle;
    FMemory: Pointer;
    FSize: UInt64;
    FPosition: UInt64;
    FCriticalSection: TCriticalSection;
    function  GetItem(const AIndex: UInt64): T;
    function  GetCapacity(): UInt64;
    function  GetIsOpen(): Boolean;
    procedure SetPosition(const AValue: UInt64);
    procedure Lock();
    procedure Unlock();
  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Opens AFilename read-only with fmShareDenyWrite semantics and
    // memory-maps the entire file. Returns False and populates
    // FErrors on any failure (OS error, file not found, zero-byte
    // file). Safe to call multiple times — closes any prior mapping
    // first. The file remains open for the lifetime of the instance
    // or until Close is called.
    function  Open(const AFilename: string): Boolean;

    // Closes and unmaps. Idempotent — safe to call when not open.
    procedure Close();

    // Reads ACount bytes from the current cursor position into
    // ABuffer. Advances Position. Returns the number of bytes
    // actually read (may be less than ACount at end of file).
    // Returns 0 if not open.
    function  Read(var ABuffer; const ACount: UInt64): UInt64;

    // True when Position has reached Size.
    function  Eob(): Boolean;

    // Typed indexed access. Read-only. Bounds violations raise
    // EArgumentOutOfRangeException — that is a programmer error,
    // not a runtime failure mode.
    property  Item[const AIndex: UInt64]: T read GetItem; default;

    property  IsOpen:   Boolean read GetIsOpen;
    property  Filename: string  read FFilename;
    property  Memory:   Pointer read FMemory;
    property  Size:     UInt64  read FSize;
    property  Capacity: UInt64  read GetCapacity;
    property  Position: UInt64  read FPosition write SetPosition;
  end;

implementation

uses
  VindexLLM.Resources;

{ TVdxVirtualFile<T> }

constructor TVdxVirtualFile<T>.Create();
begin
  inherited Create();
  FCriticalSection := TCriticalSection.Create();
  FFilename      := '';
  FFileHandle    := INVALID_HANDLE_VALUE;
  FMappingHandle := 0;
  FMemory        := nil;
  FSize          := 0;
  FPosition      := 0;
end;

destructor TVdxVirtualFile<T>.Destroy();
begin
  Close();
  FCriticalSection.Free();
  inherited;
end;

procedure TVdxVirtualFile<T>.Lock();
begin
  FCriticalSection.Enter();
end;

procedure TVdxVirtualFile<T>.Unlock();
begin
  FCriticalSection.Leave();
end;

function TVdxVirtualFile<T>.GetIsOpen(): Boolean;
begin
  Result := FMemory <> nil;
end;

function TVdxVirtualFile<T>.GetCapacity(): UInt64;
begin
  Result := FSize div UInt64(SizeOf(T));
end;

procedure TVdxVirtualFile<T>.SetPosition(const AValue: UInt64);
begin
  Lock();
  try
    // Bounds violation = programmer error. Raise.
    if AValue > FSize then
      raise EArgumentOutOfRangeException.Create('Position out of bounds');
    FPosition := AValue;
  finally
    Unlock();
  end;
end;

function TVdxVirtualFile<T>.GetItem(const AIndex: UInt64): T;
begin
  Lock();
  try
    // Bounds violation = programmer error. Raise — no sensible
    // default to return, caller cannot continue meaningfully.
    if AIndex >= GetCapacity() then
      raise EArgumentOutOfRangeException.Create('Index out of bounds');
    CopyMemory(@Result,
      Pointer(UIntPtr(FMemory) + UIntPtr(AIndex * UInt64(SizeOf(T)))),
      SizeOf(T));
  finally
    Unlock();
  end;
end;

procedure TVdxVirtualFile<T>.Close();
begin
  Lock();
  try
    if FMemory <> nil then
    begin
      UnmapViewOfFile(FMemory);
      FMemory := nil;
    end;
    if FMappingHandle <> 0 then
    begin
      CloseHandle(FMappingHandle);
      FMappingHandle := 0;
    end;
    if FFileHandle <> INVALID_HANDLE_VALUE then
    begin
      CloseHandle(FFileHandle);
      FFileHandle := INVALID_HANDLE_VALUE;
    end;
    FSize     := 0;
    FPosition := 0;
    FFilename := '';
  finally
    Unlock();
  end;
end;

function TVdxVirtualFile<T>.Open(const AFilename: string): Boolean;
var
  LFileSizeHigh: DWORD;
  LFileSizeLow: DWORD;
  LTotalSize: UInt64;
begin
  Result := False;

  // Idempotent re-open: release any prior mapping first.
  Close();

  try
    // CreateFileW with GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING.
    // fmShareDenyWrite semantics — other readers OK, writers blocked.
    FFileHandle := CreateFile(PChar(AFilename),
      GENERIC_READ,
      FILE_SHARE_READ,
      nil,
      OPEN_EXISTING,
      FILE_ATTRIBUTE_NORMAL,
      0);
    if FFileHandle = INVALID_HANDLE_VALUE then
    begin
      FErrors.Add(esFatal, VDX_ERROR_VF_OPEN_FAILED,
        RSVFOpenFailed, [AFilename, GetLastError()]);
      Exit;
    end;

    LFileSizeLow := GetFileSize(FFileHandle, @LFileSizeHigh);
    LTotalSize   := (UInt64(LFileSizeHigh) shl 32) or UInt64(LFileSizeLow);

    if LTotalSize = 0 then
    begin
      // Zero-byte files can't be memory-mapped on Windows. Treat as
      // a fatal open failure with a distinct error code so callers
      // can tell it apart from a missing file.
      FErrors.Add(esFatal, VDX_ERROR_VF_EMPTY, RSVFEmpty, [AFilename]);
      CloseHandle(FFileHandle);
      FFileHandle := INVALID_HANDLE_VALUE;
      Exit;
    end;

    FMappingHandle := CreateFileMapping(FFileHandle, nil,
      PAGE_READONLY, 0, 0, nil);
    if FMappingHandle = 0 then
    begin
      FErrors.Add(esFatal, VDX_ERROR_VF_MAPPING_FAILED,
        RSVFMappingFailed, [AFilename, GetLastError()]);
      CloseHandle(FFileHandle);
      FFileHandle := INVALID_HANDLE_VALUE;
      Exit;
    end;

    FMemory := MapViewOfFile(FMappingHandle, FILE_MAP_READ, 0, 0, 0);
    if FMemory = nil then
    begin
      FErrors.Add(esFatal, VDX_ERROR_VF_MAPVIEW_FAILED,
        RSVFMapViewFailed, [AFilename, GetLastError()]);
      CloseHandle(FMappingHandle);
      FMappingHandle := 0;
      CloseHandle(FFileHandle);
      FFileHandle := INVALID_HANDLE_VALUE;
      Exit;
    end;
  except
    on E: Exception do
    begin
      FErrors.Add(esFatal, VDX_ERROR_VF_OPEN_EXCEPTION,
        RSVFOpenException, [AFilename, E.Message]);
      Close();
      Exit;
    end;
  end;

  FFilename := AFilename;
  FSize     := LTotalSize;
  FPosition := 0;
  Result    := True;
end;

function TVdxVirtualFile<T>.Read(var ABuffer; const ACount: UInt64): UInt64;
var
  LCount: UInt64;
begin
  Result := 0;
  Lock();
  try
    if FMemory = nil then
    begin
      FErrors.Add(esError, VDX_ERROR_VF_NOT_OPEN, RSVFNotOpen);
      Exit;
    end;

    LCount := ACount;
    if FPosition + LCount > FSize then
      LCount := FSize - FPosition;

    if LCount > 0 then
      CopyMemory(@ABuffer,
        Pointer(UIntPtr(FMemory) + UIntPtr(FPosition)), LCount);

    Inc(FPosition, LCount);
    Result := LCount;
  finally
    Unlock();
  end;
end;

function TVdxVirtualFile<T>.Eob(): Boolean;
begin
  Result := FPosition >= FSize;
end;

end.
