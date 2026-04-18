{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.VirtualFile;

interface

uses
  VindexLLM.TestCase;

type

  { TVirtualFileTest }
  // 11-section coverage of TVdxVirtualFile<T>. Each Sec* method owns
  // its own fixtures (temp files included). Shared temp-file setup
  // sits in CreateTempFile, invoked by each section that needs a
  // backing file.
  TVirtualFileTest = class(TVdxTestCase)
  private
    procedure CreateTempFile(const APath: string; const ASize: Integer);
    procedure CreateEmptyFile(const APath: string);

    procedure SecInitialState();
    procedure SecOpenValid();
    procedure SecStride();
    procedure SecIndexedAccess();
    procedure SecSequentialRead();
    procedure SecEob();
    procedure SecPositionProperty();
    procedure SecClose();
    procedure SecReopen();
    procedure SecOpenMissingErrorDetails();
    procedure SecOpenEmptyErrorDetails();
  protected
    procedure Run(); override;
  public
    constructor Create(); override;
  end;

implementation

uses
  System.SysUtils,
  System.IOUtils,
  System.Classes,
  System.Generics.Collections,
  VindexLLM.Utils,
  VindexLLM.VirtualFile;

const
  CTempPath       = 'temp.bin';
  CEmptyPath      = 'empty.bin';
  CPayloadSize    = 256;

{ TVirtualFileTest }

constructor TVirtualFileTest.Create();
begin
  inherited;
  Title := 'Test_VirtualFile';
end;

procedure TVirtualFileTest.Run();
begin
  SecInitialState();
  SecOpenValid();
  SecStride();
  SecIndexedAccess();
  SecSequentialRead();
  SecEob();
  SecPositionProperty();
  SecClose();
  SecReopen();
  SecOpenMissingErrorDetails();
  SecOpenEmptyErrorDetails();
end;

// ---------------------------------------------------------------------------
// CreateTempFile — writes ASize bytes of a known pattern (byte N = N mod 256)
// to APath. Called by sections that need a backing file to Open.
// ---------------------------------------------------------------------------
procedure TVirtualFileTest.CreateTempFile(const APath: string;
  const ASize: Integer);
var
  LStream: TFileStream;
  LPayload: TBytes;
  LI: Integer;
begin
  SetLength(LPayload, ASize);
  for LI := 0 to ASize - 1 do
    LPayload[LI] := Byte(LI and $FF);

  LStream := TFileStream.Create(APath, fmCreate);
  try
    LStream.WriteBuffer(LPayload[0], ASize);
  finally
    LStream.Free();
  end;
end;

// ---------------------------------------------------------------------------
// CreateEmptyFile — creates a zero-byte file at APath. Used by the
// empty-file negative-path section.
// ---------------------------------------------------------------------------
procedure TVirtualFileTest.CreateEmptyFile(const APath: string);
var
  LStream: TFileStream;
begin
  LStream := TFileStream.Create(APath, fmCreate);
  try
  finally
    LStream.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 1. Create() initial state — IsOpen false, all state at defaults
// ---------------------------------------------------------------------------
procedure TVirtualFileTest.SecInitialState();
var
  LVF: TVdxVirtualFile<Byte>;
begin
  Section('Create() initial state');
  LVF := TVdxVirtualFile<Byte>.Create();
  try
    Check(not LVF.IsOpen, 'IsOpen is False before Open');
    Check(LVF.Memory = nil, 'Memory is nil before Open');
    Check(LVF.Size = 0, 'Size is 0 before Open');
    Check(LVF.Capacity = 0, 'Capacity is 0 before Open');
    Check(LVF.Position = 0, 'Position is 0 before Open');
    Check(LVF.Filename = '', 'Filename is empty before Open');
    Check(not LVF.GetErrors().HasFatal(), 'No fatal errors after Create');
    FlushErrors(LVF.GetErrors());
  finally
    LVF.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 2. Open a valid file as Byte — all properties populated
// ---------------------------------------------------------------------------
procedure TVirtualFileTest.SecOpenValid();
var
  LVF: TVdxVirtualFile<Byte>;
begin
  Section('Open valid file — Byte element');
  CreateTempFile(CTempPath, CPayloadSize);
  try
    LVF := TVdxVirtualFile<Byte>.Create();
    try
      Check(LVF.Open(CTempPath), 'Open returns True for valid file');
      Check(LVF.IsOpen, 'IsOpen is True after Open');
      Check(LVF.Memory <> nil, 'Memory is non-nil after Open');
      Check(LVF.Size = CPayloadSize, 'Size matches payload length');
      Check(LVF.Capacity = CPayloadSize,
        'Capacity = Size for Byte element');
      Check(LVF.Filename = CTempPath,
        'Filename property matches opened path');
      Check(LVF.Position = 0, 'Position is 0 immediately after Open');
      FlushErrors(LVF.GetErrors());
    finally
      LVF.Free();
    end;
  finally
    if TFile.Exists(CTempPath) then
      TFile.Delete(CTempPath);
  end;
end;

// ---------------------------------------------------------------------------
// 3. Stride — open same-size file as UInt32
// ---------------------------------------------------------------------------
procedure TVirtualFileTest.SecStride();
var
  LVF: TVdxVirtualFile<UInt32>;
begin
  Section('Stride — UInt32 element');
  CreateTempFile(CTempPath, CPayloadSize);
  try
    LVF := TVdxVirtualFile<UInt32>.Create();
    try
      Check(LVF.Open(CTempPath), 'Open as UInt32 returns True');
      Check(LVF.Size = CPayloadSize,
        'Size still in bytes (UInt32 view)');
      Check(LVF.Capacity = CPayloadSize div SizeOf(UInt32),
        'Capacity divides by SizeOf(UInt32)');
      FlushErrors(LVF.GetErrors());
    finally
      LVF.Free();
    end;
  finally
    if TFile.Exists(CTempPath) then
      TFile.Delete(CTempPath);
  end;
end;

// ---------------------------------------------------------------------------
// 4. Indexed access via Item[]
// ---------------------------------------------------------------------------
procedure TVirtualFileTest.SecIndexedAccess();
var
  LVF: TVdxVirtualFile<Byte>;
begin
  Section('Indexed access via Item[]');
  CreateTempFile(CTempPath, CPayloadSize);
  try
    LVF := TVdxVirtualFile<Byte>.Create();
    try
      if LVF.Open(CTempPath) then
      begin
        Check(LVF[0] = Byte(0 and $FF), 'Item[0] matches payload[0]');
        Check(LVF[100] = Byte(100 and $FF),
          'Item[100] matches payload[100]');
        Check(LVF[CPayloadSize - 1] = Byte((CPayloadSize - 1) and $FF),
          'Item[last] matches payload[last]');
      end;
      FlushErrors(LVF.GetErrors());
    finally
      LVF.Free();
    end;
  finally
    if TFile.Exists(CTempPath) then
      TFile.Delete(CTempPath);
  end;
end;

// ---------------------------------------------------------------------------
// 5. Sequential Read — bytes + Position advance + payload match
// ---------------------------------------------------------------------------
procedure TVirtualFileTest.SecSequentialRead();
var
  LVF: TVdxVirtualFile<Byte>;
  LReadBack: array[0..7] of Byte;
  LBytesRead: UInt64;
  LOK: Boolean;
  LI: Integer;
begin
  Section('Sequential Read + payload match');
  CreateTempFile(CTempPath, CPayloadSize);
  try
    LVF := TVdxVirtualFile<Byte>.Create();
    try
      if LVF.Open(CTempPath) then
      begin
        LVF.Position := 0;
        LBytesRead := LVF.Read(LReadBack[0], 8);
        Check(LBytesRead = 8, 'Read returns 8 bytes from start');

        LOK := True;
        for LI := 0 to 7 do
          if LReadBack[LI] <> Byte(LI and $FF) then
          begin
            LOK := False;
            Break;
          end;
        Check(LOK, 'First 8 bytes match payload');
        Check(LVF.Position = 8, 'Position advances to 8 after Read');
      end;
      FlushErrors(LVF.GetErrors());
    finally
      LVF.Free();
    end;
  finally
    if TFile.Exists(CTempPath) then
      TFile.Delete(CTempPath);
  end;
end;

// ---------------------------------------------------------------------------
// 6. Eob semantics
// ---------------------------------------------------------------------------
procedure TVirtualFileTest.SecEob();
var
  LVF: TVdxVirtualFile<Byte>;
begin
  Section('Eob semantics');
  CreateTempFile(CTempPath, CPayloadSize);
  try
    LVF := TVdxVirtualFile<Byte>.Create();
    try
      if LVF.Open(CTempPath) then
      begin
        LVF.Position := 0;
        Check(not LVF.Eob(), 'Eob is False at Position=0');
        LVF.Position := CPayloadSize div 2;
        Check(not LVF.Eob(), 'Eob is False mid-file');
        LVF.Position := CPayloadSize;
        Check(LVF.Eob(), 'Eob is True at Position=Size');
      end;
      FlushErrors(LVF.GetErrors());
    finally
      LVF.Free();
    end;
  finally
    if TFile.Exists(CTempPath) then
      TFile.Delete(CTempPath);
  end;
end;

// ---------------------------------------------------------------------------
// 7. Position property round-trip
// ---------------------------------------------------------------------------
procedure TVirtualFileTest.SecPositionProperty();
var
  LVF: TVdxVirtualFile<Byte>;
begin
  Section('Position property round-trip');
  CreateTempFile(CTempPath, CPayloadSize);
  try
    LVF := TVdxVirtualFile<Byte>.Create();
    try
      if LVF.Open(CTempPath) then
      begin
        LVF.Position := 42;
        Check(LVF.Position = 42, 'Position set to 42 reads back 42');
        LVF.Position := 0;
        Check(LVF.Position = 0, 'Position reset to 0');
        LVF.Position := CPayloadSize;
        Check(LVF.Position = CPayloadSize,
          'Position set to Size is accepted');
      end;
      FlushErrors(LVF.GetErrors());
    finally
      LVF.Free();
    end;
  finally
    if TFile.Exists(CTempPath) then
      TFile.Delete(CTempPath);
  end;
end;

// ---------------------------------------------------------------------------
// 8. Close — state clears back to default
// ---------------------------------------------------------------------------
procedure TVirtualFileTest.SecClose();
var
  LVF: TVdxVirtualFile<Byte>;
begin
  Section('Close semantics');
  CreateTempFile(CTempPath, CPayloadSize);
  try
    LVF := TVdxVirtualFile<Byte>.Create();
    try
      if LVF.Open(CTempPath) then
      begin
        LVF.Close();
        Check(not LVF.IsOpen, 'IsOpen is False after Close');
        Check(LVF.Memory = nil, 'Memory is nil after Close');
        Check(LVF.Size = 0, 'Size zeroed after Close');
        Check(LVF.Position = 0, 'Position zeroed after Close');
        Check(LVF.Filename = '', 'Filename cleared after Close');
      end;
      FlushErrors(LVF.GetErrors());
    finally
      LVF.Free();
    end;
  finally
    if TFile.Exists(CTempPath) then
      TFile.Delete(CTempPath);
  end;
end;

// ---------------------------------------------------------------------------
// 9. Reopen after close — state re-initialized cleanly
// ---------------------------------------------------------------------------
procedure TVirtualFileTest.SecReopen();
var
  LVF: TVdxVirtualFile<Byte>;
begin
  Section('Reopen after Close');
  CreateTempFile(CTempPath, CPayloadSize);
  try
    LVF := TVdxVirtualFile<Byte>.Create();
    try
      if LVF.Open(CTempPath) then
      begin
        LVF.Position := 42;
        LVF.Close();
        Check(LVF.Open(CTempPath), 'Reopen returns True');
        Check(LVF.IsOpen, 'IsOpen True after reopen');
        Check(LVF.Position = 0, 'Position reset to 0 on reopen');
        Check(LVF.Size = CPayloadSize, 'Size populated on reopen');
      end;
      FlushErrors(LVF.GetErrors());
    finally
      LVF.Free();
    end;
  finally
    if TFile.Exists(CTempPath) then
      TFile.Delete(CTempPath);
  end;
end;

// ---------------------------------------------------------------------------
// 10. Open missing path — precise error details (severity + code)
// ---------------------------------------------------------------------------
procedure TVirtualFileTest.SecOpenMissingErrorDetails();
var
  LVF: TVdxVirtualFile<Byte>;
  LItems: TList<TVdxError>;
begin
  Section('Open missing path error details');
  LVF := TVdxVirtualFile<Byte>.Create();
  try
    Check(not LVF.Open('missing.bin'),
      'Open on missing path returns False');
    Check(LVF.GetErrors().HasFatal(), 'HasFatal is True');
    Check(not LVF.IsOpen, 'IsOpen is False after failed Open');

    LItems := LVF.GetErrors().GetItems();
    if LItems.Count >= 1 then
    begin
      Check(LItems[0].Severity = esFatal, 'Entry severity is esFatal');
      Check(LItems[0].Code = VDX_ERROR_VF_OPEN_FAILED,
        'Entry code is VDX_ERROR_VF_OPEN_FAILED (VF02)');
    end;
    FlushErrors(LVF.GetErrors());
  finally
    LVF.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 11. Open empty file — precise error details (VF_EMPTY / VF06)
// ---------------------------------------------------------------------------
procedure TVirtualFileTest.SecOpenEmptyErrorDetails();
var
  LVF: TVdxVirtualFile<Byte>;
  LItems: TList<TVdxError>;
begin
  Section('Open empty file error details');
  CreateEmptyFile(CEmptyPath);
  try
    LVF := TVdxVirtualFile<Byte>.Create();
    try
      Check(not LVF.Open(CEmptyPath),
        'Open on zero-byte file returns False');
      Check(LVF.GetErrors().HasFatal(),
        'HasFatal is True for empty file');
      Check(not LVF.IsOpen, 'IsOpen is False after failed Open');

      LItems := LVF.GetErrors().GetItems();
      if LItems.Count >= 1 then
      begin
        Check(LItems[0].Severity = esFatal, 'Entry severity is esFatal');
        Check(LItems[0].Code = VDX_ERROR_VF_EMPTY,
          'Entry code is VDX_ERROR_VF_EMPTY (VF06)');
      end;
      FlushErrors(LVF.GetErrors());
    finally
      LVF.Free();
    end;
  finally
    if TFile.Exists(CEmptyPath) then
      TFile.Delete(CEmptyPath);
  end;
end;

end.
