{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.VirtualBuffer;

interface

uses
  VindexLLM.TestCase;

type

  { TVirtualBufferTest }
  // 17-section coverage of TVdxVirtualBuffer<T>. Each Sec* method owns
  // its own fixtures so failures do not cascade between sections.
  TVirtualBufferTest = class(TVdxTestCase)
  private
    procedure SecInitialState();
    procedure SecAllocate();
    procedure SecStride();
    procedure SecRawRoundTrip();
    procedure SecTBytesRoundTrip();
    procedure SecIndexedAccess();
    procedure SecStringRoundTrip();
    procedure SecZeroMemory();
    procedure SecCopyFrom();
    procedure SecWriteOverflow();
    procedure SecReadClamp();
    procedure SecEob();
    procedure SecPositionRoundTrip();
    procedure SecReAllocateIdempotency();
    procedure SecSaveLoadRoundTrip();
    procedure SecAllocateZeroErrorDetails();
    procedure SecLoadFromFileMissingErrorDetails();
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
  VindexLLM.VirtualBuffer;

{ TVirtualBufferTest }

constructor TVirtualBufferTest.Create();
begin
  inherited;
  Title := 'Test_VirtualBuffer';
end;

procedure TVirtualBufferTest.Run();
begin
  SecInitialState();
  SecAllocate();
  SecStride();
  SecRawRoundTrip();
  SecTBytesRoundTrip();
  SecIndexedAccess();
  SecStringRoundTrip();
  SecZeroMemory();
  SecCopyFrom();
  SecWriteOverflow();
  SecReadClamp();
  SecEob();
  SecPositionRoundTrip();
  SecReAllocateIdempotency();
  SecSaveLoadRoundTrip();
  SecAllocateZeroErrorDetails();
  SecLoadFromFileMissingErrorDetails();
end;

// ---------------------------------------------------------------------------
// 1. Create() initial state — everything nil/zero, no errors logged
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecInitialState();
var
  LBuf: TVdxVirtualBuffer<Byte>;
begin
  Section('Create() initial state');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    Check(not LBuf.GetErrors().HasFatal() and
          not LBuf.GetErrors().HasErrors(),
      'FErrors empty after Create');
    Check(LBuf.Memory = nil, 'Memory is nil before Allocate');
    Check(LBuf.Size = 0, 'Size is 0 before Allocate');
    Check(LBuf.Capacity = 0, 'Capacity is 0 before Allocate');
    Check(LBuf.Position = 0, 'Position is 0 before Allocate');
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 2. Allocate(1024) — Byte element
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecAllocate();
var
  LBuf: TVdxVirtualBuffer<Byte>;
begin
  Section('Allocate(1024) — Byte element');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    Check(LBuf.Allocate(1024), 'Allocate(1024) returns True');
    Check(LBuf.Memory <> nil, 'Memory is non-nil after Allocate');
    Check(LBuf.Size = 1024, 'Size = 1024');
    Check(LBuf.Capacity = 1024, 'Capacity = 1024 for Byte element');
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 3. Stride — UInt32 element (256 elements = 1024 bytes)
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecStride();
var
  LBuf: TVdxVirtualBuffer<UInt32>;
begin
  Section('Stride — UInt32 element');
  LBuf := TVdxVirtualBuffer<UInt32>.Create();
  try
    Check(LBuf.Allocate(256),
      'Allocate(256) returns True for UInt32 buffer');
    Check(LBuf.Size = 256 * SizeOf(UInt32),
      'Size = 256 * SizeOf(UInt32)');
    Check(LBuf.Capacity = 256, 'Capacity = 256 for UInt32 element');
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 4. Write/Read — raw buffer overload
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecRawRoundTrip();
var
  LBuf: TVdxVirtualBuffer<Byte>;
  LIn: array[0..31] of Byte;
  LOut: array[0..31] of Byte;
  LI: Integer;
  LOK: Boolean;
begin
  Section('Write/Read — raw buffer overload');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    if LBuf.Allocate(256) then
    begin
      for LI := 0 to 31 do
        LIn[LI] := Byte(LI * 17);

      LBuf.Position := 0;
      Check(LBuf.Write(LIn[0], 32) = 32,
        'Write returns 32 for 32-byte payload');

      LBuf.Position := 0;
      Check(LBuf.Read(LOut[0], 32) = 32, 'Read returns 32 bytes');

      LOK := True;
      for LI := 0 to 31 do
        if LOut[LI] <> LIn[LI] then
        begin
          LOK := False;
          Break;
        end;
      Check(LOK, 'Raw-buffer round-trip is byte-identical');
    end;
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 5. Write/Read — TBytes overload (with offset)
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecTBytesRoundTrip();
var
  LBuf: TVdxVirtualBuffer<Byte>;
  LIn: TBytes;
  LOut: TBytes;
  LI: Integer;
  LOK: Boolean;
begin
  Section('Write/Read — TBytes overload');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    if LBuf.Allocate(256) then
    begin
      SetLength(LIn, 32);
      for LI := 0 to 31 do
        LIn[LI] := Byte((LI * 3 + 7) and $FF);

      LBuf.Position := 0;
      Check(LBuf.Write(LIn, 0, 32) = 32,
        'Write(TBytes, 0, 32) returns 32');

      SetLength(LOut, 32);
      LBuf.Position := 0;
      Check(LBuf.Read(LOut, 0, 32) = 32, 'Read(TBytes, 0, 32) returns 32');

      LOK := True;
      for LI := 0 to 31 do
        if LOut[LI] <> LIn[LI] then
        begin
          LOK := False;
          Break;
        end;
      Check(LOK, 'TBytes round-trip is byte-identical');
    end;
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 6. Indexed SetItem + GetItem via [] operator
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecIndexedAccess();
var
  LBuf: TVdxVirtualBuffer<Byte>;
  LI: Integer;
  LOK: Boolean;
begin
  Section('Indexed SetItem + GetItem');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    if LBuf.Allocate(16) then
    begin
      for LI := 0 to 15 do
        LBuf[UInt64(LI)] := Byte(LI * 11);

      LOK := True;
      for LI := 0 to 15 do
        if LBuf[UInt64(LI)] <> Byte(LI * 11) then
        begin
          LOK := False;
          Break;
        end;
      Check(LOK, 'SetItem/GetItem round-trip via [] operator');
    end;
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 7. WriteString / ReadString round-trip
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecStringRoundTrip();
var
  LBuf: TVdxVirtualBuffer<Byte>;
  LIn: string;
  LOut: string;
begin
  Section('WriteString / ReadString');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    if LBuf.Allocate(1024) then
    begin
      LIn := 'Hello, VindexLLM — testing string round-trip.';
      LBuf.Position := 0;
      LBuf.WriteString(LIn);

      LBuf.Position := 0;
      LOut := LBuf.ReadString();
      Check(LOut = LIn, 'String round-trip is identical');
    end;
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 8. ZeroMemory — wipes all bytes to zero
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecZeroMemory();
var
  LBuf: TVdxVirtualBuffer<Byte>;
  LI: Integer;
  LOK: Boolean;
begin
  Section('ZeroMemory');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    if LBuf.Allocate(64) then
    begin
      for LI := 0 to 63 do
        LBuf[UInt64(LI)] := $FF;

      LBuf.ZeroMemory();

      LOK := True;
      for LI := 0 to 63 do
        if LBuf[UInt64(LI)] <> 0 then
        begin
          LOK := False;
          Break;
        end;
      Check(LOK, 'ZeroMemory zeros all bytes');
    end;
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 9. CopyFrom — external source buffer → VirtualBuffer memory
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecCopyFrom();
var
  LBuf: TVdxVirtualBuffer<Byte>;
  LSource: array[0..63] of Byte;
  LI: Integer;
  LOK: Boolean;
begin
  Section('CopyFrom');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    if LBuf.Allocate(64) then
    begin
      for LI := 0 to 63 do
        LSource[LI] := Byte((LI xor $A5) and $FF);

      LBuf.CopyFrom(@LSource[0], 64);

      LOK := True;
      for LI := 0 to 63 do
        if LBuf[UInt64(LI)] <> LSource[LI] then
        begin
          LOK := False;
          Break;
        end;
      Check(LOK, 'CopyFrom reproduces source bytes');
    end;
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 10. Write overflow contract — returns 0, leaves Position unchanged
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecWriteOverflow();
var
  LBuf: TVdxVirtualBuffer<Byte>;
  LSrc: array[0..9] of Byte;
  LI: Integer;
begin
  Section('Write overflow contract');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    if LBuf.Allocate(16) then
    begin
      for LI := 0 to 9 do
        LSrc[LI] := Byte(LI);

      LBuf.Position := 10;
      Check(LBuf.Write(LSrc[0], 10) = 0, 'Write past end returns 0');
      Check(LBuf.Position = 10, 'Position unchanged on Write overflow');
    end;
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 11. Read clamp to FSize − FPosition
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecReadClamp();
var
  LBuf: TVdxVirtualBuffer<Byte>;
  LDest: TBytes;
begin
  Section('Read clamp behavior');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    if LBuf.Allocate(16) then
    begin
      LBuf.Position := 10;
      SetLength(LDest, 10);
      Check(LBuf.Read(LDest[0], 10) = 6,
        'Read requesting 10 from Position=10 returns 6');
      Check(LBuf.Position = 16, 'Position advanced to Size');
    end;
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 12. Eob semantics
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecEob();
var
  LBuf: TVdxVirtualBuffer<Byte>;
begin
  Section('Eob semantics');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    if LBuf.Allocate(32) then
    begin
      LBuf.Position := 0;
      Check(not LBuf.Eob(), 'Eob is False at Position=0');
      LBuf.Position := 16;
      Check(not LBuf.Eob(), 'Eob is False mid-buffer');
      LBuf.Position := 32;
      Check(LBuf.Eob(), 'Eob is True at Position=Size');
    end;
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 13. Position property round-trip
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecPositionRoundTrip();
var
  LBuf: TVdxVirtualBuffer<Byte>;
begin
  Section('Position property round-trip');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    if LBuf.Allocate(128) then
    begin
      LBuf.Position := 42;
      Check(LBuf.Position = 42, 'Position set to 42 reads back 42');
      LBuf.Position := 0;
      Check(LBuf.Position = 0, 'Position reset to 0');
      LBuf.Position := 128;
      Check(LBuf.Position = 128, 'Position set to Size is accepted');
    end;
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 14. Re-Allocate idempotency — new mapping wipes prior state
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecReAllocateIdempotency();
var
  LBuf: TVdxVirtualBuffer<Byte>;
begin
  Section('Re-Allocate idempotency');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    Check(LBuf.Allocate(1024), 'First Allocate(1024) succeeds');
    if LBuf.Memory <> nil then
      LBuf[0] := $AA;

    Check(LBuf.Allocate(2048), 'Second Allocate(2048) succeeds');
    Check(LBuf.Size = 2048, 'Size updated to 2048');
    Check(LBuf.Capacity = 2048, 'Capacity updated to 2048');
    if LBuf.Memory <> nil then
      Check(LBuf[0] = 0, 'Fresh mapping is zero-initialized');
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 15. SaveToFile + LoadFromFile round-trip
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecSaveLoadRoundTrip();
var
  LSaveBuf: TVdxVirtualBuffer<Byte>;
  LLoadBuf: TVdxVirtualBuffer<Byte>;
  LI: Integer;
  LOK: Boolean;
begin
  Section('SaveToFile + LoadFromFile round-trip');
  try
    LSaveBuf := TVdxVirtualBuffer<Byte>.Create();
    try
      if LSaveBuf.Allocate(64) then
      begin
        for LI := 0 to 63 do
          LSaveBuf[UInt64(LI)] := Byte((LI * 7) and $FF);
        LSaveBuf.SaveToFile('roundtrip.bin');
      end;
      FlushErrors(LSaveBuf.GetErrors());
    finally
      LSaveBuf.Free();
    end;

    LLoadBuf := TVdxVirtualBuffer<Byte>.LoadFromFile('roundtrip.bin');
    try
      Check(not LLoadBuf.GetErrors().HasFatal(),
        'LoadFromFile has no fatal errors');
      Check(LLoadBuf.Size = 64, 'Loaded Size matches saved size');
      if LLoadBuf.Memory <> nil then
      begin
        LOK := True;
        for LI := 0 to 63 do
          if LLoadBuf[UInt64(LI)] <> Byte((LI * 7) and $FF) then
          begin
            LOK := False;
            Break;
          end;
        Check(LOK, 'Loaded bytes match saved pattern');
      end;
      FlushErrors(LLoadBuf.GetErrors());
    finally
      LLoadBuf.Free();
    end;
  finally
    if TFile.Exists('roundtrip.bin') then
      TFile.Delete('roundtrip.bin');
  end;
end;

// ---------------------------------------------------------------------------
// 16. Allocate(0) — precise error details (severity + code)
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecAllocateZeroErrorDetails();
var
  LBuf: TVdxVirtualBuffer<Byte>;
  LItems: TList<TVdxError>;
begin
  Section('Allocate(0) error details');
  LBuf := TVdxVirtualBuffer<Byte>.Create();
  try
    Check(not LBuf.Allocate(0), 'Allocate(0) returns False');

    LItems := LBuf.GetErrors().GetItems();
    Check(LItems.Count = 1, 'Exactly one entry logged in FErrors');
    if LItems.Count >= 1 then
    begin
      Check(LItems[0].Severity = esError, 'Entry severity is esError');
      Check(LItems[0].Code = VDX_ERROR_VB_SIZE_ZERO,
        'Entry code is VDX_ERROR_VB_SIZE_ZERO (VB01)');
    end;
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 17. LoadFromFile missing path — precise error details
// ---------------------------------------------------------------------------
procedure TVirtualBufferTest.SecLoadFromFileMissingErrorDetails();
var
  LBuf: TVdxVirtualBuffer<UInt32>;
  LItems: TList<TVdxError>;
begin
  Section('LoadFromFile missing path error details');
  LBuf := TVdxVirtualBuffer<UInt32>.LoadFromFile('nonexistent.bin');
  try
    Check(LBuf <> nil, 'LoadFromFile returns non-nil instance on missing path');
    Check(LBuf.GetErrors().HasFatal(), 'HasFatal is True');

    LItems := LBuf.GetErrors().GetItems();
    if LItems.Count >= 1 then
    begin
      Check(LItems[0].Severity = esFatal, 'Entry severity is esFatal');
      Check(LItems[0].Code = VDX_ERROR_VB_LOADFILE_EXCEPTION,
        'Entry code is VDX_ERROR_VB_LOADFILE_EXCEPTION (VB06)');
    end;
    FlushErrors(LBuf.GetErrors());
  finally
    LBuf.Free();
  end;
end;

end.
