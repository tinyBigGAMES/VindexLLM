{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.GGUFReader;

interface

uses
  VindexLLM.TestCase;

type

  { TGGUFReaderTest }
  // Section-based coverage of TVdxGGUFReader. The positive path
  // opens a real GGUF (Gemma 3 4B IT F16 at the hardcoded model
  // path) — if that file is absent on this machine the positive
  // sections will fail, which is the intended signal.
  TGGUFReaderTest = class(TVdxTestCase)
  private
    procedure WriteBadMagicFile(const APath: string);

    procedure SecInitialState();
    procedure SecOpenMissingFile();
    procedure SecOpenBadMagic();
    procedure SecOpenValidModel();
    procedure SecHeaderFields();
    procedure SecMetadataLookup();
    procedure SecTensorLookup();
    procedure SecMissingMetadata();
    procedure SecMissingTensor();
    procedure SecClose();
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
  VindexLLM.GGUFReader;

const
  CModelPath   = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf';
  CBadMagicBin = 'bad_magic.gguf';

{ TGGUFReaderTest }

constructor TGGUFReaderTest.Create();
begin
  inherited;
  Title := 'Test_GGUFReader';
end;

procedure TGGUFReaderTest.Run();
begin
  SecInitialState();
  SecOpenMissingFile();
  SecOpenBadMagic();
  SecOpenValidModel();
  SecHeaderFields();
  SecMetadataLookup();
  SecTensorLookup();
  SecMissingMetadata();
  SecMissingTensor();
  SecClose();
end;

// ---------------------------------------------------------------------------
// WriteBadMagicFile — writes a small file with a deliberately wrong
// magic number (4 bytes) followed by zeros. Used by the bad-magic
// section to drive the reader's magic-validation path without needing
// a real broken GGUF.
// ---------------------------------------------------------------------------
procedure TGGUFReaderTest.WriteBadMagicFile(const APath: string);
var
  LStream: TFileStream;
  LPayload: TBytes;
begin
  SetLength(LPayload, 32);
  // 'JUNK' — any 32-bit value that isn't $46554747 ('GGUF').
  LPayload[0] := Ord('J');
  LPayload[1] := Ord('U');
  LPayload[2] := Ord('N');
  LPayload[3] := Ord('K');

  LStream := TFileStream.Create(APath, fmCreate);
  try
    LStream.WriteBuffer(LPayload[0], Length(LPayload));
  finally
    LStream.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 1. Create() initial state — no errors, no backing file
// ---------------------------------------------------------------------------
procedure TGGUFReaderTest.SecInitialState();
var
  LReader: TVdxGGUFReader;
begin
  Section('Create() initial state');
  LReader := TVdxGGUFReader.Create();
  try
    Check(not LReader.GetErrors().HasFatal(),
      'No fatal errors after Create');
    Check(LReader.GetVersion() = 0,
      'Version is 0 before Open');
    Check(LReader.GetTensorCount() = 0,
      'TensorCount is 0 before Open');
    Check(LReader.GetMetadataCount() = 0,
      'MetadataCount is 0 before Open');
    Check(LReader.GetFileSize() = 0,
      'FileSize is 0 before Open');
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 2. Open missing file — error comes through from the contained
// TVdxVirtualFile<Byte> via the shared FErrors buffer, proving the
// composition-time SetErrors wiring works.
// ---------------------------------------------------------------------------
procedure TGGUFReaderTest.SecOpenMissingFile();
var
  LReader: TVdxGGUFReader;
  LItems: TList<TVdxError>;
begin
  Section('Open missing file - VF error propagates via shared buffer');
  LReader := TVdxGGUFReader.Create();
  try
    Check(not LReader.Open('does_not_exist.gguf'),
      'Open on missing path returns False');
    Check(LReader.GetErrors().HasFatal(),
      'HasFatal is True');

    LItems := LReader.GetErrors().GetItems();
    if LItems.Count >= 1 then
    begin
      Check(LItems[0].Severity = esFatal,
        'Entry severity is esFatal');
      // VF02 = TVdxVirtualFile's "open failed" — it bubbled up
      // through the shared error buffer.
      Check(LItems[0].Code = 'VF02',
        'Entry code is VF02 (from contained TVdxVirtualFile)');
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 3. Open bad-magic file — magic validation at GG layer (not VF).
// ---------------------------------------------------------------------------
procedure TGGUFReaderTest.SecOpenBadMagic();
var
  LReader: TVdxGGUFReader;
  LItems: TList<TVdxError>;
  LFoundBadMagic: Boolean;
  LI: Integer;
begin
  Section('Open bad-magic file - reports GG_BAD_MAGIC');
  WriteBadMagicFile(CBadMagicBin);
  try
    LReader := TVdxGGUFReader.Create();
    try
      Check(not LReader.Open(CBadMagicBin),
        'Open on bad-magic file returns False');
      Check(LReader.GetErrors().HasFatal(),
        'HasFatal is True');

      LItems := LReader.GetErrors().GetItems();
      LFoundBadMagic := False;
      for LI := 0 to LItems.Count - 1 do
        if LItems[LI].Code = VDX_ERROR_GG_BAD_MAGIC then
        begin
          LFoundBadMagic := True;
          Break;
        end;
      Check(LFoundBadMagic,
        'Error list contains GG_BAD_MAGIC (GG02)');
      FlushErrors(LReader.GetErrors());
    finally
      LReader.Free();
    end;
  finally
    if TFile.Exists(CBadMagicBin) then
      TFile.Delete(CBadMagicBin);
  end;
end;

// ---------------------------------------------------------------------------
// 4. Open the real Gemma 3 4B F16 GGUF. If this file is missing on
// disk the test fails here with VF02 — that's the intended signal.
// ---------------------------------------------------------------------------
procedure TGGUFReaderTest.SecOpenValidModel();
var
  LReader: TVdxGGUFReader;
begin
  Section('Open valid Gemma 3 4B F16 GGUF');
  LReader := TVdxGGUFReader.Create();
  try
    Check(LReader.Open(CModelPath),
      'Open returns True on real GGUF file');
    Check(not LReader.GetErrors().HasFatal(),
      'No fatal errors after successful Open');
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 5. Header fields populated after Open — version, counts, size
// ---------------------------------------------------------------------------
procedure TGGUFReaderTest.SecHeaderFields();
var
  LReader: TVdxGGUFReader;
begin
  Section('Header fields populated after Open');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      Check(LReader.GetVersion() >= 2,
        'Version is >= 2');
      Check(LReader.GetTensorCount() > 0,
        'TensorCount is > 0');
      Check(LReader.GetMetadataCount() > 0,
        'MetadataCount is > 0');
      Check(LReader.GetFileSize() > 0,
        'FileSize is > 0');
      Check(LReader.GetAlignment() > 0,
        'Alignment is > 0');
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 6. Metadata lookup — general.architecture exists and is a non-empty
// string. Reader stays architecture-agnostic, so no assertion on the
// specific string value — just that it's present and populated.
// ---------------------------------------------------------------------------
procedure TGGUFReaderTest.SecMetadataLookup();
var
  LReader: TVdxGGUFReader;
  LValue: TVdxGGUFMetaValue;
  LArch: string;
begin
  Section('Metadata lookup - general.architecture present');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      Check(LReader.HasMetadata('general.architecture'),
        'HasMetadata(general.architecture) is True');
      Check(LReader.GetMetadata('general.architecture', LValue),
        'GetMetadata(general.architecture) returns True');
      Check(LValue.ValueType = gmtString,
        'Metadata ValueType is gmtString');
      Check(Length(LValue.AsString) > 0,
        'Architecture string is non-empty');

      // Convenience accessor returns the same string
      LArch := LReader.GetMetadataString('general.architecture');
      Check(LArch = LValue.AsString,
        'GetMetadataString convenience matches direct value');
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 7. Tensor lookup — token_embd.weight is universal across GGUF LLMs.
// Check it's present, has dims, and yields a non-nil mmap pointer.
// ---------------------------------------------------------------------------
procedure TGGUFReaderTest.SecTensorLookup();
var
  LReader: TVdxGGUFReader;
  LInfo: TVdxGGUFTensorInfo;
  LPtr: PByte;
begin
  Section('Tensor lookup - token_embd.weight');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      Check(LReader.HasTensor('token_embd.weight'),
        'HasTensor(token_embd.weight) is True');
      Check(LReader.GetTensorInfo('token_embd.weight', LInfo),
        'GetTensorInfo(token_embd.weight) returns True');
      Check(LInfo.NumDimensions > 0,
        'Tensor has at least one dimension');
      Check(Length(LInfo.Dimensions) = Integer(LInfo.NumDimensions),
        'Dimensions array length matches NumDimensions');

      LPtr := LReader.GetTensorDataPtr('token_embd.weight');
      Check(LPtr <> nil,
        'GetTensorDataPtr returns non-nil mmap pointer');
      Check(LReader.GetTensorList().Count > 0,
        'GetTensorList is non-empty');
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 8. Missing metadata — not found returns False silently, no fatal
// logged (missing keys are the caller's concern, not the reader's).
// ---------------------------------------------------------------------------
procedure TGGUFReaderTest.SecMissingMetadata();
var
  LReader: TVdxGGUFReader;
  LValue: TVdxGGUFMetaValue;
  LDefault: string;
begin
  Section('Missing metadata - silent False, no fatal');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      Check(not LReader.HasMetadata('this.key.does.not.exist'),
        'HasMetadata returns False for missing key');
      Check(not LReader.GetMetadata('this.key.does.not.exist', LValue),
        'GetMetadata returns False for missing key');
      Check(not LReader.GetErrors().HasFatal(),
        'No fatal logged for missing metadata');

      // Convenience accessor returns the default silently
      LDefault := LReader.GetMetadataString(
        'this.key.does.not.exist', 'fallback-value');
      Check(LDefault = 'fallback-value',
        'GetMetadataString returns default on miss');
      Check(not LReader.GetErrors().HasFatal(),
        'Still no fatal after convenience-accessor miss');
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 9. Missing tensor — same policy as missing metadata: silent False,
// no fatal.
// ---------------------------------------------------------------------------
procedure TGGUFReaderTest.SecMissingTensor();
var
  LReader: TVdxGGUFReader;
  LInfo: TVdxGGUFTensorInfo;
begin
  Section('Missing tensor - silent False, no fatal');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      Check(not LReader.HasTensor('this.tensor.does.not.exist'),
        'HasTensor returns False for missing tensor');
      Check(not LReader.GetTensorInfo(
        'this.tensor.does.not.exist', LInfo),
        'GetTensorInfo returns False for missing tensor');
      Check(not LReader.GetErrors().HasFatal(),
        'No fatal logged for missing tensor');
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

// ---------------------------------------------------------------------------
// 10. Close — tear-down resets state, is idempotent, and the reader
// can be reopened cleanly.
// ---------------------------------------------------------------------------
procedure TGGUFReaderTest.SecClose();
var
  LReader: TVdxGGUFReader;
begin
  Section('Close tear-down and reopen');
  LReader := TVdxGGUFReader.Create();
  try
    if LReader.Open(CModelPath) then
    begin
      LReader.Close();
      Check(LReader.GetTensorCount() = 0,
        'TensorCount reset to 0 after Close');
      Check(LReader.GetMetadataCount() = 0,
        'MetadataCount reset to 0 after Close');
      Check(LReader.GetFileSize() = 0,
        'FileSize reset to 0 after Close');

      // Idempotent — second Close is a no-op.
      LReader.Close();
      Check(not LReader.GetErrors().HasFatal(),
        'Double-Close logs no fatal');

      // Reopen the same path works cleanly.
      Check(LReader.Open(CModelPath),
        'Reopen after Close returns True');
      Check(LReader.GetTensorCount() > 0,
        'TensorCount populated again after reopen');
    end;
    FlushErrors(LReader.GetErrors());
  finally
    LReader.Free();
  end;
end;

end.
