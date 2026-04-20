{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Embeddings;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Math,
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.Compute,
  VindexLLM.Model;

type

  { TVdxEmbeddingsEvent }
  TVdxEmbeddingsEvent = (
    eeLoadStart,
    eeLoadEnd,
    eeUnloadStart,
    eeUnloadEnd,
    eeEmbedStart,
    eeEmbedEnd
  );

  { TVdxEmbeddingsEventCallback }
  TVdxEmbeddingsEventCallback = reference to procedure(
    const AEvent: TVdxEmbeddingsEvent;
    const AUserData: Pointer);

  { TVdxEmbeddings }
  TVdxEmbeddings = class(TVdxBaseObject)
  private
    // Model — owns all subsystems, GPU plumbing, forward pass
    FModel: TVdxModel;

    // Staging buffer for downloading ResidualMat (device-local) to CPU
    FStagingBuf: TVdxGpuBuffer;

    FEventCallback: TVdxCallback<TVdxEmbeddingsEventCallback>;

    FModelLoaded: Boolean;
    FEmbeddingDim: Integer;

    // Post-pooling dense projections (CPU-side)
    FDense1Weights: array of Single;
    FDense2Weights: array of Single;
    FDense1In: Integer;
    FDense1Out: Integer;
    FDense2In: Integer;
    FDense2Out: Integer;
    FProjectionsLoaded: Boolean;

    // Private helpers
    procedure FireEvent(const AEvent: TVdxEmbeddingsEvent);

    function MeanPool(const AHiddenMat: array of Single;
      const ANumTokens: Integer): TArray<Single>;
    procedure L2Normalize(var AVec: TArray<Single>);
    function TryLoadDenseProjections(): Boolean;
    procedure ApplyLinear(const AIn: array of Single;
      const AWeights: array of Single; const AInDim, AOutDim: Integer;
      out AOut: TArray<Single>);

  public
    constructor Create(); override;
    destructor Destroy(); override;

    function LoadModel(const AGGUFPath: string): Boolean;
    procedure UnloadModel();
    function IsLoaded(): Boolean;

    function Embed(const AText: string; const AIsQuery: Boolean):
      TArray<Single>;

    procedure SetEmbeddingsEventCallback(
      const ACallback: TVdxEmbeddingsEventCallback;
      const AUserData: Pointer);

    function GetEmbeddingDim(): Integer;
    function GetMaxSeqLen(): Integer;
    function GetArchitecture(): string;

    class function CosineSimilarity(const AVecA, AVecB: TArray<Single>):
      Single; static;
  end;

implementation

uses
  VindexLLM.GGUFReader,
  VindexLLM.Tokenizer;

{ TVdxEmbeddings }

constructor TVdxEmbeddings.Create();
begin
  inherited;
  FModel := nil;
  FModelLoaded := False;
  FEmbeddingDim := 0;
  FProjectionsLoaded := False;
  FDense1In := 0;
  FDense1Out := 0;
  FDense2In := 0;
  FDense2Out := 0;
  FillChar(FStagingBuf, SizeOf(FStagingBuf), 0);
end;

destructor TVdxEmbeddings.Destroy();
begin
  if FModelLoaded then
    UnloadModel();
  inherited;
end;

function TVdxEmbeddings.IsLoaded(): Boolean;
begin
  Result := FModelLoaded;
end;

function TVdxEmbeddings.GetEmbeddingDim(): Integer;
begin
  Result := FEmbeddingDim;
end;

function TVdxEmbeddings.GetMaxSeqLen(): Integer;
begin
  if FModelLoaded then
    Result := Integer(FModel.MaxSeqLen)
  else
    Result := 0;
end;

function TVdxEmbeddings.GetArchitecture(): string;
begin
  if FModelLoaded then
    Result := FModel.Architecture
  else
    Result := '';
end;

procedure TVdxEmbeddings.SetEmbeddingsEventCallback(
  const ACallback: TVdxEmbeddingsEventCallback;
  const AUserData: Pointer);
begin
  FEventCallback.Callback := ACallback;
  FEventCallback.UserData := AUserData;
end;

procedure TVdxEmbeddings.FireEvent(const AEvent: TVdxEmbeddingsEvent);
begin
  if FEventCallback.IsAssigned() then
    FEventCallback.Callback(AEvent, FEventCallback.UserData);
end;

function TVdxEmbeddings.MeanPool(const AHiddenMat: array of Single;
  const ANumTokens: Integer): TArray<Single>;
var
  LI: Integer;
  LT: Integer;
  LSum: Single;
  LInv: Single;
  LHiddenDim: Integer;
begin
  LHiddenDim := Integer(FModel.HiddenDim);
  SetLength(Result, LHiddenDim);
  if ANumTokens <= 0 then
    Exit;

  LInv := 1.0 / Single(ANumTokens);
  for LI := 0 to LHiddenDim - 1 do
  begin
    LSum := 0.0;
    for LT := 0 to ANumTokens - 1 do
      LSum := LSum + AHiddenMat[LT * LHiddenDim + LI];
    Result[LI] := LSum * LInv;
  end;
end;

procedure TVdxEmbeddings.L2Normalize(var AVec: TArray<Single>);
var
  LI: Integer;
  LSumSq: Single;
  LInvNorm: Single;
begin
  LSumSq := 0.0;
  for LI := 0 to High(AVec) do
    LSumSq := LSumSq + AVec[LI] * AVec[LI];

  if LSumSq <= 1E-20 then
    Exit;

  LInvNorm := 1.0 / Sqrt(LSumSq);
  for LI := 0 to High(AVec) do
    AVec[LI] := AVec[LI] * LInvNorm;
end;

class function TVdxEmbeddings.CosineSimilarity(
  const AVecA, AVecB: TArray<Single>): Single;
var
  LI: Integer;
  LDot: Single;
begin
  Result := 0.0;
  if (Length(AVecA) = 0) or (Length(AVecA) <> Length(AVecB)) then
    Exit;

  LDot := 0.0;
  for LI := 0 to High(AVecA) do
    LDot := LDot + AVecA[LI] * AVecB[LI];
  Result := LDot;
end;

procedure TVdxEmbeddings.ApplyLinear(const AIn: array of Single;
  const AWeights: array of Single; const AInDim, AOutDim: Integer;
  out AOut: TArray<Single>);
var
  LO: Integer;
  LI: Integer;
  LAcc: Single;
  LRowBase: Integer;
begin
  SetLength(AOut, AOutDim);
  for LO := 0 to AOutDim - 1 do
  begin
    LRowBase := LO * AInDim;
    LAcc := 0.0;
    for LI := 0 to AInDim - 1 do
      LAcc := LAcc + AWeights[LRowBase + LI] * AIn[LI];
    AOut[LO] := LAcc;
  end;
end;

function TVdxEmbeddings.TryLoadDenseProjections(): Boolean;
const
  CNamePairs: array[0..3, 0..1] of string = (
    ('dense_2.weight',        'dense_3.weight'),
    ('2_Dense.weight',        '3_Dense.weight'),
    ('pooler.dense_1.weight', 'pooler.dense_2.weight'),
    ('dense.0.weight',        'dense.1.weight')
  );
var
  LPairIdx: Integer;
  LName1: string;
  LName2: string;
  LInfo1: TVdxGGUFTensorInfo;
  LInfo2: TVdxGGUFTensorInfo;
  LPtr1: Pointer;
  LPtr2: Pointer;
  LCount1: Integer;
  LCount2: Integer;
begin
  Result := False;

  for LPairIdx := 0 to High(CNamePairs) do
  begin
    LName1 := CNamePairs[LPairIdx, 0];
    LName2 := CNamePairs[LPairIdx, 1];
    if FModel.Reader.HasTensor(LName1) and FModel.Reader.HasTensor(LName2) then
    begin
      if not FModel.Reader.GetTensorInfo(LName1, LInfo1) then Continue;
      if not FModel.Reader.GetTensorInfo(LName2, LInfo2) then Continue;

      if (LInfo1.TensorType <> gtF32) or (LInfo2.TensorType <> gtF32) then
      begin
        Status('  Dense projections found (%s, %s) but not F32 (%s, %s) '
             + '— skipping for now',
          [LName1, LName2,
           VdxGGMLTypeName(LInfo1.TensorType),
           VdxGGMLTypeName(LInfo2.TensorType)]);
        Exit;
      end;

      FDense1In  := Integer(LInfo1.Dimensions[0]);
      FDense1Out := Integer(LInfo1.Dimensions[1]);
      FDense2In  := Integer(LInfo2.Dimensions[0]);
      FDense2Out := Integer(LInfo2.Dimensions[1]);

      LCount1 := FDense1In * FDense1Out;
      LCount2 := FDense2In * FDense2Out;

      LPtr1 := FModel.Reader.GetTensorDataPtr(LName1);
      LPtr2 := FModel.Reader.GetTensorDataPtr(LName2);

      SetLength(FDense1Weights, LCount1);
      SetLength(FDense2Weights, LCount2);
      Move(LPtr1^, FDense1Weights[0], LCount1 * SizeOf(Single));
      Move(LPtr2^, FDense2Weights[0], LCount2 * SizeOf(Single));

      Status('  Dense projections loaded: %s [%d->%d], %s [%d->%d]',
        [LName1, FDense1In, FDense1Out,
         LName2, FDense2In, FDense2Out]);
      FEmbeddingDim := FDense2Out;
      Result := True;
      Exit;
    end;
  end;

  Status('  Dense projections NOT found — tried %d name pair(s). ' +
    'Embed() will return normalized pooled hidden states without ' +
    'the final 768->3072->768 projection.', [Length(CNamePairs)]);
end;

function TVdxEmbeddings.LoadModel(const AGGUFPath: string): Boolean;
begin
  Result := False;
  FErrors.Clear();
  FireEvent(eeLoadStart);

  if FModelLoaded then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Model already loaded — call UnloadModel() first');
    Exit;
  end;

  // Create model via factory — no max context override for embeddings,
  // the model's native context length is used directly.
  FModel := TVdxModel.LoadModel(AGGUFPath, 0,
    FStatusCallback.Callback, FStatusCallback.UserData);
  if FModel = nil then
  begin
    FErrors.Add(esFatal, 'LOAD', 'TVdxModel.LoadModel returned nil for: %s',
      [AGGUFPath]);
    Exit;
  end;

  // Wire shared error buffer
  FModel.SetErrors(FErrors);

  if FErrors.HasFatal() then
  begin
    FreeAndNil(FModel);
    Exit;
  end;

  // Validate embedding support
  if not FModel.SupportsEmbedding() then
  begin
    FErrors.Add(esFatal, 'ARCH',
      'Model architecture "%s" does not support embedding. ' +
      'TVdxEmbeddings requires a model with SupportsEmbedding=True.',
      [FModel.Architecture]);
    FreeAndNil(FModel);
    Exit;
  end;

  // Output dim = hidden dim unless projections override it below
  FEmbeddingDim := Integer(FModel.HiddenDim);

  // Create staging buffer for downloading ResidualMat (device-local) to CPU.
  // Sized for max sequence length — we only copy what we need per Embed() call.
  FStagingBuf := FModel.Compute.CreateGpuBuffer(
    UInt64(FModel.MaxSeqLen) * FModel.HiddenDim * SizeOf(Single),
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Post-pooling dense projections
  Status('Looking for post-pooling dense projections...');
  FProjectionsLoaded := TryLoadDenseProjections();

  FModelLoaded := True;
  Result := True;
  FireEvent(eeLoadEnd);
  Status('Embedding model loaded successfully');
end;

function TVdxEmbeddings.Embed(const AText: string;
  const AIsQuery: Boolean): TArray<Single>;
var
  LPrefixed: string;
  LTokens: TArray<Integer>;
  LWithBos: TArray<Integer>;
  LNumTokens: Integer;
  LLayer: Integer;
  LHiddenMat: array of Single;
  LMatBytes: UInt64;
  LDense1Out: TArray<Single>;
begin
  SetLength(Result, 0);

  if not FModelLoaded then
  begin
    FErrors.Add(esError, 'EMBED', 'Model not loaded');
    Exit;
  end;

  FireEvent(eeEmbedStart);

  // 1. Apply task prefix via model's template
  LPrefixed := FModel.FormatEmbedding(AText, AIsQuery);

  // 2. Tokenize with BOS prepended
  LTokens := FModel.Tokenizer.Encode(LPrefixed, False);
  SetLength(LWithBos, Length(LTokens) + 1);
  LWithBos[0] := FModel.Tokenizer.GetBosId();
  if Length(LTokens) > 0 then
    Move(LTokens[0], LWithBos[1], Length(LTokens) * SizeOf(Integer));

  // 3. Clamp to max context
  LNumTokens := Length(LWithBos);
  if LNumTokens > Integer(FModel.MaxSeqLen) then
  begin
    LNumTokens := Integer(FModel.MaxSeqLen);
    SetLength(LWithBos, LNumTokens);
  end;

  Status('Embedding %d tokens (query=%s)',
    [LNumTokens, BoolToStr(AIsQuery, True)]);

  // 4. Run forward pass in one GPU batch
  FModel.Compute.BeginBatch();
  FModel.EmbedTokensBatch(LWithBos, LNumTokens, FModel.ResidualMatBuffer);
  for LLayer := 0 to Integer(FModel.NumLayers) - 1 do
    FModel.RunLayerForwardBatch(LLayer, UInt32(LNumTokens), 0, True);
  FModel.Compute.EndBatch();

  // 5. Apply final output norm on residual matrix (own batch internally)
  FModel.ApplyOutputNormBatch(UInt32(LNumTokens));

  // 6. Download token-wise hidden states via staging buffer.
  // ResidualMat is device-local, so we CopyBuffer → staging → CPU.
  LMatBytes := UInt64(LNumTokens) * FModel.HiddenDim * SizeOf(Single);
  SetLength(LHiddenMat, LNumTokens * Integer(FModel.HiddenDim));
  FModel.Compute.CopyBuffer(FModel.ResidualMatBuffer, FStagingBuf, LMatBytes);
  FModel.Compute.DownloadFromBuffer(FStagingBuf, @LHiddenMat[0], LMatBytes);

  // 7. Mean-pool across tokens → single HiddenDim vector
  Result := MeanPool(LHiddenMat, LNumTokens);

  // 7a. Post-pooling dense projections (if GGUF included them)
  if FProjectionsLoaded then
  begin
    ApplyLinear(Result, FDense1Weights, FDense1In, FDense1Out, LDense1Out);
    ApplyLinear(LDense1Out, FDense2Weights, FDense2In, FDense2Out, Result);
  end;

  // 8. L2 normalize
  L2Normalize(Result);

  FireEvent(eeEmbedEnd);
end;

procedure TVdxEmbeddings.UnloadModel();
begin
  if not FModelLoaded then
    Exit;
  FireEvent(eeUnloadStart);

  // Destroy staging buffer before freeing model (it owns FCompute)
  if (FModel <> nil) and (FStagingBuf.Buffer <> VK_NULL_HANDLE) then
  begin
    FModel.Compute.DestroyGpuBuffer(FStagingBuf);
    FillChar(FStagingBuf, SizeOf(FStagingBuf), 0);
  end;

  FreeAndNil(FModel);

  FModelLoaded := False;
  FEmbeddingDim := 0;
  SetLength(FDense1Weights, 0);
  SetLength(FDense2Weights, 0);
  FProjectionsLoaded := False;
  FDense1In := 0;
  FDense1Out := 0;
  FDense2In := 0;
  FDense2Out := 0;

  FireEvent(eeUnloadEnd);
end;

end.
