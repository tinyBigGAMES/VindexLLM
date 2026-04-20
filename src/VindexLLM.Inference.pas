{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Inference;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Classes,
  System.Math,
  System.Diagnostics,
  System.Generics.Collections,
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.VirtualBuffer,
  VindexLLM.Model,
  VindexLLM.Sampler,
  VindexLLM.Compute;

type

  { TVdxStopReason }
  TVdxStopReason = (
    srNone,              // Not yet generated
    srEOS,               // End-of-sequence token
    srStopToken,         // User-defined stop token (e.g., <end_of_turn>)
    srMaxTokens,         // Reached max token limit
    srContextFull,       // Context length exhausted during generation
    srCancelled          // Cancel callback returned True
  );

  { TVdxVRAMUsage }
  TVdxVRAMUsage = record
    WeightsBytes: UInt64;
    CacheBytes: UInt64;
    BuffersBytes: UInt64;
    TotalBytes: UInt64;
  end;

  { TVdxInferenceStats }
  TVdxInferenceStats = record
    PrefillTokens: Integer;
    PrefillTimeMs: Double;
    PrefillTokPerSec: Double;
    GeneratedTokens: Integer;
    GenerationTimeMs: Double;
    GenerationTokPerSec: Double;
    TimeToFirstTokenMs: Double;
    TotalTimeMs: Double;
    StopReason: TVdxStopReason;
    VRAMUsage: TVdxVRAMUsage;
  end;
  PVdxInferenceStats = ^TVdxInferenceStats;

  { TVdxTokenCallback }
  TVdxTokenCallback = reference to procedure(
    const AToken: string;
    const AUserData: Pointer);

  { TVdxInferenceEvent }
  TVdxInferenceEvent = (
    ieLoadStart,
    ieLoadEnd,
    ieUnloadStart,
    ieUnloadEnd,
    iePrefillStart,
    iePrefillEnd,
    ieGenerateStart,
    ieGenerateEnd
  );

  { TVdxInferenceEventCallback }
  TVdxInferenceEventCallback = reference to procedure(
    const AEvent: TVdxInferenceEvent;
    const AUserData: Pointer);

  { TVdxCancelCallback }
  TVdxCancelCallback = reference to function(
    const AUserData: Pointer): Boolean;

  { TVdxRebuildCallback }
  TVdxRebuildCallback = reference to function(
    const APosition: UInt32;
    const AMaxContext: UInt32;
    const APrompt: string;
    const AUserData: Pointer): string;

  { TVdxInference }
  TVdxInference = class(TVdxBaseObject)
  private
    // Model — owns all subsystems, GPU plumbing, forward pass
    FModel: TVdxModel;

    // Sampler
    FSampler: TVdxSampler;

    // Callbacks
    FTokenCallback: TVdxCallback<TVdxTokenCallback>;
    FEventCallback: TVdxCallback<TVdxInferenceEventCallback>;
    FCancelCallback: TVdxCallback<TVdxCancelCallback>;
    FRebuildCallback: TVdxCallback<TVdxRebuildCallback>;

    // Rebuild threshold — absolute token position at which Generate()
    // invokes the rebuild callback. Zero means disabled.
    FRebuildAt: UInt32;

    // Model state
    FModelLoaded: Boolean;

    // KV cache next-write position
    FCurrentPosition: UInt32;

    // Stop token IDs — generation stops when any of these are produced
    FStopTokenIds: TList<Integer>;

    // Stats — filled by Generate()
    FStats: TVdxInferenceStats;

    // VRAM usage — filled by LoadModel(), copied into FStats per Generate()
    FVRAMUsage: TVdxVRAMUsage;

    // Private helpers
    procedure DownloadLayerCache(const ALayerCache: TVdxGpuBuffer;
      const AStaging: TVdxGpuBuffer; const ACacheBuffer: Pointer);
    procedure UploadLayerCache(const ALayerCache: TVdxGpuBuffer;
      const AStaging: TVdxGpuBuffer; const ACacheBuffer: Pointer);
    function RunUnembedding(): Integer;
    procedure FireEvent(const AEvent: TVdxInferenceEvent);
    function IsCancelled(): Boolean;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    function LoadModel(const AGGUFPath: string;
      const AMaxContext: Integer = 2048;
      const ARebuildAt: Integer = -1): Boolean;

    procedure SetTokenCallback(const ACallback: TVdxTokenCallback;
      const AUserData: Pointer);

    procedure SetInferenceEventCallback(
      const ACallback: TVdxInferenceEventCallback;
      const AUserData: Pointer);

    procedure SetCancelCallback(const ACallback: TVdxCancelCallback;
      const AUserData: Pointer);

    procedure SetRebuildCallback(const ACallback: TVdxRebuildCallback;
      const AUserData: Pointer);

    function GetRebuildAt(): UInt32;

    function Generate(const APrompt: string;
      const AMaxTokens: Integer = 256;
      const AFormat: Boolean = True): string;

    procedure UnloadModel();

    procedure AddStopToken(const ATokenId: Integer);
    procedure ClearStopTokens();

    procedure SetSamplerConfig(const AConfig: TVdxSamplerConfig);

    function GetKVCachePosition(): UInt32;
    procedure ResetKVCache();
    function SaveKVCache(const AFilename: string): Boolean;
    function LoadKVCache(const AFilename: string): Boolean;

    function GetStats(): PVdxInferenceStats;

    // Model accessor (read-only)
    property Model: TVdxModel read FModel;
  end;

const
  { CVdxStopReasons }
  CVdxStopReasons: array[TVdxStopReason] of string = (
    'none', 'eos', 'stop_token', 'max_tokens', 'context_full', 'cancelled');

  { CVdxEventNames }
  CVdxEventNames: array[TVdxInferenceEvent] of string = (
    'load start', 'load end',
    'unload start', 'unload end',
    'prefill start', 'prefill end',
    'generate start', 'generate end'
  );

implementation

uses
  VindexLLM.GGUFReader,
  VindexLLM.Attention,
  VindexLLM.Tokenizer;

{ TVdxInference }
constructor TVdxInference.Create();
begin
  inherited;
  FModel := nil;
  FSampler := nil;
  FModelLoaded := False;
  FStopTokenIds := TList<Integer>.Create();
  FCurrentPosition := 0;
  FRebuildAt := 0;
end;

destructor TVdxInference.Destroy();
begin
  if FModelLoaded then
    UnloadModel();
  FreeAndNil(FStopTokenIds);
  inherited;
end;

procedure TVdxInference.FireEvent(const AEvent: TVdxInferenceEvent);
begin
  if FEventCallback.IsAssigned() then
    FEventCallback.Callback(AEvent, FEventCallback.UserData);
end;

function TVdxInference.IsCancelled(): Boolean;
begin
  Result := FCancelCallback.IsAssigned() and
    FCancelCallback.Callback(FCancelCallback.UserData);
end;

procedure TVdxInference.SetTokenCallback(const ACallback: TVdxTokenCallback;
  const AUserData: Pointer);
begin
  FTokenCallback.Callback := ACallback;
  FTokenCallback.UserData := AUserData;
end;

procedure TVdxInference.SetInferenceEventCallback(
  const ACallback: TVdxInferenceEventCallback;
  const AUserData: Pointer);
begin
  FEventCallback.Callback := ACallback;
  FEventCallback.UserData := AUserData;
end;

procedure TVdxInference.SetCancelCallback(const ACallback: TVdxCancelCallback;
  const AUserData: Pointer);
begin
  FCancelCallback.Callback := ACallback;
  FCancelCallback.UserData := AUserData;
end;

procedure TVdxInference.SetRebuildCallback(
  const ACallback: TVdxRebuildCallback;
  const AUserData: Pointer);
begin
  FRebuildCallback.Callback := ACallback;
  FRebuildCallback.UserData := AUserData;
end;

function TVdxInference.GetRebuildAt(): UInt32;
begin
  Result := FRebuildAt;
end;

procedure TVdxInference.SetSamplerConfig(const AConfig: TVdxSamplerConfig);
begin
  FSampler.SetConfig(AConfig);
end;

procedure TVdxInference.AddStopToken(const ATokenId: Integer);
begin
  if not FStopTokenIds.Contains(ATokenId) then
    FStopTokenIds.Add(ATokenId);
end;

procedure TVdxInference.ClearStopTokens();
begin
  FStopTokenIds.Clear();
  if FModelLoaded then
    FStopTokenIds.Add(FModel.Tokenizer.GetEosId());
end;

function TVdxInference.GetStats(): PVdxInferenceStats;
begin
  Result := @FStats;
end;

function TVdxInference.GetKVCachePosition(): UInt32;
begin
  Result := FCurrentPosition;
end;

procedure TVdxInference.ResetKVCache();
begin
  FCurrentPosition := 0;
end;

procedure TVdxInference.DownloadLayerCache(const ALayerCache: TVdxGpuBuffer;
  const AStaging: TVdxGpuBuffer; const ACacheBuffer: Pointer);
var
  LSize: UInt64;
begin
  LSize := FModel.Attn.GetLayerKVCacheTQ3Bytes();
  FModel.Compute.CopyBuffer(ALayerCache, AStaging, LSize);
  FModel.Compute.DownloadFromBuffer(AStaging, ACacheBuffer, LSize);
end;

procedure TVdxInference.UploadLayerCache(const ALayerCache: TVdxGpuBuffer;
  const AStaging: TVdxGpuBuffer; const ACacheBuffer: Pointer);
var
  LSize: UInt64;
begin
  LSize := FModel.Attn.GetLayerKVCacheTQ3Bytes();
  FModel.Compute.UploadToBuffer(AStaging, ACacheBuffer, LSize);
  FModel.Compute.CopyBuffer(AStaging, ALayerCache, LSize);
end;

function TVdxInference.RunUnembedding(): Integer;
begin
  FModel.UnembedToLogits(FModel.LogitsBuffer);

  // Download logits into model's pre-allocated VirtualBuffer
  FModel.Compute.DownloadFromBuffer(FModel.LogitsBuffer,
    FModel.LogitsVBuf.Memory,
    UInt64(FModel.VocabSize) * SizeOf(Single));

  // Sample next token
  Result := FSampler.Process(System.PSingle(FModel.LogitsVBuf.Memory),
    FModel.VocabSize);
end;

function TVdxInference.LoadModel(const AGGUFPath: string;
  const AMaxContext: Integer;
  const ARebuildAt: Integer): Boolean;
var
  LStopStrings: TArray<string>;
  LStopStr: string;
  LProbeIds: TArray<Integer>;
  LKVPerLayer: UInt64;
  LKVTotal: UInt64;
  LVramWeights: UInt64;
  LVramBuffers: UInt64;
begin
  Result := False;
  FErrors.Clear();
  FireEvent(ieLoadStart);

  if FModelLoaded then
  begin
    FErrors.Add(esFatal, 'LOAD', 'Model already loaded — call UnloadModel() first');
    Exit;
  end;

  // Create model via factory
  FModel := TVdxModel.LoadModel(AGGUFPath, AMaxContext,
    FStatusCallback.Callback, FStatusCallback.UserData);
  if FModel = nil then
  begin
    FErrors.Add(esFatal, 'LOAD', 'TVdxModel.LoadModel returned nil for: %s',
      [AGGUFPath]);
    Exit;
  end;

  // Wire shared error buffer
  FModel.SetErrors(FErrors);

  // Check for load errors from the model
  if FErrors.HasFatal() then
  begin
    FreeAndNil(FModel);
    Exit;
  end;

  // Rebuild threshold
  if ARebuildAt <= 0 then
    FRebuildAt := (FModel.MaxSeqLen * 3) div 4
  else
    FRebuildAt := Min(UInt32(ARebuildAt), FModel.MaxSeqLen);
  Status('Rebuild threshold: %d tokens (%.0f%% of %d)',
    [FRebuildAt, (FRebuildAt * 100.0) / FModel.MaxSeqLen, FModel.MaxSeqLen]);

  // Create sampler
  FSampler := TVdxSampler.Create();
  FSampler.SetErrors(FErrors);

  // Populate stop tokens from model
  FStopTokenIds.Clear();
  FStopTokenIds.Add(FModel.Tokenizer.GetEosId());

  // EOT from GGUF metadata
  if FModel.Reader.HasMetadata('tokenizer.ggml.eot_token_id') then
    FStopTokenIds.Add(
      Integer(FModel.Reader.GetMetadataUInt32('tokenizer.ggml.eot_token_id')));

  // Model-reported stop token strings
  LStopStrings := FModel.GetStopTokenStrings();
  for LStopStr in LStopStrings do
  begin
    LProbeIds := FModel.Tokenizer.Encode(LStopStr, False);
    if (Length(LProbeIds) = 1) and (not FStopTokenIds.Contains(LProbeIds[0])) then
    begin
      FStopTokenIds.Add(LProbeIds[0]);
      Status('  Found stop token: id=%d ("%s")',
        [LProbeIds[0], FModel.Tokenizer.GetTokenStr(LProbeIds[0])]);
    end;
  end;

  // Probe vocab for common end-of-turn tokens across model families
  LProbeIds := FModel.Tokenizer.Encode('<|im_end|>', False);
  if (Length(LProbeIds) = 1) and (not FStopTokenIds.Contains(LProbeIds[0])) then
    FStopTokenIds.Add(LProbeIds[0]);
  LProbeIds := FModel.Tokenizer.Encode('<|eot_id|>', False);
  if (Length(LProbeIds) = 1) and (not FStopTokenIds.Contains(LProbeIds[0])) then
    FStopTokenIds.Add(LProbeIds[0]);
  LProbeIds := FModel.Tokenizer.Encode('<|end|>', False);
  if (Length(LProbeIds) = 1) and (not FStopTokenIds.Contains(LProbeIds[0])) then
    FStopTokenIds.Add(LProbeIds[0]);
  LProbeIds := FModel.Tokenizer.Encode('<|endoftext|>', False);
  if (Length(LProbeIds) = 1) and (not FStopTokenIds.Contains(LProbeIds[0])) then
    FStopTokenIds.Add(LProbeIds[0]);

  Status('Stop tokens: %d configured', [FStopTokenIds.Count]);

  // --- VRAM usage summary (computed from model dimensions) ---
  LKVPerLayer := UInt64(2) * FModel.MaxSeqLen * FModel.NumKVHeads *
    FModel.HeadDim * SizeOf(Single);
  LKVTotal := UInt64(FModel.NumLayers) * LKVPerLayer;

  LVramWeights :=
    UInt64(FModel.NumLayers) * VdxGGMLTensorBytes(FModel.WeightType, FModel.HiddenDim, FModel.FFNWidth) +
    UInt64(FModel.NumLayers) * VdxGGMLTensorBytes(FModel.WeightType, FModel.FFNWidth, FModel.HiddenDim) +
    UInt64(FModel.NumLayers) * VdxGGMLTensorBytes(FModel.WeightType, FModel.HiddenDim, FModel.FFNWidth) +
    UInt64(FModel.NumLayers) * VdxGGMLTensorBytes(FModel.WeightType, FModel.HiddenDim, FModel.NumQHeads * FModel.HeadDim) +
    UInt64(FModel.NumLayers) * VdxGGMLTensorBytes(FModel.WeightType, FModel.HiddenDim, FModel.NumKVHeads * FModel.HeadDim) +
    UInt64(FModel.NumLayers) * VdxGGMLTensorBytes(FModel.WeightType, FModel.HiddenDim, FModel.NumKVHeads * FModel.HeadDim) +
    UInt64(FModel.NumLayers) * VdxGGMLTensorBytes(FModel.WeightType, FModel.NumQHeads * FModel.HeadDim, FModel.HiddenDim) +
    UInt64(FModel.NumLayers) * (4 * FModel.HiddenDim + 2 * FModel.HeadDim) * SizeOf(Single) +
    UInt64(FModel.HiddenDim) * SizeOf(Single) +
    VdxGGMLTensorBytes(FModel.EmbedType, FModel.HiddenDim, UInt64(FModel.VocabSize));

  LVramBuffers :=
    UInt64(4) * FModel.HiddenDim * SizeOf(Single) +
    UInt64(2) * FModel.FFNWidth * SizeOf(Single) +
    UInt64(4) * FModel.MaxSeqLen * FModel.HiddenDim * SizeOf(Single) +
    UInt64(FModel.MaxSeqLen) * FModel.NumQHeads * FModel.HeadDim * SizeOf(Single) +
    UInt64(2) * FModel.MaxSeqLen * FModel.NumKVHeads * FModel.HeadDim * SizeOf(Single) +
    UInt64(2) * FModel.MaxSeqLen * FModel.FFNWidth * SizeOf(Single) +
    UInt64(FModel.VocabSize) * SizeOf(Single) +
    UInt64(FModel.MaxSeqLen) * SizeOf(UInt32);

  FVRAMUsage.WeightsBytes := LVramWeights;
  FVRAMUsage.CacheBytes := LKVTotal;
  FVRAMUsage.BuffersBytes := LVramBuffers;
  FVRAMUsage.TotalBytes := LVramWeights + LKVTotal + LVramBuffers;

  Status('VRAM usage: %d MB (weights: %d, cache: %d, buffers: %d)',
    [FVRAMUsage.TotalBytes div (1024 * 1024),
     LVramWeights div (1024 * 1024),
     LKVTotal div (1024 * 1024),
     LVramBuffers div (1024 * 1024)]);

  FModelLoaded := True;
  Result := True;
  FireEvent(ieLoadEnd);
  Status('Model loaded successfully');
end;

function TVdxInference.Generate(const APrompt: string;
  const AMaxTokens: Integer; const AFormat: Boolean): string;
var
  LFormatted: string;
  LTokenIds: TArray<Integer>;
  LTokenCount: Integer;
  LLayer: Integer;
  LNextTokenId: Integer;
  LTokenStr: string;
  LResult: TStringBuilder;
  LGenerated: Integer;
  LTotalWatch: TStopwatch;
  LPrefillWatch: TStopwatch;
  LGenWatch: TStopwatch;
  LEffectivePrompt: string;
  LReplacement: string;
  LRebuilt: Boolean;
begin
  Result := '';
  FErrors.Clear();

  if not FModelLoaded then
  begin
    FErrors.Add(esError, 'GEN', 'Model not loaded');
    Exit;
  end;

  // Lazy rebuild: if the cache has crossed the configured threshold and a
  // handler is installed, give it a chance to return a replacement prompt.
  LEffectivePrompt := APrompt;
  LRebuilt := False;
  if FRebuildCallback.IsAssigned() and
     (FRebuildAt > 0) and
     (FCurrentPosition >= FRebuildAt) then
  begin
    LReplacement := FRebuildCallback.Callback(
      FCurrentPosition, FModel.MaxSeqLen, APrompt,
      FRebuildCallback.UserData);

    // Callback firing IS the rebuild signal — always reset the cache.
    ResetKVCache();
    if LReplacement <> '' then
    begin
      LEffectivePrompt := LReplacement;
      LRebuilt := True;
    end;
  end;

  // Reset stats (preserve VRAM usage from LoadModel)
  FStats := Default(TVdxInferenceStats);
  FStats.VRAMUsage := FVRAMUsage;

  // Format prompt with model's template. Skip when the caller already
  // formatted (AFormat=False), UNLESS a rebuild just produced a raw
  // replacement prompt that needs wrapping.
  if AFormat or LRebuilt then
    LFormatted := FModel.FormatPrompt(LEffectivePrompt)
  else
    LFormatted := LEffectivePrompt;

  // Tokenize. Only add BOS on fresh sessions.
  LTokenIds := FModel.Tokenizer.Encode(LFormatted, FCurrentPosition = 0);
  LTokenCount := Length(LTokenIds);

  // Guard: context overflow check with reactive rebuild fallback
  if UInt64(FCurrentPosition) + UInt64(LTokenCount) + UInt64(AMaxTokens)
    > UInt64(FModel.MaxSeqLen) then
  begin
    if (not LRebuilt) and FRebuildCallback.IsAssigned() and
       (FCurrentPosition > 0) then
    begin
      LReplacement := FRebuildCallback.Callback(
        FCurrentPosition, FModel.MaxSeqLen, APrompt,
        FRebuildCallback.UserData);

      ResetKVCache();

      if LReplacement <> '' then
      begin
        LEffectivePrompt := LReplacement;
        LFormatted := FModel.FormatPrompt(LEffectivePrompt);
      end;

      // Re-tokenize (BOS added since FCurrentPosition is now 0)
      LTokenIds := FModel.Tokenizer.Encode(LFormatted, True);
      LTokenCount := Length(LTokenIds);
    end;

    // Final check — if still overflowing after rebuild, give up
    if UInt64(FCurrentPosition) + UInt64(LTokenCount) + UInt64(AMaxTokens)
      > UInt64(FModel.MaxSeqLen) then
    begin
      FStats.StopReason := srContextFull;
      FErrors.Add(esError, 'GEN',
        'Context overflow: pos=%d + prompt=%d + max=%d exceeds max context %d',
        [FCurrentPosition, LTokenCount, AMaxTokens, FModel.MaxSeqLen]);
      Exit;
    end;
  end;

  LTotalWatch := TStopwatch.StartNew();
  LResult := TStringBuilder.Create();
  try
    // Reset sampler history for this generation
    FSampler.ResetHistory();

    // --- Prefill: batch all prompt tokens through the model ---
    FireEvent(iePrefillStart);
    LPrefillWatch := TStopwatch.StartNew();

    FModel.Compute.BeginBatch();
    FModel.EmbedTokensBatch(LTokenIds, LTokenCount, FModel.ResidualMatBuffer);
    for LLayer := 0 to Integer(FModel.NumLayers) - 1 do
    begin
      if IsCancelled() then
      begin
        FStats.StopReason := srCancelled;
        Break;
      end;
      FModel.RunLayerForwardBatch(LLayer, UInt32(LTokenCount), FCurrentPosition);
    end;
    FModel.Compute.EndBatch();

    // Advance write position past the prefilled tokens
    if FStats.StopReason <> srCancelled then
      FCurrentPosition := FCurrentPosition + UInt32(LTokenCount);

    // Copy last token's residual from matrix to vector for generation handoff
    FModel.SeedResidualFromBatchLast(UInt32(LTokenCount));

    LPrefillWatch.Stop();
    FireEvent(iePrefillEnd);

    // --- Autoregressive generation ---
    LGenerated := 0;
    FillChar(LGenWatch, SizeOf(LGenWatch), 0);

    // Skip generation if cancelled during prefill
    if FStats.StopReason <> srCancelled then
    begin
      FireEvent(ieGenerateStart);
      LGenWatch := TStopwatch.StartNew();
      while LGenerated < AMaxTokens do
      begin
        // Check context overflow
        if FCurrentPosition >= FModel.MaxSeqLen then
        begin
          FStats.StopReason := srContextFull;
          Break;
        end;

        // Check cancel before unembedding
        if IsCancelled() then
        begin
          FStats.StopReason := srCancelled;
          Break;
        end;

        LNextTokenId := RunUnembedding();

        // Check for stop tokens
        if FStopTokenIds.Contains(LNextTokenId) then
        begin
          if LNextTokenId = FModel.Tokenizer.GetEosId() then
            FStats.StopReason := srEOS
          else
            FStats.StopReason := srStopToken;
          Break;
        end;

        // Decode token and append to result
        LTokenStr := FModel.Tokenizer.Decode(
          TArray<Integer>.Create(LNextTokenId));
        LResult.Append(LTokenStr);

        // Track token for repetition penalty
        FSampler.AddToHistory(LNextTokenId);

        // Notify token callback if assigned
        if FTokenCallback.IsAssigned() then
          FTokenCallback.Callback(LTokenStr, FTokenCallback.UserData);

        // Feed predicted token back into the model
        FModel.Compute.BeginBatch();
        FModel.EmbedToken(LNextTokenId);
        for LLayer := 0 to Integer(FModel.NumLayers) - 1 do
        begin
          if IsCancelled() then
          begin
            FStats.StopReason := srCancelled;
            Break;
          end;
          FModel.RunLayerForward(LLayer, Integer(FCurrentPosition));
        end;
        FModel.Compute.EndBatch();

        // Break outer loop if cancelled during forward pass
        if FStats.StopReason = srCancelled then
          Break;

        // Successful token write — advance past this slot
        FCurrentPosition := FCurrentPosition + 1;
        Inc(LGenerated);
      end;
      LGenWatch.Stop();
      FireEvent(ieGenerateEnd);
    end; // if not cancelled during prefill

    // Max tokens reached without a stop token
    if FStats.StopReason = srNone then
      FStats.StopReason := srMaxTokens;

    LTotalWatch.Stop();

    // Fill stats
    FStats.PrefillTokens := LTokenCount;
    FStats.PrefillTimeMs := LPrefillWatch.Elapsed.TotalMilliseconds;
    if FStats.PrefillTimeMs > 0 then
      FStats.PrefillTokPerSec := (LTokenCount * 1000.0) / FStats.PrefillTimeMs;

    FStats.GeneratedTokens := LGenerated;
    FStats.GenerationTimeMs := LGenWatch.Elapsed.TotalMilliseconds;
    if FStats.GenerationTimeMs > 0 then
      FStats.GenerationTokPerSec := (LGenerated * 1000.0) / FStats.GenerationTimeMs;

    FStats.TimeToFirstTokenMs := LPrefillWatch.Elapsed.TotalMilliseconds;
    FStats.TotalTimeMs := LTotalWatch.Elapsed.TotalMilliseconds;

    Result := LResult.ToString();
  finally
    LResult.Free();
  end;
end;

procedure TVdxInference.UnloadModel();
begin
  if not FModelLoaded then
  begin
    // Partial load failed — free what was created
    FreeAndNil(FSampler);
    FreeAndNil(FModel);
    Exit;
  end;

  FireEvent(ieUnloadStart);

  FreeAndNil(FSampler);
  FreeAndNil(FModel);

  FModelLoaded := False;
  FCurrentPosition := 0;

  FireEvent(ieUnloadEnd);
  Status('Model unloaded');
end;

type
  TVdxKVCacheHeader = packed record
    Magic: UInt32;
    Version: UInt32;
    NumLayers: UInt32;
    NumKVHeads: UInt32;
    HeadDim: UInt32;
    MaxSeqLen: UInt32;
    CurrentPosition: UInt32;
    Reserved: UInt32;
    ModelFingerprint: array[0..31] of Byte;
  end;

const
  CVdxKVCacheMagic: UInt32 = $43564B56;  // 'VKVC' little-endian
  CVdxKVCacheVersion: UInt32 = 1;

function TVdxInference.SaveKVCache(const AFilename: string): Boolean;
var
  LHeader: TVdxKVCacheHeader;
  LStream: TFileStream;
  LStaging: TVdxGpuBuffer;
  LLayerBuf: Pointer;
  LSize: UInt64;
  LLayer: Integer;
begin
  Result := False;
  FErrors.Clear();

  if not FModelLoaded then
  begin
    FErrors.Add(esError, 'SAVE', 'Model not loaded');
    Exit;
  end;

  if FCurrentPosition = 0 then
  begin
    FErrors.Add(esError, 'SAVE',
      'KV cache is empty (position = 0) — nothing to save');
    Exit;
  end;

  // Build header
  FillChar(LHeader, SizeOf(LHeader), 0);
  LHeader.Magic := CVdxKVCacheMagic;
  LHeader.Version := CVdxKVCacheVersion;
  LHeader.NumLayers := FModel.NumLayers;
  LHeader.NumKVHeads := FModel.NumKVHeads;
  LHeader.HeadDim := FModel.HeadDim;
  LHeader.MaxSeqLen := FModel.MaxSeqLen;
  LHeader.CurrentPosition := FCurrentPosition;

  LSize := FModel.Attn.GetLayerKVCacheTQ3Bytes();

  LStaging := FModel.Compute.CreateGpuBuffer(
    LSize,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  try
    GetMem(LLayerBuf, LSize);
    try
      try
        LStream := TFileStream.Create(AFilename, fmCreate);
        try
          LStream.WriteBuffer(LHeader, SizeOf(LHeader));

          for LLayer := 0 to Integer(FModel.NumLayers) - 1 do
          begin
            DownloadLayerCache(FModel.Attn.GetLayerKCacheTQ3(LLayer),
              LStaging, LLayerBuf);
            LStream.WriteBuffer(LLayerBuf^, LSize);

            DownloadLayerCache(FModel.Attn.GetLayerVCacheTQ3(LLayer),
              LStaging, LLayerBuf);
            LStream.WriteBuffer(LLayerBuf^, LSize);
          end;
        finally
          LStream.Free();
        end;

        Result := True;
      except
        on E: Exception do
        begin
          FErrors.Add(esError, 'SAVE',
            'Failed to write %s: %s', [AFilename, E.Message]);
          Result := False;
        end;
      end;
    finally
      FreeMem(LLayerBuf);
    end;
  finally
    FModel.Compute.DestroyGpuBuffer(LStaging);
  end;
end;

function TVdxInference.LoadKVCache(const AFilename: string): Boolean;
var
  LHeader: TVdxKVCacheHeader;
  LStream: TFileStream;
  LStaging: TVdxGpuBuffer;
  LLayerBuf: Pointer;
  LSize: UInt64;
  LExpectedPayload: UInt64;
  LLayer: Integer;
  LCurrentLayer: Integer;
begin
  Result := False;
  FErrors.Clear();

  if not FModelLoaded then
  begin
    FErrors.Add(esError, 'LOAD', 'Model not loaded');
    Exit;
  end;

  LSize := FModel.Attn.GetLayerKVCacheTQ3Bytes();
  LExpectedPayload := UInt64(FModel.NumLayers) * 2 * LSize;

  try
    LStream := TFileStream.Create(AFilename, fmOpenRead or fmShareDenyWrite);
  except
    on E: Exception do
    begin
      FErrors.Add(esError, 'LOAD',
        'Failed to open %s: %s', [AFilename, E.Message]);
      Exit;
    end;
  end;

  try
    try
      if LStream.Size < SizeOf(LHeader) then
      begin
        FErrors.Add(esError, 'LOAD',
          'File too small to contain header (%d bytes)', [LStream.Size]);
        Exit;
      end;

      LStream.ReadBuffer(LHeader, SizeOf(LHeader));

      if LHeader.Magic <> CVdxKVCacheMagic then
      begin
        FErrors.Add(esError, 'LOAD',
          'Bad magic: expected VKVC, got $%08x', [LHeader.Magic]);
        Exit;
      end;

      if LHeader.Version <> CVdxKVCacheVersion then
      begin
        FErrors.Add(esError, 'LOAD',
          'Unsupported version %d (this build reads version %d)',
          [LHeader.Version, CVdxKVCacheVersion]);
        Exit;
      end;

      if LHeader.Reserved <> 0 then
      begin
        FErrors.Add(esError, 'LOAD',
          'Corrupt header: reserved field is nonzero (%d)',
          [LHeader.Reserved]);
        Exit;
      end;

      if LHeader.NumLayers <> FModel.NumLayers then
      begin
        FErrors.Add(esError, 'LOAD',
          'Model layer count mismatch: file=%d, model=%d',
          [LHeader.NumLayers, FModel.NumLayers]);
        Exit;
      end;

      if LHeader.NumKVHeads <> FModel.NumKVHeads then
      begin
        FErrors.Add(esError, 'LOAD',
          'KV head count mismatch: file=%d, model=%d',
          [LHeader.NumKVHeads, FModel.NumKVHeads]);
        Exit;
      end;

      if LHeader.HeadDim <> FModel.HeadDim then
      begin
        FErrors.Add(esError, 'LOAD',
          'Head dimension mismatch: file=%d, model=%d',
          [LHeader.HeadDim, FModel.HeadDim]);
        Exit;
      end;

      if LHeader.MaxSeqLen <> FModel.MaxSeqLen then
      begin
        FErrors.Add(esError, 'LOAD',
          'Max sequence length mismatch: file=%d, model=%d',
          [LHeader.MaxSeqLen, FModel.MaxSeqLen]);
        Exit;
      end;

      if LHeader.CurrentPosition > FModel.MaxSeqLen then
      begin
        FErrors.Add(esError, 'LOAD',
          'Invalid current position %d exceeds max %d',
          [LHeader.CurrentPosition, FModel.MaxSeqLen]);
        Exit;
      end;

      if UInt64(LStream.Size - SizeOf(LHeader)) < LExpectedPayload then
      begin
        FErrors.Add(esError, 'LOAD',
          'File truncated: payload %d bytes, expected %d',
          [LStream.Size - SizeOf(LHeader), LExpectedPayload]);
        Exit;
      end;

      // All validation passed. Now mutate GPU state.
      LStaging := FModel.Compute.CreateGpuBuffer(
        LSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      try
        GetMem(LLayerBuf, LSize);
        try
          LCurrentLayer := 0;
          try
            for LLayer := 0 to Integer(FModel.NumLayers) - 1 do
            begin
              LCurrentLayer := LLayer;

              // K cache
              LStream.ReadBuffer(LLayerBuf^, LSize);
              UploadLayerCache(FModel.Attn.GetLayerKCacheTQ3(LLayer),
                LStaging, LLayerBuf);

              // V cache
              LStream.ReadBuffer(LLayerBuf^, LSize);
              UploadLayerCache(FModel.Attn.GetLayerVCacheTQ3(LLayer),
                LStaging, LLayerBuf);
            end;

            FCurrentPosition := LHeader.CurrentPosition;
            Result := True;
          except
            on E: Exception do
            begin
              FErrors.Add(esError, 'LOAD',
                'Read failed mid-load at layer %d: %s (GPU state partially overwritten; call ResetKVCache to recover)',
                [LCurrentLayer, E.Message]);
              Result := False;
            end;
          end;
        finally
          FreeMem(LLayerBuf);
        end;
      finally
        FModel.Compute.DestroyGpuBuffer(LStaging);
      end;
    except
      on E: Exception do
      begin
        FErrors.Add(esError, 'LOAD',
          'Unexpected error: %s', [E.Message]);
        Result := False;
      end;
    end;
  finally
    LStream.Free();
  end;
end;

end.
