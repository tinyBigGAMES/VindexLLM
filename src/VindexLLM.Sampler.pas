{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Sampler;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  VindexLLM.Utils,
  VindexLLM.VirtualBuffer;

type

  { TVdxSamplerConfig }
  TVdxSamplerConfig = record
    Temperature: Single;      // 0.0 = greedy argmax, >0 = divide logits by this
    TopK: Integer;             // 0 = disabled, >0 = keep only top K logits
    TopP: Single;              // 1.0 = disabled, <1.0 = nucleus sampling threshold
    MinP: Single;              // 0.0 = disabled, >0 = floor relative to top logit
    RepeatPenalty: Single;     // 1.0 = disabled, >1.0 = penalize repeated tokens
    RepeatWindow: Integer;     // How many recent tokens to track for penalty
    Seed: Int64;               // 0 = random seed, >0 = deterministic
  end;

  { TVdxSampler }
  TVdxSampler = class(TVdxBaseObject)
  private
    FConfig: TVdxSamplerConfig;

    // xoshiro256** PRNG state (4 x UInt64)
    FState0: UInt64;
    FState1: UInt64;
    FState2: UInt64;
    FState3: UInt64;

    // Repetition penalty ring buffer
    FHistory: array of Integer;
    FHistoryPos: Integer;
    FHistoryCount: Integer;

    // Workspace for sampling (lazy-allocated on first non-greedy call)
    FProbsVBuf: TVdxVirtualBuffer<Single>;
    FIndicesVBuf: TVdxVirtualBuffer<Integer>;
    FTopKBuf: array of Single;
    FTopKIdx: array of Integer;
    FCachedVocabSize: Integer;

    // Overflow-safe modular arithmetic (mod 2^64)
    class function WrapAdd(const AA, AB: UInt64): UInt64; static; inline;
    class function WrapMul(const AA, AB: UInt64): UInt64; static;

    // PRNG helpers
    class function SplitMix64(var AState: UInt64): UInt64; static;
    function RotLeft(const AX: UInt64; const AK: Integer): UInt64; inline;
    function NextUInt64(): UInt64;
    function NextFloat(): Single;
    procedure SeedPRNG(const ASeed: Int64);

    // Workspace management
    procedure EnsureWorkspace(const AVocabSize: Integer);

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Configure the sampler — re-seeds PRNG if seed changed
    procedure SetConfig(const AConfig: TVdxSamplerConfig);
    function GetConfig(): TVdxSamplerConfig;

    // Returns the default (greedy) config
    class function DefaultConfig(): TVdxSamplerConfig; static;

    // Main entry: takes raw logits pointer + vocab size, returns token ID
    function Process(const ALogits: System.PSingle;
      const AVocabSize: Integer): Integer;

    // Repetition history management
    procedure AddToHistory(const ATokenId: Integer);
    procedure ResetHistory();
  end;

implementation

uses
  WinApi.Windows;

{ TVdxSampler }

class function TVdxSampler.DefaultConfig(): TVdxSamplerConfig;
begin
  Result.Temperature := 0.0;
  Result.TopK := 0;
  Result.TopP := 1.0;
  Result.MinP := 0.0;
  Result.RepeatPenalty := 1.0;
  Result.RepeatWindow := 64;
  Result.Seed := 0;
end;

constructor TVdxSampler.Create();
begin
  inherited;
  FConfig := DefaultConfig();
  FHistoryPos := 0;
  FHistoryCount := 0;
  FCachedVocabSize := 0;
  FProbsVBuf := nil;
  FIndicesVBuf := nil;
  SetLength(FHistory, FConfig.RepeatWindow);
  SeedPRNG(0);
end;

destructor TVdxSampler.Destroy();
begin
  FreeAndNil(FProbsVBuf);
  FreeAndNil(FIndicesVBuf);
  FHistory := nil;
  inherited;
end;

procedure TVdxSampler.SetConfig(const AConfig: TVdxSamplerConfig);
var
  LSeedChanged: Boolean;
begin
  LSeedChanged := AConfig.Seed <> FConfig.Seed;
  FConfig := AConfig;

  // Resize history ring buffer if window changed
  if Length(FHistory) <> FConfig.RepeatWindow then
  begin
    SetLength(FHistory, FConfig.RepeatWindow);
    FHistoryPos := 0;
    FHistoryCount := 0;
  end;

  // Re-seed PRNG if seed changed
  if LSeedChanged then
    SeedPRNG(FConfig.Seed);
end;

function TVdxSampler.GetConfig(): TVdxSamplerConfig;
begin
  Result := FConfig;
end;

// --- Overflow-safe modular arithmetic (mod 2^64) ---
// Delphi's {$Q+} traps unsigned wrapping, so we avoid it explicitly.

class function TVdxSampler.WrapAdd(const AA, AB: UInt64): UInt64;
begin
  // If adding would wrap, subtract the complement instead
  if AB <= High(UInt64) - AA then
    Result := AA + AB
  else
    Result := AB - (High(UInt64) - AA) - 1;
end;

class function TVdxSampler.WrapMul(const AA, AB: UInt64): UInt64;
var
  LALo, LAHi, LBLo, LBHi: UInt64;
  LCross1, LCross2: UInt64;
begin
  // Split into 32-bit halves — each partial product fits in UInt64
  LALo := AA and $FFFFFFFF;
  LAHi := AA shr 32;
  LBLo := AB and $FFFFFFFF;
  LBHi := AB shr 32;

  // Low x Low — always fits
  Result := LALo * LBLo;

  // Cross terms — only the low 32 bits of each product matter after shift
  LCross1 := LALo * LBHi;
  LCross2 := LAHi * LBLo;

  // Add cross terms shifted left 32 — using WrapAdd to avoid overflow
  Result := WrapAdd(Result, (LCross1 and $FFFFFFFF) shl 32);
  Result := WrapAdd(Result, (LCross2 and $FFFFFFFF) shl 32);
end;

// --- PRNG: xoshiro256** with SplitMix64 seeding ---

class function TVdxSampler.SplitMix64(var AState: UInt64): UInt64;
begin
  AState := WrapAdd(AState, UInt64($9E3779B97F4A7C15));
  Result := AState;
  Result := WrapMul(Result xor (Result shr 30), UInt64($BF58476D1CE4E5B9));
  Result := WrapMul(Result xor (Result shr 27), UInt64($94D049BB133111EB));
  Result := Result xor (Result shr 31);
end;

function TVdxSampler.RotLeft(const AX: UInt64; const AK: Integer): UInt64;
begin
  Result := (AX shl AK) or (AX shr (64 - AK));
end;

function TVdxSampler.NextUInt64(): UInt64;
var
  LT: UInt64;
begin
  // xoshiro256** algorithm
  Result := WrapMul(RotLeft(WrapMul(FState1, 5), 7), 9);

  LT := FState1 shl 17;
  FState2 := FState2 xor FState0;
  FState3 := FState3 xor FState1;
  FState1 := FState1 xor FState2;
  FState0 := FState0 xor FState3;
  FState2 := FState2 xor LT;
  FState3 := RotLeft(FState3, 45);
end;

procedure TVdxSampler.SeedPRNG(const ASeed: Int64);
var
  LSeedState: UInt64;
begin
  if ASeed = 0 then
  begin
    // Non-deterministic: seed from high-resolution timer
    QueryPerformanceCounter(Int64(LSeedState));
  end
  else
    LSeedState := UInt64(ASeed);

  // Expand single seed into 4 state words via SplitMix64
  FState0 := SplitMix64(LSeedState);
  FState1 := SplitMix64(LSeedState);
  FState2 := SplitMix64(LSeedState);
  FState3 := SplitMix64(LSeedState);
end;

function TVdxSampler.NextFloat(): Single;
begin
  // Convert upper 24 bits of UInt64 to [0.0, 1.0)
  // 24 bits gives full Single mantissa precision
  Result := (NextUInt64() shr 40) / 16777216.0; // 2^24
end;

procedure TVdxSampler.EnsureWorkspace(const AVocabSize: Integer);
begin
  if AVocabSize = FCachedVocabSize then
    Exit;

  FreeAndNil(FProbsVBuf);
  FreeAndNil(FIndicesVBuf);

  FProbsVBuf := TVdxVirtualBuffer<Single>.Create();
  FProbsVBuf.SetErrors(FErrors);
  FProbsVBuf.Allocate(AVocabSize);

  FIndicesVBuf := TVdxVirtualBuffer<Integer>.Create();
  FIndicesVBuf.SetErrors(FErrors);
  FIndicesVBuf.Allocate(AVocabSize);

  FCachedVocabSize := AVocabSize;
end;

function TVdxSampler.Process(const ALogits: System.PSingle;
  const AVocabSize: Integer): Integer;
var
  LI, LJ: Integer;
  LBestId: Integer;
  LBestScore: Single;
  LPtr: System.PSingle;
  LProbs: System.PSingle;
  LMaxLogit: Single;
  LSum: Double;
  LVal: Single;
  LTarget: Double;
  LCumulative: Double;
  LK: Integer;
  LMinIdx: Integer;
  LMinVal: Single;
  LInvTemp: Single;
  LTempIdx: Integer;
  LThreshold: Single;
begin
  // --- Step 0: Repetition penalty (modifies input logits in-place) ---
  // Safe: logits buffer is overwritten by GPU download each token
  if (FConfig.RepeatPenalty <> 1.0) and (FHistoryCount > 0) then
  begin
    for LI := 0 to FHistoryCount - 1 do
    begin
      LBestId := FHistory[LI];  // reuse LBestId as token index
      if (LBestId >= 0) and (LBestId < AVocabSize) then
      begin
        LVal := System.PSingle(PByte(ALogits) + UInt64(LBestId) * SizeOf(Single))^;
        if LVal > 0.0 then
          System.PSingle(PByte(ALogits) + UInt64(LBestId) * SizeOf(Single))^ :=
            LVal / FConfig.RepeatPenalty
        else if LVal < 0.0 then
          System.PSingle(PByte(ALogits) + UInt64(LBestId) * SizeOf(Single))^ :=
            LVal * FConfig.RepeatPenalty;
      end;
    end;
  end;

  // --- Greedy fast path (Temperature=0): pure argmax, no allocation ---
  if FConfig.Temperature = 0.0 then
  begin
    LPtr := ALogits;
    LBestId := 0;
    LBestScore := LPtr^;
    for LI := 1 to AVocabSize - 1 do
    begin
      Inc(LPtr);
      if LPtr^ > LBestScore then
      begin
        LBestId := LI;
        LBestScore := LPtr^;
      end;
    end;
    Result := LBestId;
    Exit;
  end;

  // --- Sampling path: Temperature > 0 ---
  LInvTemp := 1.0 / FConfig.Temperature;

  if (FConfig.TopK > 0) and (FConfig.TopK < AVocabSize) then
  begin
    // === Top-K fast path: O(N) read-only + O(K) softmax+select ===
    // Temperature doesn't change ranking, so find top-K on raw logits
    LK := FConfig.TopK;
    if Length(FTopKBuf) <> LK then
    begin
      SetLength(FTopKBuf, LK);
      SetLength(FTopKIdx, LK);
    end;

    // --- Pass 1: Single O(N) read-only scan over raw logits ---
    // Track global max and maintain K-element (index, value) min-buffer
    // No temperature multiply — ranking is invariant under positive scaling
    LPtr := ALogits;
    LMaxLogit := LPtr^;
    LMinIdx := 0;
    LMinVal := -1e30;

    for LI := 0 to AVocabSize - 1 do
    begin
      LVal := LPtr^;
      Inc(LPtr);

      if LVal > LMaxLogit then
        LMaxLogit := LVal;

      if LI < LK then
      begin
        FTopKBuf[LI] := LVal;
        FTopKIdx[LI] := LI;
        if LI = LK - 1 then
        begin
          LMinIdx := 0;
          LMinVal := FTopKBuf[0];
          for LJ := 1 to LK - 1 do
            if FTopKBuf[LJ] < LMinVal then
            begin
              LMinIdx := LJ;
              LMinVal := FTopKBuf[LJ];
            end;
        end;
      end
      else if LVal > LMinVal then
      begin
        FTopKBuf[LMinIdx] := LVal;
        FTopKIdx[LMinIdx] := LI;
        LMinIdx := 0;
        LMinVal := FTopKBuf[0];
        for LJ := 1 to LK - 1 do
          if FTopKBuf[LJ] < LMinVal then
          begin
            LMinIdx := LJ;
            LMinVal := FTopKBuf[LJ];
          end;
      end;
    end;

    // --- Pass 2: Temperature + softmax over K entries ---
    LSum := 0.0;
    for LI := 0 to LK - 1 do
    begin
      FTopKBuf[LI] := Exp((FTopKBuf[LI] - LMaxLogit) * LInvTemp);
      LSum := LSum + FTopKBuf[LI];
    end;

    // --- Min-P filtering (on K entries) ---
    if FConfig.MinP > 0.0 then
    begin
      // Find max unnormalized prob — threshold is relative to it
      LBestScore := FTopKBuf[0];
      for LI := 1 to LK - 1 do
        if FTopKBuf[LI] > LBestScore then
          LBestScore := FTopKBuf[LI];

      // Zero entries below threshold, recalculate sum
      LThreshold := LBestScore * FConfig.MinP;
      LSum := 0.0;
      for LI := 0 to LK - 1 do
        if FTopKBuf[LI] < LThreshold then
          FTopKBuf[LI] := 0.0
        else
          LSum := LSum + FTopKBuf[LI];
    end;

    // --- Top-P (nucleus) filtering (on K entries) ---
    if FConfig.TopP < 1.0 then
    begin
      // Insertion sort descending — K is small (~40), this is trivial
      for LI := 1 to LK - 1 do
      begin
        LVal := FTopKBuf[LI];
        LTempIdx := FTopKIdx[LI];
        LJ := LI - 1;
        while (LJ >= 0) and (FTopKBuf[LJ] < LVal) do
        begin
          FTopKBuf[LJ + 1] := FTopKBuf[LJ];
          FTopKIdx[LJ + 1] := FTopKIdx[LJ];
          Dec(LJ);
        end;
        FTopKBuf[LJ + 1] := LVal;
        FTopKIdx[LJ + 1] := LTempIdx;
      end;

      // Accumulate from top until >= TopP fraction, zero the rest
      LTarget := FConfig.TopP * LSum;
      LCumulative := 0.0;
      LSum := 0.0;
      for LI := 0 to LK - 1 do
      begin
        if (LCumulative < LTarget) and (FTopKBuf[LI] > 0.0) then
        begin
          LCumulative := LCumulative + FTopKBuf[LI];
          LSum := LSum + FTopKBuf[LI];
        end
        else
          FTopKBuf[LI] := 0.0;
      end;
    end;

    // --- Random weighted selection over survivors ---
    LTarget := NextFloat() * LSum;
    LCumulative := 0.0;
    Result := FTopKIdx[LK - 1];  // fallback
    for LI := 0 to LK - 1 do
    begin
      LCumulative := LCumulative + FTopKBuf[LI];
      if LCumulative > LTarget then
      begin
        Result := FTopKIdx[LI];
        Exit;
      end;
    end;
  end
  else
  begin
    // === Full softmax path (no Top-K) ===
    EnsureWorkspace(AVocabSize);
    LProbs := System.PSingle(FProbsVBuf.Memory);

    // Pass 1: Copy logits * invTemp, find max
    LMaxLogit := System.PSingle(ALogits)^ * LInvTemp;
    for LI := 0 to AVocabSize - 1 do
    begin
      LVal := System.PSingle(PByte(ALogits) + UInt64(LI) * SizeOf(Single))^
        * LInvTemp;
      System.PSingle(PByte(LProbs) + UInt64(LI) * SizeOf(Single))^ := LVal;
      if LVal > LMaxLogit then
        LMaxLogit := LVal;
    end;

    // Pass 2: Softmax (Exp + accumulate)
    LSum := 0.0;
    for LI := 0 to AVocabSize - 1 do
    begin
      LVal := Exp(
        System.PSingle(PByte(LProbs) + UInt64(LI) * SizeOf(Single))^ - LMaxLogit);
      System.PSingle(PByte(LProbs) + UInt64(LI) * SizeOf(Single))^ := LVal;
      LSum := LSum + LVal;
    end;

    // Min-P filtering (full path)
    if FConfig.MinP > 0.0 then
    begin
      // Find max unnormalized prob
      LBestScore := System.PSingle(LProbs)^;
      for LI := 1 to AVocabSize - 1 do
      begin
        LVal := System.PSingle(PByte(LProbs) + UInt64(LI) * SizeOf(Single))^;
        if LVal > LBestScore then
          LBestScore := LVal;
      end;

      // Zero entries below threshold, recalculate sum
      LThreshold := LBestScore * FConfig.MinP;
      LSum := 0.0;
      for LI := 0 to AVocabSize - 1 do
      begin
        LVal := System.PSingle(PByte(LProbs) + UInt64(LI) * SizeOf(Single))^;
        if LVal < LThreshold then
          System.PSingle(PByte(LProbs) + UInt64(LI) * SizeOf(Single))^ := 0.0
        else
          LSum := LSum + LVal;
      end;
    end;

    // Pass 3: Random selection (scaled threshold, no renormalize)
    LTarget := NextFloat() * LSum;
    LCumulative := 0.0;
    Result := AVocabSize - 1;  // fallback
    for LI := 0 to AVocabSize - 1 do
    begin
      LCumulative := LCumulative +
        System.PSingle(PByte(LProbs) + UInt64(LI) * SizeOf(Single))^;
      if LCumulative > LTarget then
      begin
        Result := LI;
        Exit;
      end;
    end;
  end;
end;

procedure TVdxSampler.AddToHistory(const ATokenId: Integer);
begin
  if FConfig.RepeatWindow <= 0 then
    Exit;

  FHistory[FHistoryPos] := ATokenId;
  FHistoryPos := (FHistoryPos + 1) mod FConfig.RepeatWindow;
  if FHistoryCount < FConfig.RepeatWindow then
    Inc(FHistoryCount);
end;

procedure TVdxSampler.ResetHistory();
begin
  FHistoryPos := 0;
  FHistoryCount := 0;

  // Re-seed PRNG for deterministic per-generation reproducibility
  if FConfig.Seed > 0 then
    SeedPRNG(FConfig.Seed);
end;

end.
