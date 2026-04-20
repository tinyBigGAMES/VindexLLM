{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.TurboQuant;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Math,
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.Compute,
  VindexLLM.Shaders;
const
  // TQ3 block: 32 F32 values -> 16 bytes (4 x uint32, aligned)
  CTQ3BlockSize    = 32;
  CTQ3PackedBytes  = 16;  // 4 x uint32
  CTQ3PackedWords  = 4;

  // Lloyd-Max 3-bit centroids (8 levels, optimal for N(0,1))
  CTQ3Centroids: array[0..7] of Single = (
    -2.1573, -1.3336, -0.7434, -0.2428,
    +0.2428, +0.7434, +1.3336, +2.1573
  );

  // Decision boundaries (midpoints between adjacent centroids)
  CTQ3Boundaries: array[0..6] of Single = (
    -1.7455, -1.0385, -0.4931, 0.0,
    +0.4931, +1.0385, +1.7455
  );

  // WHT normalization factor: 1/sqrt(32)
  CWHT_Norm: Single = 0.17677669529663688;

  // Centroid max (outermost value, used as scale divisor)
  CCentroidMax: Single = 2.1573;

  // Fixed sign flip pattern (must match shaders exactly)
  CTQ3Signs: array[0..31] of Integer = (
    +1,-1,+1,+1,-1,+1,-1,-1, +1,+1,-1,+1,+1,-1,+1,-1,
    -1,+1,+1,-1,+1,-1,+1,+1, -1,-1,+1,-1,+1,+1,-1,+1
  );
type

  { TVdxTQ3Push }
  TVdxTQ3Push = record
    NumBlocks: UInt32;
  end;

  { TVdxTQ3Block }
  TVdxTQ3Block = record
    QS0: UInt32;    // qs bytes [0..3]: elements 0-15, low 2 index bits
    QS1: UInt32;    // qs bytes [4..7]: elements 16-31, low 2 index bits
    QR:  UInt32;    // qr bytes [0..3]: elements 0-31, high 1 index bit
    Gamma: UInt32;  // FP16 scale in low 16 bits
  end;

  { TVdxTurboQuant }
  TVdxTurboQuant = class(TVdxBaseObject)
  private
    FCompute: TVdxCompute;

    // Quantize pipeline
    FQuantShader: VkShaderModule;
    FQuantBundle: TVdxComputePipelineBundle;
    FQuantDescLayout: VkDescriptorSetLayout;

    // Dequantize pipeline
    FDequantShader: VkShaderModule;
    FDequantBundle: TVdxComputePipelineBundle;
    FDequantDescLayout: VkDescriptorSetLayout;
  public
    constructor Create(); override;
    destructor Destroy(); override;
    // Initialize GPU pipelines
    procedure Init(const ACompute: TVdxCompute);

    // GPU dispatch: quantize F32 buffer -> TQ3 buffer
    // AInputBuf must contain ANumBlocks * 32 floats
    // AOutputBuf must hold ANumBlocks * 4 uint32s (16 bytes per block)
    procedure Quantize(
      const AInputBuf: TVdxGpuBuffer;
      const AOutputBuf: TVdxGpuBuffer;
      const ANumBlocks: Integer;
      const ADescPool: VkDescriptorPool;
      const ADescSet: VkDescriptorSet);

    // GPU dispatch: dequantize TQ3 buffer -> F32 buffer
    procedure Dequantize(
      const AInputBuf: TVdxGpuBuffer;
      const AOutputBuf: TVdxGpuBuffer;
      const ANumBlocks: Integer;
      const ADescPool: VkDescriptorPool;
      const ADescSet: VkDescriptorSet);

    // CPU reference implementations (for validation)
    class procedure QuantizeBlockCPU(
      const AInput: PSingle;
      var AOutput: TVdxTQ3Block); static;

    class procedure DequantizeBlockCPU(
      const AInput: TVdxTQ3Block;
      const AOutput: PSingle); static;

    // Descriptor set layouts (needed for external pool/set allocation)
    property QuantDescLayout: VkDescriptorSetLayout read FQuantDescLayout;
    property DequantDescLayout: VkDescriptorSetLayout read FDequantDescLayout;

    // Utility: compute MSE between two float arrays
    class function ComputeMSE(
      const AA: PSingle;
      const AB: PSingle;
      const ACount: Integer): Double; static;
  end;

implementation

{  FP16 <-> FP32 conversion helpers }
function SingleToHalf(const AValue: Single): UInt32;
var
  LBits: UInt32;
  LSign: UInt32;
  LExp: Integer;
  LMant: UInt32;
begin
  LBits := PUInt32(@AValue)^;
  LSign := (LBits shr 16) and $8000;
  LExp := Integer((LBits shr 23) and $FF) - 127 + 15;
  LMant := (LBits and $7FFFFF) shr 13;

  if LExp <= 0 then
    Result := LSign
  else if LExp >= 31 then
    Result := LSign or $7C00
  else
    Result := LSign or (UInt32(LExp) shl 10) or LMant;
end;

function HalfToSingle(const AHalf: UInt32): Single;
var
  LSign: UInt32;
  LExp: UInt32;
  LMant: UInt32;
  LBits: UInt32;
begin
  LSign := (AHalf and $8000) shl 16;
  LExp := (AHalf shr 10) and $1F;
  LMant := AHalf and $3FF;

  if LExp = 0 then
    LBits := LSign
  else if LExp = 31 then
    LBits := LSign or $7F800000 or (LMant shl 13)
  else
    LBits := LSign or ((LExp + 127 - 15) shl 23) or (LMant shl 13);

  Result := PSingle(@LBits)^;
end;


{ TVdxTurboQuant }
constructor TVdxTurboQuant.Create();
begin
  inherited Create();
  FCompute := nil;
  FQuantShader := VK_NULL_HANDLE;
  FDequantShader := VK_NULL_HANDLE;
end;

destructor TVdxTurboQuant.Destroy();
begin
  if FCompute <> nil then
  begin
    if FQuantBundle.Pipeline <> VK_NULL_HANDLE then
      FCompute.DestroyComputePipelineBundle(FQuantBundle);
    if FQuantDescLayout <> VK_NULL_HANDLE then
      FCompute.DestroyDescriptorSetLayoutHandle(FQuantDescLayout);
    if FQuantShader <> VK_NULL_HANDLE then
      FCompute.DestroyShaderModuleHandle(FQuantShader);

    if FDequantBundle.Pipeline <> VK_NULL_HANDLE then
      FCompute.DestroyComputePipelineBundle(FDequantBundle);
    if FDequantDescLayout <> VK_NULL_HANDLE then
      FCompute.DestroyDescriptorSetLayoutHandle(FDequantDescLayout);
    if FDequantShader <> VK_NULL_HANDLE then
      FCompute.DestroyShaderModuleHandle(FDequantShader);
  end;
  inherited Destroy();
end;

procedure TVdxTurboQuant.Init(const ACompute: TVdxCompute);
var
  LSpvData: TBytes;
begin
  FCompute := ACompute;

  // --- Quantize pipeline (2 bindings: input F32, output TQ3) ---
  LSpvData := VdxLoadShader('TQ3_QUANTIZE');
  FQuantShader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  FQuantDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  FQuantBundle := FCompute.CreateComputePipelineWithPush(
    FQuantShader, 'main', FQuantDescLayout, SizeOf(TVdxTQ3Push));

  // --- Dequantize pipeline (2 bindings: input TQ3, output F32) ---
  LSpvData := VdxLoadShader('TQ3_DEQUANTIZE');
  FDequantShader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  FDequantDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  FDequantBundle := FCompute.CreateComputePipelineWithPush(
    FDequantShader, 'main', FDequantDescLayout, SizeOf(TVdxTQ3Push));
end;

procedure TVdxTurboQuant.Quantize(
  const AInputBuf: TVdxGpuBuffer;
  const AOutputBuf: TVdxGpuBuffer;
  const ANumBlocks: Integer;
  const ADescPool: VkDescriptorPool;
  const ADescSet: VkDescriptorSet);
var
  LPush: TVdxTQ3Push;
begin
  LPush.NumBlocks := UInt32(ANumBlocks);

  FCompute.UpdateDescriptorSetBuffers(ADescSet, [AInputBuf, AOutputBuf]);
  FCompute.DispatchComputeWithPush(
    FQuantBundle.Pipeline, FQuantBundle.PipelineLayout,
    ADescSet, @LPush, SizeOf(LPush), UInt32(ANumBlocks));
end;

procedure TVdxTurboQuant.Dequantize(
  const AInputBuf: TVdxGpuBuffer;
  const AOutputBuf: TVdxGpuBuffer;
  const ANumBlocks: Integer;
  const ADescPool: VkDescriptorPool;
  const ADescSet: VkDescriptorSet);
var
  LPush: TVdxTQ3Push;
begin
  LPush.NumBlocks := UInt32(ANumBlocks);

  FCompute.UpdateDescriptorSetBuffers(ADescSet, [AInputBuf, AOutputBuf]);
  FCompute.DispatchComputeWithPush(
    FDequantBundle.Pipeline, FDequantBundle.PipelineLayout,
    ADescSet, @LPush, SizeOf(LPush), UInt32(ANumBlocks));
end;

class procedure TVdxTurboQuant.QuantizeBlockCPU(
  const AInput: PSingle;
  var AOutput: TVdxTQ3Block);
var
  LTemp: array[0..31] of Single;
  LI: Integer;
  LStep: Integer;
  LJ: Integer;
  LA: Single;
  LB: Single;
  LAmax: Single;
  LGamma: Single;
  LInvGamma: Single;
  LScaled: Single;
  LIdx: Integer;
  LQSWord: Integer;
  LShift: Integer;
begin
  // 1. Copy input and apply sign flips
  for LI := 0 to 31 do
    LTemp[LI] := PSingle(PByte(AInput) + LI * SizeOf(Single))^ * CTQ3Signs[LI];
  // 2. WHT butterfly (5 stages: step = 1, 2, 4, 8, 16)
  LStep := 1;
  while LStep <= 16 do
  begin
    LI := 0;
    while LI < 32 do
    begin
      for LJ := LI to LI + LStep - 1 do
      begin
        LA := LTemp[LJ];
        LB := LTemp[LJ + LStep];
        LTemp[LJ]         := LA + LB;
        LTemp[LJ + LStep] := LA - LB;
      end;
      Inc(LI, LStep * 2);
    end;
    LStep := LStep * 2;
  end;

  // 3. Normalize by 1/sqrt(32)
  for LI := 0 to 31 do
    LTemp[LI] := LTemp[LI] * CWHT_Norm;

  // 4. Find amax and compute scale
  LAmax := 0.0;
  for LI := 0 to 31 do
    if Abs(LTemp[LI]) > LAmax then
      LAmax := Abs(LTemp[LI]);

  LGamma := LAmax / CCentroidMax;
  if LGamma > 0.0 then
    LInvGamma := 1.0 / LGamma
  else
    LInvGamma := 0.0;
  // 5. Quantize each value and pack into output block
  AOutput.QS0 := 0;
  AOutput.QS1 := 0;
  AOutput.QR  := 0;

  for LI := 0 to 31 do
  begin
    LScaled := LTemp[LI] * LInvGamma;

    // Find nearest centroid index (0-7)
    if      LScaled < CTQ3Boundaries[0] then LIdx := 0
    else if LScaled < CTQ3Boundaries[1] then LIdx := 1
    else if LScaled < CTQ3Boundaries[2] then LIdx := 2
    else if LScaled < CTQ3Boundaries[3] then LIdx := 3
    else if LScaled < CTQ3Boundaries[4] then LIdx := 4
    else if LScaled < CTQ3Boundaries[5] then LIdx := 5
    else if LScaled < CTQ3Boundaries[6] then LIdx := 6
    else                                     LIdx := 7;

    // Pack low 2 bits into qs word
    LQSWord := LI div 16;  // 0 or 1
    LShift := ((LI div 4) mod 4) * 8 + (LI mod 4) * 2;
    if LQSWord = 0 then
      AOutput.QS0 := AOutput.QS0 or (UInt32(LIdx and 3) shl LShift)
    else
      AOutput.QS1 := AOutput.QS1 or (UInt32(LIdx and 3) shl LShift);

    // Pack high 1 bit into qr word
    AOutput.QR := AOutput.QR or (UInt32((LIdx shr 2) and 1) shl LI);
  end;
  // 6. Store gamma as FP16 packed in low 16 bits
  //    Manual FP32 -> FP16 conversion for CPU reference
  AOutput.Gamma := SingleToHalf(LGamma);
end;

class procedure TVdxTurboQuant.DequantizeBlockCPU(
  const AInput: TVdxTQ3Block;
  const AOutput: PSingle);
var
  LTemp: array[0..31] of Single;
  LI: Integer;
  LStep: Integer;
  LJ: Integer;
  LA: Single;
  LB: Single;
  LLow2: UInt32;
  LHigh1: UInt32;
  LIdx: UInt32;
  LGamma: Single;
  LQSWord: UInt32;
  LShift: Integer;
begin
  // 1. Unpack gamma from FP16
  LGamma := HalfToSingle(AInput.Gamma);
  // 2. Unpack indices and lookup centroids
  for LI := 0 to 31 do
  begin
    // Low 2 bits from qs
    if LI < 16 then
      LQSWord := AInput.QS0
    else
      LQSWord := AInput.QS1;

    LShift := ((LI div 4) mod 4) * 8 + (LI mod 4) * 2;
    LLow2 := (LQSWord shr LShift) and 3;

    // High 1 bit from qr
    LHigh1 := (AInput.QR shr LI) and 1;
    LIdx := LLow2 or (LHigh1 shl 2);

    LTemp[LI] := CTQ3Centroids[LIdx] * LGamma;
  end;
  // 3. Inverse WHT butterfly (same code as forward — self-inverse)
  LStep := 1;
  while LStep <= 16 do
  begin
    LI := 0;
    while LI < 32 do
    begin
      for LJ := LI to LI + LStep - 1 do
      begin
        LA := LTemp[LJ];
        LB := LTemp[LJ + LStep];
        LTemp[LJ]         := LA + LB;
        LTemp[LJ + LStep] := LA - LB;
      end;
      Inc(LI, LStep * 2);
    end;
    LStep := LStep * 2;
  end;

  // 4. Normalize by 1/sqrt(32) and undo sign flips, write output
  for LI := 0 to 31 do
    PSingle(PByte(AOutput) + LI * SizeOf(Single))^ :=
      LTemp[LI] * CWHT_Norm * CTQ3Signs[LI];
end;

class function TVdxTurboQuant.ComputeMSE(
  const AA: PSingle;
  const AB: PSingle;
  const ACount: Integer): Double;
var
  LI: Integer;
  LDiff: Double;
  LSum: Double;
begin
  LSum := 0.0;
  for LI := 0 to ACount - 1 do
  begin
    LDiff := Double(PSingle(PByte(AA) + LI * SizeOf(Single))^) -
             Double(PSingle(PByte(AB) + LI * SizeOf(Single))^);
    LSum := LSum + LDiff * LDiff;
  end;
  if ACount > 0 then
    Result := LSum / ACount
  else
    Result := 0.0;
end;

end.