{===============================================================================
  VindexLLM� - Liberating LLM inference

  Copyright � 2026-present tinyBigGAMES� LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UVdxTestbed;

interface

procedure RunVdxTestbed();

implementation

uses
  WinAPI.Windows,
  System.SysUtils,
  System.IOUtils,
  System.Math,
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.Compute,
  VindexLLM.TurboQuant,
  VindexLLM.Inference;

procedure StatusCallback(const AText: string; const AUserData: Pointer);
begin
  TVdxUtils.PrintLn(AText);
end;

function PrintToken(const AToken: string; const AUserData: Pointer): Boolean;
begin
  Write(AToken);
  Result := (GetAsyncKeyState(VK_ESCAPE) and $8000) = 0;
end;

procedure PrintStats(const AStats: PVdxInferenceStats);
begin
  TVdxUtils.PrintLn();
  TVdxUtils.PrintLn('Prefill:    %d tokens in %.0fms (%.1f tok/s)', [
    AStats.PrefillTokens, AStats.PrefillTimeMs, AStats.PrefillTokPerSec]);
  TVdxUtils.PrintLn('Generation: %d tokens in %.0fms (%.1f tok/s)', [
    AStats.GeneratedTokens, AStats.GenerationTimeMs, AStats.GenerationTokPerSec]);
  TVdxUtils.PrintLn('TTFT: %.0fms | Total: %.0fms | Stop: %s', [
    AStats.TimeToFirstTokenMs, AStats.TotalTimeMs,
    CVdxStopReasons[AStats.StopReason]]);
end;

procedure Test01();
const
  CPrompt =
  '''
    Explain the differences between these three sorting algorithms: bubble sort,
    merge sort, and quicksort. For each one, describe how it works step by step,
    give the best-case and worst-case time complexity using big-O notation,
    explain when you would choose it over the others, and provide a real-world
    analogy that helps illustrate the concept. Also discuss whether each
    algorithm is stable or unstable, and what that means in practice.
  ''';
var
  LInference: TVdxInference;
begin
  TVdxUtils.Pause();
  LInference := TVdxInference.Create();
  try
    LInference.SetStatusCallback(StatusCallback, nil);
    //LInference.LoadModel('C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.f16.gguf');
    LInference.LoadModel('C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.Q8_0.gguf');
    LInference.SetTokenCallback(PrintToken, nil);
    //LInference.Generate(CPrompt);
    //LInference.Generate('how to make kno3?');
    LInference.Generate('what is the capital of france?');
    PrintStats(LInference.GetStats());
    LInference.UnloadModel();
  finally
    LInference.Free();
  end;
end;

procedure Test02();
const
  CNumBlocks = 256;   // 256 blocks x 32 = 8192 floats
  CNumFloats = CNumBlocks * CTQ3BlockSize;
var
  LOriginal: array of Single;
  LRestored: array of Single;
  LBlock: TVdxTQ3Block;
  LI: Integer;
  LB: Integer;
  LMSE: Double;
  LMaxErr: Double;
  LErr: Double;
begin
  TVdxUtils.PrintLn(COLOR_CYAN + '=== TurboQuant TQ3 Round-Trip Test (CPU) ===');

  // Generate random data with realistic distribution
  SetLength(LOriginal, CNumFloats);
  SetLength(LRestored, CNumFloats);
  RandSeed := 42;
  for LI := 0 to CNumFloats - 1 do
    LOriginal[LI] := (Random - 0.5) * 4.0;  // uniform [-2, +2]

  // Round-trip: quantize then dequantize each block
  for LB := 0 to CNumBlocks - 1 do
  begin
    TVdxTurboQuant.QuantizeBlockCPU(@LOriginal[LB * CTQ3BlockSize], LBlock);
    TVdxTurboQuant.DequantizeBlockCPU(LBlock, @LRestored[LB * CTQ3BlockSize]);
  end;

  // Compute MSE and max error
  LMSE := TVdxTurboQuant.ComputeMSE(@LOriginal[0], @LRestored[0], CNumFloats);
  LMaxErr := 0.0;
  for LI := 0 to CNumFloats - 1 do
  begin
    LErr := Abs(LOriginal[LI] - LRestored[LI]);
    if LErr > LMaxErr then
      LMaxErr := LErr;
  end;

  TVdxUtils.PrintLn('  Blocks:    %d (%d floats)', [CNumBlocks, CNumFloats]);
  TVdxUtils.PrintLn('  MSE:       %.6f (target ~0.034)', [LMSE]);
  TVdxUtils.PrintLn('  Max error: %.6f', [LMaxErr]);

  if LMSE < 0.1 then
    TVdxUtils.PrintLn(COLOR_GREEN + '  PASS: MSE within expected range')
  else
    TVdxUtils.PrintLn(COLOR_RED + '  FAIL: MSE too high');
end;

procedure Test03();
const
  CNumBlocks = 256;
  CNumFloats = CNumBlocks * CTQ3BlockSize;
  CTQ3BufSize = CNumBlocks * CTQ3PackedBytes;  // 256 * 16 = 4096 bytes
  CF32BufSize = CNumFloats * SizeOf(Single);   // 8192 * 4 = 32768 bytes
var
  LCompute: TVdxVulkanCompute;
  LTQ: TVdxTurboQuant;
  LOriginal: array of Single;
  LGpuResult: array of Single;
  LCpuResult: array of Single;
  LBlock: TVdxTQ3Block;
  LInputBuf: TVdxGpuBuffer;
  LTQ3Buf: TVdxGpuBuffer;
  LOutputBuf: TVdxGpuBuffer;
  LDescPool: VkDescriptorPool;
  LQuantSet: VkDescriptorSet;
  LDequantSet: VkDescriptorSet;
  LI: Integer;
  LB: Integer;
  LMseGpu: Double;
  LMseCpu: Double;
  LMseGpuVsCpu: Double;
  LMaxErrGpuVsCpu: Double;
  LErr: Double;
begin
  TVdxUtils.PrintLn(COLOR_CYAN + '=== TurboQuant TQ3 Round-Trip Test (GPU) ===');

  // Generate random data (same seed as Test02)
  SetLength(LOriginal, CNumFloats);
  SetLength(LGpuResult, CNumFloats);
  SetLength(LCpuResult, CNumFloats);
  RandSeed := 42;
  for LI := 0 to CNumFloats - 1 do
    LOriginal[LI] := (Random - 0.5) * 4.0;

  // CPU reference round-trip
  for LB := 0 to CNumBlocks - 1 do
  begin
    TVdxTurboQuant.QuantizeBlockCPU(@LOriginal[LB * CTQ3BlockSize], LBlock);
    TVdxTurboQuant.DequantizeBlockCPU(LBlock, @LCpuResult[LB * CTQ3BlockSize]);
  end;
  LMseCpu := TVdxTurboQuant.ComputeMSE(@LOriginal[0], @LCpuResult[0], CNumFloats);

  // Init Vulkan
  TVdxUtils.PrintLn('  Initializing Vulkan...');
  LCompute := TVdxVulkanCompute.Create();
  try
    LCompute.Init();

    // Init TurboQuant GPU pipelines
    LTQ := TVdxTurboQuant.Create();
    try
      LTQ.Init(LCompute);

      // Create GPU buffers
      LInputBuf := LCompute.CreateGpuBuffer(
        CF32BufSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

      LTQ3Buf := LCompute.CreateGpuBuffer(
        CTQ3BufSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

      LOutputBuf := LCompute.CreateGpuBuffer(
        CF32BufSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

      // Upload input data
      LCompute.UploadToBuffer(LInputBuf, @LOriginal[0], CF32BufSize);

      // Create descriptor pool (2 sets, 4 storage descriptors total)
      LDescPool := LCompute.CreateDescriptorPoolForStorage(2, 4);

      // Allocate descriptor sets
      LQuantSet := LCompute.AllocateDescriptorSetForBuffers(
        LDescPool, LTQ.QuantDescLayout, [LInputBuf, LTQ3Buf]);
      LDequantSet := LCompute.AllocateDescriptorSetForBuffers(
        LDescPool, LTQ.DequantDescLayout, [LTQ3Buf, LOutputBuf]);

      // Dispatch quantize
      TVdxUtils.PrintLn('  Dispatching quantize (%d blocks)...', [CNumBlocks]);
      LTQ.Quantize(LInputBuf, LTQ3Buf, CNumBlocks, LDescPool, LQuantSet);

      // Dispatch dequantize
      TVdxUtils.PrintLn('  Dispatching dequantize (%d blocks)...', [CNumBlocks]);
      LTQ.Dequantize(LTQ3Buf, LOutputBuf, CNumBlocks, LDescPool, LDequantSet);

      // Download GPU result
      LCompute.DownloadFromBuffer(LOutputBuf, @LGpuResult[0], CF32BufSize);

      // Compute metrics
      LMseGpu := TVdxTurboQuant.ComputeMSE(@LOriginal[0], @LGpuResult[0], CNumFloats);
      LMseGpuVsCpu := TVdxTurboQuant.ComputeMSE(@LGpuResult[0], @LCpuResult[0], CNumFloats);

      LMaxErrGpuVsCpu := 0.0;
      for LI := 0 to CNumFloats - 1 do
      begin
        LErr := Abs(LGpuResult[LI] - LCpuResult[LI]);
        if LErr > LMaxErrGpuVsCpu then
          LMaxErrGpuVsCpu := LErr;
      end;

      // Print results
      TVdxUtils.PrintLn('');
      TVdxUtils.PrintLn('  Blocks:          %d (%d floats)', [CNumBlocks, CNumFloats]);
      TVdxUtils.PrintLn('  CPU round-trip MSE:  %.6f', [LMseCpu]);
      TVdxUtils.PrintLn('  GPU round-trip MSE:  %.6f', [LMseGpu]);
      TVdxUtils.PrintLn('  GPU vs CPU MSE:      %.9f', [LMseGpuVsCpu]);
      TVdxUtils.PrintLn('  GPU vs CPU max err:  %.9f', [LMaxErrGpuVsCpu]);

      if LMseGpuVsCpu < 1e-4 then
        TVdxUtils.PrintLn(COLOR_GREEN + '  PASS: GPU matches CPU reference')
      else
        TVdxUtils.PrintLn(COLOR_RED + '  FAIL: GPU does not match CPU reference');

      // Cleanup
      LCompute.DestroyDescriptorPoolHandle(LDescPool);
      LCompute.DestroyGpuBuffer(LOutputBuf);
      LCompute.DestroyGpuBuffer(LTQ3Buf);
      LCompute.DestroyGpuBuffer(LInputBuf);
    finally
      LTQ.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

procedure RunVdxTestbed();
var
  LIndex: Integer;
begin
  try
    LIndex := 1;

    case LIndex of
      1: Test01();
      2: Test02();
      3: Test03();
    end;
  except
    on E: Exception do
    begin
      TVdxUtils.PrintLn('');
      TVdxUtils.PrintLn(COLOR_RED + 'EXCEPTION: %s', [E.Message]);
    end;
  end;

  if TVdxUtils.RunFromIDE() then
    TVdxUtils.Pause();
end;

end.
