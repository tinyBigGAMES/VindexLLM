{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.TurboQuant;

interface

uses
  VindexLLM.TestCase,
  VindexLLM.Compute,
  VindexLLM.TurboQuant;

type

  { TTurboQuantTest }
  TTurboQuantTest = class(TVdxTestCase)
  private
    procedure SecCreateDestroy();
    procedure SecInitSuccess();
    procedure SecCPURoundTrip();
    procedure SecGPURoundTrip();
    procedure SecGPUvsCPU();
  protected
    procedure Run(); override;
  public
    constructor Create(); override;
  end;

implementation

uses
  System.SysUtils,
  System.Math,
  VindexLLM.Utils,
  VindexLLM.Vulkan;

const
  CTestNumBlocks = 64;
  CTestCount     = CTestNumBlocks * CTQ3BlockSize;
  CTestCompressedBytes = CTestNumBlocks * CTQ3PackedBytes;

function NextLCGFloat(var ASeed: UInt32): Single;
var
  LScratch: Int64;
begin
  LScratch := (Int64(ASeed) * 1664525 + 1013904223) and $FFFFFFFF;
  ASeed    := UInt32(LScratch);
  Result   := Single(ASeed) / 4294967296.0;
end;

{ TTurboQuantTest }

constructor TTurboQuantTest.Create();
begin
  inherited;
  Title := 'Test_TurboQuant';
end;

procedure TTurboQuantTest.Run();
begin
  SecCreateDestroy();
  SecInitSuccess();
  SecCPURoundTrip();
  SecGPURoundTrip();
  SecGPUvsCPU();
end;

procedure TTurboQuantTest.SecCreateDestroy();
var
  LTQ: TVdxTurboQuant;
begin
  Section('Create + Destroy');
  LTQ := TVdxTurboQuant.Create();
  try
    Check(LTQ <> nil, 'Create returned non-nil');
    FlushErrors(LTQ.GetErrors());
  finally
    LTQ.Free();
  end;
end;

procedure TTurboQuantTest.SecInitSuccess();
var
  LCompute: TVdxCompute;
  LTQ:      TVdxTurboQuant;
begin
  Section('Init on real GPU');

  LCompute := TVdxCompute.Create();
  try
    LCompute.Init(-1);
    FlushErrors(LCompute.GetErrors());

    LTQ := TVdxTurboQuant.Create();
    try
      LTQ.SetErrors(LCompute.GetErrors());
      LTQ.Init(LCompute);
      Check(not LTQ.GetErrors().HasFatal(), 'Init no fatal');
      FlushErrors(LTQ.GetErrors());
    finally
      LTQ.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

procedure TTurboQuantTest.SecCPURoundTrip();
var
  LInput:  array[0..CTQ3BlockSize - 1] of Single;
  LOutput: array[0..CTQ3BlockSize - 1] of Single;
  LBlock:  TVdxTQ3Block;
  LI:      Integer;
  LSeed:   UInt32;
  LU:      Single;
  LMSE:    Double;
begin
  Section('CPU Quantize -> Dequantize round trip');

  LSeed := $C0FFEE42;
  for LI := 0 to CTQ3BlockSize - 1 do
  begin
    LU := NextLCGFloat(LSeed);
    LInput[LI] := (LU * 4.0) - 2.0;
  end;

  TVdxTurboQuant.QuantizeBlockCPU(@LInput[0], LBlock);
  TVdxTurboQuant.DequantizeBlockCPU(LBlock, @LOutput[0]);

  LMSE := TVdxTurboQuant.ComputeMSE(@LInput[0], @LOutput[0], CTQ3BlockSize);
  Check(LMSE < 0.05, Format('CPU round-trip MSE %.6f < 0.05', [LMSE]));
end;

procedure TTurboQuantTest.SecGPURoundTrip();
var
  LCompute:    TVdxCompute;
  LTQ:         TVdxTurboQuant;
  LInput:      array of Single;
  LOutput:     array of Single;
  LInputBuf:   TVdxGpuBuffer;
  LTQ3Buf:     TVdxGpuBuffer;
  LOutputBuf:  TVdxGpuBuffer;
  LDescPool:   VkDescriptorPool;
  LQuantSet:   VkDescriptorSet;
  LDequantSet: VkDescriptorSet;
  LDummyBuf:   TVdxGpuBuffer;
  LInputBytes: UInt64;
  LTQ3Bytes:   UInt64;
  LI:          Integer;
  LSeed:       UInt32;
  LU:          Single;
  LMSE:        Double;
begin
  Section('GPU Quantize -> Dequantize round trip');

  SetLength(LInput, CTestCount);
  SetLength(LOutput, CTestCount);

  LSeed := $BADC0DE1;
  for LI := 0 to CTestCount - 1 do
  begin
    LU := NextLCGFloat(LSeed);
    LInput[LI] := (LU * 4.0) - 2.0;
  end;

  LInputBytes := UInt64(CTestCount) * SizeOf(Single);
  LTQ3Bytes   := UInt64(CTestCompressedBytes);

  LCompute := TVdxCompute.Create();
  try
    LCompute.Init(-1);
    FlushErrors(LCompute.GetErrors());

    LTQ := TVdxTurboQuant.Create();
    try
      LTQ.SetErrors(LCompute.GetErrors());
      LTQ.Init(LCompute);
      FlushErrors(LTQ.GetErrors());

      LInputBuf := LCompute.CreateGpuBuffer(LInputBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      LTQ3Buf := LCompute.CreateGpuBuffer(LTQ3Bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      LOutputBuf := LCompute.CreateGpuBuffer(LInputBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

      LDummyBuf := Default(TVdxGpuBuffer);
      LDescPool := LCompute.CreateDescriptorPoolForStorage(2, 4);
      LQuantSet := LCompute.AllocateDescriptorSetForBuffers(
        LDescPool, LTQ.QuantDescLayout, [LDummyBuf, LDummyBuf]);
      LDequantSet := LCompute.AllocateDescriptorSetForBuffers(
        LDescPool, LTQ.DequantDescLayout, [LDummyBuf, LDummyBuf]);
      try
        LCompute.UploadToBuffer(LInputBuf, @LInput[0], LInputBytes);
        FlushErrors(LCompute.GetErrors());

        LCompute.BeginBatch();
        try
          LTQ.Quantize(LInputBuf, LTQ3Buf, CTestNumBlocks,
            LDescPool, LQuantSet);
          LCompute.BatchBarrier();
          LTQ.Dequantize(LTQ3Buf, LOutputBuf, CTestNumBlocks,
            LDescPool, LDequantSet);
        finally
          LCompute.EndBatch();
        end;
        FlushErrors(LCompute.GetErrors());

        LCompute.DownloadFromBuffer(LOutputBuf, @LOutput[0], LInputBytes);
        FlushErrors(LCompute.GetErrors());

        LMSE := TVdxTurboQuant.ComputeMSE(
          @LInput[0], @LOutput[0], CTestCount);
        Check(LMSE < 0.05,
          Format('GPU round-trip MSE %.6f < 0.05', [LMSE]));
      finally
        LCompute.DestroyDescriptorPoolHandle(LDescPool);
        LCompute.DestroyGpuBuffer(LInputBuf);
        LCompute.DestroyGpuBuffer(LTQ3Buf);
        LCompute.DestroyGpuBuffer(LOutputBuf);
      end;
    finally
      LTQ.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

procedure TTurboQuantTest.SecGPUvsCPU();
var
  LCompute:    TVdxCompute;
  LTQ:         TVdxTurboQuant;
  LInput:      array of Single;
  LGpuBlocks:  array of TVdxTQ3Block;
  LCpuBlocks:  array of TVdxTQ3Block;
  LGpuRecon:   array of Single;
  LCpuRecon:   array of Single;
  LInputBuf:   TVdxGpuBuffer;
  LTQ3Buf:     TVdxGpuBuffer;
  LDescPool:   VkDescriptorPool;
  LQuantSet:   VkDescriptorSet;
  LDummyBuf:   TVdxGpuBuffer;
  LInputBytes: UInt64;
  LTQ3Bytes:   UInt64;
  LI:          Integer;
  LSeed:       UInt32;
  LU:          Single;
  LCrossMSE:   Double;
begin
  Section('GPU vs CPU reconstruction equivalence');

  SetLength(LInput, CTestCount);
  SetLength(LGpuBlocks, CTestNumBlocks);
  SetLength(LCpuBlocks, CTestNumBlocks);
  SetLength(LGpuRecon, CTestCount);
  SetLength(LCpuRecon, CTestCount);

  LSeed := $FEEDFACE;
  for LI := 0 to CTestCount - 1 do
  begin
    LU := NextLCGFloat(LSeed);
    LInput[LI] := (LU * 4.0) - 2.0;
  end;

  for LI := 0 to CTestNumBlocks - 1 do
    TVdxTurboQuant.QuantizeBlockCPU(
      @LInput[LI * CTQ3BlockSize], LCpuBlocks[LI]);

  LInputBytes := UInt64(CTestCount) * SizeOf(Single);
  LTQ3Bytes   := UInt64(CTestCompressedBytes);

  LCompute := TVdxCompute.Create();
  try
    LCompute.Init(-1);
    FlushErrors(LCompute.GetErrors());

    LTQ := TVdxTurboQuant.Create();
    try
      LTQ.SetErrors(LCompute.GetErrors());
      LTQ.Init(LCompute);
      FlushErrors(LTQ.GetErrors());

      LInputBuf := LCompute.CreateGpuBuffer(LInputBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      LTQ3Buf := LCompute.CreateGpuBuffer(LTQ3Bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

      LDummyBuf := Default(TVdxGpuBuffer);
      LDescPool := LCompute.CreateDescriptorPoolForStorage(1, 2);
      LQuantSet := LCompute.AllocateDescriptorSetForBuffers(
        LDescPool, LTQ.QuantDescLayout, [LDummyBuf, LDummyBuf]);
      try
        LCompute.UploadToBuffer(LInputBuf, @LInput[0], LInputBytes);
        FlushErrors(LCompute.GetErrors());

        LCompute.BeginBatch();
        try
          LTQ.Quantize(LInputBuf, LTQ3Buf, CTestNumBlocks,
            LDescPool, LQuantSet);
        finally
          LCompute.EndBatch();
        end;
        FlushErrors(LCompute.GetErrors());

        LCompute.DownloadFromBuffer(
          LTQ3Buf, @LGpuBlocks[0], LTQ3Bytes);
        FlushErrors(LCompute.GetErrors());

        for LI := 0 to CTestNumBlocks - 1 do
        begin
          TVdxTurboQuant.DequantizeBlockCPU(
            LGpuBlocks[LI], @LGpuRecon[LI * CTQ3BlockSize]);
          TVdxTurboQuant.DequantizeBlockCPU(
            LCpuBlocks[LI], @LCpuRecon[LI * CTQ3BlockSize]);
        end;

        LCrossMSE := TVdxTurboQuant.ComputeMSE(
          @LGpuRecon[0], @LCpuRecon[0], CTestCount);
        Check(LCrossMSE < 0.005,
          Format('GPU vs CPU reconstruction MSE %.6f < 0.005',
            [LCrossMSE]));
      finally
        LCompute.DestroyDescriptorPoolHandle(LDescPool);
        LCompute.DestroyGpuBuffer(LInputBuf);
        LCompute.DestroyGpuBuffer(LTQ3Buf);
      end;
    finally
      LTQ.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

end.
