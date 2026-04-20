{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.LayerNorm;

interface

uses
  VindexLLM.TestCase,
  VindexLLM.Compute,
  VindexLLM.GGUFReader,
  VindexLLM.LayerNorm;

type

  { TLayerNormTest }
  TLayerNormTest = class(TVdxTestCase)
  private
    procedure SecCreateDestroy();
    procedure SecInitSuccess();
    procedure SecApplyBasic();
    procedure SecApplyCopy();
    procedure SecUploadNormWeights();
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
  VindexLLM.Vulkan,
  UTest.Common;

{ TLayerNormTest }

constructor TLayerNormTest.Create();
begin
  inherited;
  Title := 'Test_LayerNorm';
end;

procedure TLayerNormTest.Run();
begin
  SecCreateDestroy();
  SecInitSuccess();
  SecApplyBasic();
  SecApplyCopy();
  SecUploadNormWeights();
end;

procedure TLayerNormTest.SecCreateDestroy();
var
  LLN: TVdxLayerNorm;
begin
  Section('Create + Destroy');
  LLN := TVdxLayerNorm.Create();
  try
    Check(LLN <> nil, 'Create returned non-nil');
    FlushErrors(LLN.GetErrors());
  finally
    LLN.Free();
  end;
end;

procedure TLayerNormTest.SecInitSuccess();
var
  LCompute: TVdxCompute;
  LLN:      TVdxLayerNorm;
begin
  Section('Init success');

  LCompute := TVdxCompute.Create();
  try
    LCompute.Init(-1);
    FlushErrors(LCompute.GetErrors());

    LLN := TVdxLayerNorm.Create();
    try
      LLN.SetErrors(LCompute.GetErrors());
      LLN.Init(LCompute, 1e-6);
      Check(not LLN.GetErrors().HasFatal(), 'Init no fatal');
      FlushErrors(LLN.GetErrors());
    finally
      LLN.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

procedure TLayerNormTest.SecApplyBasic();
var
  LCompute:    TVdxCompute;
  LLN:         TVdxLayerNorm;
  LResBuf:     TVdxGpuBuffer;
  LWeightBuf:  TVdxGpuBuffer;
  LResData:    array of Single;
  LWData:      array of Single;
  LOutData:    array of Single;
  LBytes:      UInt64;
  LI:          Integer;
  LAnyNaN:     Boolean;
  LAllZero:    Boolean;
begin
  Section('Apply in-place RMSNorm');

  LBytes := UInt64(CHiddenDim) * SizeOf(Single);

  LCompute := TVdxCompute.Create();
  try
    LCompute.Init(-1);
    FlushErrors(LCompute.GetErrors());

    LLN := TVdxLayerNorm.Create();
    try
      LLN.SetErrors(LCompute.GetErrors());
      LLN.Init(LCompute, 1e-6);
      FlushErrors(LLN.GetErrors());

      // Residual: small varied pattern
      SetLength(LResData, CHiddenDim);
      for LI := 0 to Integer(CHiddenDim) - 1 do
        LResData[LI] := 0.1 * Single((LI mod 7) - 3);

      // Weights: all ones (identity norm scale)
      SetLength(LWData, CHiddenDim);
      for LI := 0 to Integer(CHiddenDim) - 1 do
        LWData[LI] := 1.0;

      LResBuf := LCompute.CreateGpuBuffer(LBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      LWeightBuf := LCompute.CreateGpuBuffer(LBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      try
        LCompute.UploadToBuffer(LResBuf, @LResData[0], LBytes);
        LCompute.UploadToBuffer(LWeightBuf, @LWData[0], LBytes);

        LCompute.BeginBatch();
        try
          LLN.Apply(LResBuf, LWeightBuf, CHiddenDim);
        finally
          LCompute.EndBatch();
        end;
        FlushErrors(LLN.GetErrors());

        SetLength(LOutData, CHiddenDim);
        LCompute.DownloadFromBuffer(LResBuf, @LOutData[0], LBytes);

        LAnyNaN := False;
        LAllZero := True;
        for LI := 0 to Integer(CHiddenDim) - 1 do
        begin
          if IsNan(LOutData[LI]) then LAnyNaN := True;
          if LOutData[LI] <> 0.0 then LAllZero := False;
        end;

        Check(not LAnyNaN, 'No NaN in output');
        Check(not LAllZero, 'Output not all zero');
      finally
        LCompute.DestroyGpuBuffer(LResBuf);
        LCompute.DestroyGpuBuffer(LWeightBuf);
      end;
    finally
      LLN.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

procedure TLayerNormTest.SecApplyCopy();
var
  LCompute:   TVdxCompute;
  LLN:        TVdxLayerNorm;
  LSrcBuf:    TVdxGpuBuffer;
  LWeightBuf: TVdxGpuBuffer;
  LDstBuf:    TVdxGpuBuffer;
  LSrcData:   array of Single;
  LWData:     array of Single;
  LOutData:   array of Single;
  LBytes:     UInt64;
  LI:         Integer;
  LAnyNaN:    Boolean;
  LAllZero:   Boolean;
begin
  Section('ApplyCopy fused copy+norm');

  LBytes := UInt64(CHiddenDim) * SizeOf(Single);

  LCompute := TVdxCompute.Create();
  try
    LCompute.Init(-1);
    FlushErrors(LCompute.GetErrors());

    LLN := TVdxLayerNorm.Create();
    try
      LLN.SetErrors(LCompute.GetErrors());
      LLN.Init(LCompute, 1e-6);
      FlushErrors(LLN.GetErrors());

      SetLength(LSrcData, CHiddenDim);
      for LI := 0 to Integer(CHiddenDim) - 1 do
        LSrcData[LI] := 0.05 * Single((LI mod 5) - 2);
      SetLength(LWData, CHiddenDim);
      for LI := 0 to Integer(CHiddenDim) - 1 do
        LWData[LI] := 1.0;

      LSrcBuf := LCompute.CreateGpuBuffer(LBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      LWeightBuf := LCompute.CreateGpuBuffer(LBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      LDstBuf := LCompute.CreateGpuBuffer(LBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      try
        LCompute.UploadToBuffer(LSrcBuf, @LSrcData[0], LBytes);
        LCompute.UploadToBuffer(LWeightBuf, @LWData[0], LBytes);

        LCompute.BeginBatch();
        try
          LLN.ApplyCopy(LSrcBuf, LWeightBuf, LDstBuf, CHiddenDim);
        finally
          LCompute.EndBatch();
        end;
        FlushErrors(LLN.GetErrors());

        SetLength(LOutData, CHiddenDim);
        LCompute.DownloadFromBuffer(LDstBuf, @LOutData[0], LBytes);

        LAnyNaN := False;
        LAllZero := True;
        for LI := 0 to Integer(CHiddenDim) - 1 do
        begin
          if IsNan(LOutData[LI]) then LAnyNaN := True;
          if LOutData[LI] <> 0.0 then LAllZero := False;
        end;

        Check(not LAnyNaN, 'No NaN in dest');
        Check(not LAllZero, 'Dest not all zero');
      finally
        LCompute.DestroyGpuBuffer(LSrcBuf);
        LCompute.DestroyGpuBuffer(LWeightBuf);
        LCompute.DestroyGpuBuffer(LDstBuf);
      end;
    finally
      LLN.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

procedure TLayerNormTest.SecUploadNormWeights();
var
  LCompute: TVdxCompute;
  LReader:  TVdxGGUFReader;
  LLN:      TVdxLayerNorm;
  LWeights: TVdxNormLayerWeights;
begin
  Section('UploadNormWeights (Gemma 3 4B layer 0)');

  LCompute := TVdxCompute.Create();
  try
    LCompute.Init(-1);
    FlushErrors(LCompute.GetErrors());

    LReader := TVdxGGUFReader.Create();
    try
      LReader.SetErrors(LCompute.GetErrors());
      Check(LReader.Open(CModelPath), 'Open model');
      FlushErrors(LReader.GetErrors());

      LLN := TVdxLayerNorm.Create();
      try
        LLN.SetErrors(LCompute.GetErrors());
        LLN.Init(LCompute, 1e-6);
        FlushErrors(LLN.GetErrors());

        LLN.UploadNormWeights(LReader, 0, LWeights);
        Check(not LLN.GetErrors().HasFatal(),
          'UploadNormWeights no fatal');
        FlushErrors(LLN.GetErrors());

        Check(LWeights.AttnNormGpu.Buffer <> VK_NULL_HANDLE,
          'AttnNorm GPU allocated');
        Check(LWeights.PostAttnNormGpu.Buffer <> VK_NULL_HANDLE,
          'PostAttnNorm GPU allocated');
        Check(LWeights.FFNNormGpu.Buffer <> VK_NULL_HANDLE,
          'FFNNorm GPU allocated');
        Check(LWeights.PostFFNNormGpu.Buffer <> VK_NULL_HANDLE,
          'PostFFNNorm GPU allocated');
        Check(LWeights.QNormGpu.Buffer <> VK_NULL_HANDLE,
          'QNorm GPU allocated');
        Check(LWeights.KNormGpu.Buffer <> VK_NULL_HANDLE,
          'KNorm GPU allocated');

        LLN.FreeNormWeights(LWeights);
      finally
        LLN.Free();
      end;
    finally
      LReader.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

end.
