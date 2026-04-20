{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.Attention;

interface

uses
  VindexLLM.TestCase,
  VindexLLM.Compute,
  VindexLLM.Attention;

type

  { TAttentionTest }
  TAttentionTest = class(TVdxTestCase)
  private
    procedure SecCreateDestroy();
    procedure SecInitSuccess();
    procedure SecResolveWeights();
    procedure SecForwardSmoke();
    procedure SecForwardBatchSmoke();
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
  VindexLLM.GGUFReader,
  VindexLLM.LayerNorm,
  UTest.Common;

const
  CNumQHeads:  UInt32 = 8;
  CNumKVHeads: UInt32 = 4;
  CHeadDim:    UInt32 = 256;
  CNumLayers:  UInt32 = 34;
  CMaxSeqLen:  UInt32 = 2048;
  CFFNWidth:   UInt32 = 10240;

{ TAttentionTest }

constructor TAttentionTest.Create();
begin
  inherited;
  Title := 'Test_Attention';
end;

procedure TAttentionTest.Run();
begin
  SecCreateDestroy();
  SecInitSuccess();
  SecResolveWeights();
  SecForwardSmoke();
  SecForwardBatchSmoke();
end;

procedure TAttentionTest.SecCreateDestroy();
var
  LAttn: TVdxAttention;
begin
  Section('Create + Destroy (no compute)');

  LAttn := TVdxAttention.Create();
  try
    Check(LAttn <> nil, 'Create returned a non-nil instance');
    FlushErrors(LAttn.GetErrors());
  finally
    LAttn.Free();
  end;
end;

procedure TAttentionTest.SecInitSuccess();
var
  LCompute: TVdxCompute;
  LAttn:    TVdxAttention;
begin
  Section('Init success (real compute + Gemma 3 4B dims)');

  LCompute := TVdxCompute.Create();
  try
    LCompute.Init(-1);
    Check(not LCompute.GetErrors().HasFatal(), 'Compute.Init no fatal');
    FlushErrors(LCompute.GetErrors());

    LAttn := TVdxAttention.Create();
    try
      LAttn.SetErrors(LCompute.GetErrors());
      LAttn.Init(LCompute,
        CHiddenDim, CNumQHeads, CNumKVHeads, CHeadDim,
        CNumLayers, CMaxSeqLen, CFFNWidth);
      Check(not LAttn.GetErrors().HasFatal(),
        'Attention.Init no fatal');
      FlushErrors(LAttn.GetErrors());
    finally
      LAttn.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

procedure TAttentionTest.SecResolveWeights();
var
  LCompute: TVdxCompute;
  LReader:  TVdxGGUFReader;
  LAttn:    TVdxAttention;
  LWeights: TVdxAttnLayerWeights;
begin
  Section('UploadAttnWeights (Gemma 3 4B layer 0)');

  LCompute := TVdxCompute.Create();
  try
    LCompute.Init(-1);
    Check(not LCompute.GetErrors().HasFatal(), 'Compute.Init');
    FlushErrors(LCompute.GetErrors());

    LReader := TVdxGGUFReader.Create();
    try
      LReader.SetErrors(LCompute.GetErrors());
      Check(LReader.Open(CModelPath),
        Format('Open model file (%s)', [CModelPath]));
      FlushErrors(LReader.GetErrors());

      LAttn := TVdxAttention.Create();
      try
        LAttn.SetErrors(LCompute.GetErrors());
        LAttn.Init(LCompute,
          CHiddenDim, CNumQHeads, CNumKVHeads, CHeadDim,
          CNumLayers, CMaxSeqLen, CFFNWidth);
        FlushErrors(LAttn.GetErrors());

        LAttn.UploadAttnWeights(LReader, 0, LWeights);
        Check(not LAttn.GetErrors().HasFatal(),
          'UploadAttnWeights no fatal');
        FlushErrors(LAttn.GetErrors());

        Check(LWeights.QWeightGpu.Buffer <> VK_NULL_HANDLE,
          'Q GPU buffer allocated');
        Check(LWeights.KWeightGpu.Buffer <> VK_NULL_HANDLE,
          'K GPU buffer allocated');
        Check(LWeights.VWeightGpu.Buffer <> VK_NULL_HANDLE,
          'V GPU buffer allocated');
        Check(LWeights.OWeightGpu.Buffer <> VK_NULL_HANDLE,
          'O GPU buffer allocated');
        Check(LWeights.WeightType = gtQ4_0,
          'WeightType is gtQ4_0');
      finally
        LAttn.Free();
      end;
    finally
      LReader.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

procedure TAttentionTest.SecForwardSmoke();
var
  LCompute:     TVdxCompute;
  LReader:      TVdxGGUFReader;
  LLN:          TVdxLayerNorm;
  LAttn:        TVdxAttention;
  LNormW:       TVdxNormLayerWeights;
  LWeights:     TVdxAttnLayerWeights;
  LInputBuf:    TVdxGpuBuffer;
  LOutBuf:      TVdxGpuBuffer;
  LHiddenBytes: UInt64;
  LInput:       array of Single;
  LOutput:      array of Single;
  LI:           Integer;
  LAnyNaN:      Boolean;
  LAnyInf:      Boolean;
  LAllZero:     Boolean;
  LNonZero:     Integer;
  LMaxAbs:      Single;
begin
  Section('Forward smoke (Gemma 3 4B, layer 0, pos 0)');

  LHiddenBytes := UInt64(CHiddenDim) * SizeOf(Single);

  LCompute := TVdxCompute.Create();
  try
    LCompute.Init(-1);
    FlushErrors(LCompute.GetErrors());

    LReader := TVdxGGUFReader.Create();
    try
      LReader.SetErrors(LCompute.GetErrors());
      Check(LReader.Open(CModelPath), 'Open model file');
      FlushErrors(LReader.GetErrors());

      LLN   := TVdxLayerNorm.Create();
      LAttn := TVdxAttention.Create();
      try
        LLN.SetErrors(LCompute.GetErrors());
        LAttn.SetErrors(LCompute.GetErrors());

        LLN.Init(LCompute, 1e-6);
        LAttn.Init(LCompute,
          CHiddenDim, CNumQHeads, CNumKVHeads, CHeadDim,
          CNumLayers, CMaxSeqLen, CFFNWidth);
        FlushErrors(LAttn.GetErrors());

        LLN.UploadNormWeights(LReader, 0, LNormW);
        LAttn.UploadAttnWeights(LReader, 0, LWeights);
        FlushErrors(LAttn.GetErrors());

        SetLength(LInput, CHiddenDim);
        for LI := 0 to Integer(CHiddenDim) - 1 do
          LInput[LI] := 0.01 * Single((LI mod 7) - 3);

        LInputBuf := LCompute.CreateGpuBuffer(LHiddenBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        LOutBuf := LCompute.CreateGpuBuffer(LHiddenBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        try
          LCompute.UploadToBuffer(LInputBuf, @LInput[0], LHiddenBytes);
          FlushErrors(LCompute.GetErrors());

          LCompute.BeginBatch();
          try
            LAttn.Forward(LInputBuf, LWeights,
              LNormW.QNormGpu, LNormW.KNormGpu,
              0, 0, 10000.0, LOutBuf);
          finally
            LCompute.EndBatch();
          end;
          Check(not LAttn.GetErrors().HasFatal(),
            'No fatal errors after Forward');
          FlushErrors(LAttn.GetErrors());

          SetLength(LOutput, CHiddenDim);
          LCompute.DownloadFromBuffer(LOutBuf, @LOutput[0], LHiddenBytes);
          FlushErrors(LCompute.GetErrors());

          LAnyNaN  := False;
          LAnyInf  := False;
          LAllZero := True;
          LNonZero := 0;
          LMaxAbs  := 0.0;
          for LI := 0 to Integer(CHiddenDim) - 1 do
          begin
            if IsNan(LOutput[LI]) then LAnyNaN := True;
            if IsInfinite(LOutput[LI]) then LAnyInf := True;
            if LOutput[LI] <> 0.0 then
            begin
              LAllZero := False;
              Inc(LNonZero);
            end;
            if Abs(LOutput[LI]) > LMaxAbs then
              LMaxAbs := Abs(LOutput[LI]);
          end;

          TVdxUtils.PrintLn('    non-zero=%d / %d   max|out|=%g',
            [LNonZero, CHiddenDim, Double(LMaxAbs)]);

          Check(not LAnyNaN, 'Output contains no NaN');
          Check(not LAnyInf, 'Output contains no Inf');
          Check(not LAllZero, 'Output is not entirely zero');
        finally
          LCompute.DestroyGpuBuffer(LInputBuf);
          LCompute.DestroyGpuBuffer(LOutBuf);
        end;

        LLN.FreeNormWeights(LNormW);
      finally
        LAttn.Free();
        LLN.Free();
      end;
    finally
      LReader.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

procedure TAttentionTest.SecForwardBatchSmoke();
const
  CNumTokens: UInt32 = 4;
var
  LCompute:    TVdxCompute;
  LReader:     TVdxGGUFReader;
  LLN:         TVdxLayerNorm;
  LAttn:       TVdxAttention;
  LNormW:      TVdxNormLayerWeights;
  LWeights:    TVdxAttnLayerWeights;
  LInputMat:   TVdxGpuBuffer;
  LQMat:       TVdxGpuBuffer;
  LKMat:       TVdxGpuBuffer;
  LVMat:       TVdxGpuBuffer;
  LOutMat:     TVdxGpuBuffer;
  LInputBytes: UInt64;
  LQBytes:     UInt64;
  LKVBytes:    UInt64;
  LInput:      array of Single;
  LOutput:     array of Single;
  LI:          Integer;
  LAnyNaN:     Boolean;
  LAnyInf:     Boolean;
  LNonZero:    Integer;
  LMaxAbs:     Single;
begin
  Section('ForwardBatch smoke (Gemma 3 4B, layer 0, 4 tokens)');

  LInputBytes := UInt64(CNumTokens) * CHiddenDim * SizeOf(Single);
  LQBytes     := UInt64(CNumTokens) * (CNumQHeads  * CHeadDim) * SizeOf(Single);

  LKVBytes := UInt64(CNumTokens) * (CNumKVHeads * CHeadDim) * SizeOf(Single);

  LCompute := TVdxCompute.Create();
  try
    LCompute.Init(-1);
    FlushErrors(LCompute.GetErrors());

    LReader := TVdxGGUFReader.Create();
    try
      LReader.SetErrors(LCompute.GetErrors());
      Check(LReader.Open(CModelPath), 'Open model file');
      FlushErrors(LReader.GetErrors());

      LLN   := TVdxLayerNorm.Create();
      LAttn := TVdxAttention.Create();
      try
        LLN.SetErrors(LCompute.GetErrors());
        LAttn.SetErrors(LCompute.GetErrors());

        LLN.Init(LCompute, 1e-6);
        LAttn.Init(LCompute,
          CHiddenDim, CNumQHeads, CNumKVHeads, CHeadDim,
          CNumLayers, CMaxSeqLen, CFFNWidth);
        FlushErrors(LAttn.GetErrors());

        LLN.UploadNormWeights(LReader, 0, LNormW);
        LAttn.UploadAttnWeights(LReader, 0, LWeights);
        FlushErrors(LAttn.GetErrors());

        SetLength(LInput, CNumTokens * CHiddenDim);
        for LI := 0 to High(LInput) do
          LInput[LI] := 0.01 * Single((LI mod 11) - 5);

        LInputMat := LCompute.CreateGpuBuffer(LInputBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        LQMat := LCompute.CreateGpuBuffer(LQBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        LKMat := LCompute.CreateGpuBuffer(LKVBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        LVMat := LCompute.CreateGpuBuffer(LKVBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        LOutMat := LCompute.CreateGpuBuffer(LInputBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        try
          LCompute.UploadToBuffer(LInputMat, @LInput[0], LInputBytes);
          FlushErrors(LCompute.GetErrors());

          LCompute.BeginBatch();
          try
            LAttn.ForwardBatch(LInputMat, LWeights,
              LNormW.QNormGpu, LNormW.KNormGpu,
              0, CNumTokens, 0, 10000.0,
              LQMat, LKMat, LVMat, LOutMat, False);
          finally
            LCompute.EndBatch();
          end;
          Check(not LAttn.GetErrors().HasFatal(),
            'No fatal errors after ForwardBatch');
          FlushErrors(LAttn.GetErrors());

          SetLength(LOutput, CNumTokens * CHiddenDim);
          LCompute.DownloadFromBuffer(LOutMat, @LOutput[0], LInputBytes);
          FlushErrors(LCompute.GetErrors());

          LAnyNaN  := False;
          LAnyInf  := False;
          LNonZero := 0;
          LMaxAbs  := 0.0;
          for LI := 0 to High(LOutput) do
          begin
            if IsNan(LOutput[LI]) then LAnyNaN := True;
            if IsInfinite(LOutput[LI]) then LAnyInf := True;

            if LOutput[LI] <> 0.0 then Inc(LNonZero);
            if Abs(LOutput[LI]) > LMaxAbs then
              LMaxAbs := Abs(LOutput[LI]);
          end;

          TVdxUtils.PrintLn('    non-zero=%d / %d   max|out|=%g',
            [LNonZero, Length(LOutput), Double(LMaxAbs)]);

          Check(not LAnyNaN, 'Output contains no NaN');
          Check(not LAnyInf, 'Output contains no Inf');
          Check(LNonZero > 0, 'Output is not entirely zero');
        finally
          LCompute.DestroyGpuBuffer(LInputMat);
          LCompute.DestroyGpuBuffer(LQMat);
          LCompute.DestroyGpuBuffer(LKMat);
          LCompute.DestroyGpuBuffer(LVMat);
          LCompute.DestroyGpuBuffer(LOutMat);
        end;

        LLN.FreeNormWeights(LNormW);
      finally
        LAttn.Free();
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
