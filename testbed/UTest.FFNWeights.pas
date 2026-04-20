{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UTest.FFNWeights;

interface

uses
  VindexLLM.TestCase,
  VindexLLM.Compute,
  VindexLLM.FFN;

type

  { TFFNWeightsTest }
  TFFNWeightsTest = class(TVdxTestCase)
  private
    procedure SecCreateDestroy();
    procedure SecBuildFromGGUF();
    procedure SecUploadLayer();
  protected
    procedure Run(); override;
  public
    constructor Create(); override;
  end;

implementation

uses
  System.SysUtils,
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.GGUFReader,
  UTest.Common;


{ TFFNWeightsTest }

constructor TFFNWeightsTest.Create();
begin
  inherited;
  Title := 'Test_FFNWeights';
end;

procedure TFFNWeightsTest.Run();
begin
  SecCreateDestroy();
  SecBuildFromGGUF();
  SecUploadLayer();
end;

procedure TFFNWeightsTest.SecCreateDestroy();
var
  LFFN: TVdxFFN;
begin
  Section('Create + Destroy');

  LFFN := TVdxFFN.Create();
  try
    Check(LFFN <> nil, 'Create returned non-nil');
    Check(LFFN.GetLayerCount() = 0, 'LayerCount is 0 before Build');
    FlushErrors(LFFN.GetErrors());
  finally
    LFFN.Free();
  end;
end;

procedure TFFNWeightsTest.SecBuildFromGGUF();
var
  LReader: TVdxGGUFReader;
  LFFN:    TVdxFFN;
begin
  Section('BuildFromGGUF (Gemma 3 4B)');

  LReader := TVdxGGUFReader.Create();
  try
    Check(LReader.Open(CModelPath),
      Format('Open model (%s)', [CModelPath]));
    FlushErrors(LReader.GetErrors());

    LFFN := TVdxFFN.Create();
    try
      LFFN.SetErrors(LReader.GetErrors());
      Check(LFFN.BuildFromGGUF(LReader),
        'BuildFromGGUF returns True');
      FlushErrors(LFFN.GetErrors());

      Check(LFFN.GetLayerCount() = 34,
        Format('LayerCount = 34 (got %d)', [LFFN.GetLayerCount()]));
      Check(LFFN.GetHiddenDim() = 2560,
        Format('HiddenDim = 2560 (got %d)', [LFFN.GetHiddenDim()]));
      Check(LFFN.GetFFNWidth() = 10240,
        Format('FFNWidth = 10240 (got %d)', [LFFN.GetFFNWidth()]));

      Check(LFFN.GetLayer(0).GatePtr <> nil,
        'Layer 0 GatePtr non-nil (mmap pointer)');
      Check(LFFN.GetLayer(0).UpPtr <> nil,
        'Layer 0 UpPtr non-nil');
      Check(LFFN.GetLayer(0).DownPtr <> nil,
        'Layer 0 DownPtr non-nil');
    finally
      LFFN.Free();
    end;
  finally
    LReader.Free();
  end;
end;

procedure TFFNWeightsTest.SecUploadLayer();
var
  LCompute: TVdxCompute;
  LReader:  TVdxGGUFReader;
  LFFN:     TVdxFFN;
  LLayer:   TVdxFFNLayerView;
begin
  Section('UploadLayer + FreeLayerGpu (layer 0)');

  LCompute := TVdxCompute.Create();
  try
    LCompute.Init(-1);
    FlushErrors(LCompute.GetErrors());

    LReader := TVdxGGUFReader.Create();
    try
      LReader.SetErrors(LCompute.GetErrors());
      Check(LReader.Open(CModelPath), 'Open model');
      FlushErrors(LReader.GetErrors());

      LFFN := TVdxFFN.Create();
      try
        LFFN.SetErrors(LCompute.GetErrors());
        Check(LFFN.BuildFromGGUF(LReader), 'BuildFromGGUF');
        FlushErrors(LFFN.GetErrors());

        LFFN.UploadLayer(0, LCompute);
        Check(not LFFN.GetErrors().HasFatal(),
          'UploadLayer(0) no fatal');
        FlushErrors(LFFN.GetErrors());

        LLayer := LFFN.GetLayer(0);
        Check(LLayer.GateGpuBuffer.Buffer <> VK_NULL_HANDLE,
          'Gate GPU buffer allocated');
        Check(LLayer.DownGpuBuffer.Buffer <> VK_NULL_HANDLE,
          'Down GPU buffer allocated');

        LFFN.FreeLayerGpu(0, LCompute);
        FlushErrors(LFFN.GetErrors());
      finally
        LFFN.Free();
      end;
    finally
      LReader.Free();
    end;
  finally
    LCompute.Free();
  end;
end;

end.
