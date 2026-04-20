{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
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
  System.Generics.Collections,
  System.SysUtils,
  VindexLLM.Utils,
  VindexLLM.TestCase,
  UTest.VirtualBuffer,
  UTest.VirtualFile,
  UTest.Compute,
  UTest.GGUFReader,
  UTest.TurboQuant,
  UTest.LayerNorm,
  UTest.FFNWeights,
  UTest.Attention,
  UTest.Model.Gemma3,
  UTest.Sampler,
  UTest.Tokenizer,
  UTest.Inference,
  UTest.Demo;

// ---------------------------------------------------------------------------
// RunVdxTestbed — entry point for the testbed application.
// Selects which test to run via LIndex, wraps in top-level exception handler,
// and pauses for keypress when running from the Delphi IDE so you can read
// the console output before the window closes.
// ---------------------------------------------------------------------------
procedure RunVdxTestbed();
var
  LIndex: Integer;
begin
  try
    TVdxUtils.Pause('Press any key to start inference...');

    LIndex := 13;

    case LIndex of
      01: VdxRunTestCase(TVirtualBufferTest);
      02: VdxRunTestCase(TVirtualFileTest);
      03: VdxRunTestCase(TComputeTest);
      04: VdxRunTestCase(TGGUFReaderTest);
      05: VdxRunTestCase(TTurboQuantTest);
      06: VdxRunTestCase(TLayerNormTest);
      07: VdxRunTestCase(TFFNWeightsTest);
      08: VdxRunTestCase(TAttentionTest);
      09: VdxRunTestCase(TModelGemma3Test);
      10: VdxRunTestCase(TSamplerTest);
      11: VdxRunTestCase(TTokenizerTest);
      12: VdxRunTestCase(TInferenceTest);
      13: Demo_Inference();
      14: Demo_Chat();
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
