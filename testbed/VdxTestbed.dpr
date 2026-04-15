{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

program VdxTestbed;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  System.SysUtils,
  UVdxTestbed in 'UVdxTestbed.pas',
  VindexLLM.Attention in '..\src\VindexLLM.Attention.pas',
  VindexLLM.ChatTemplate in '..\src\VindexLLM.ChatTemplate.pas',
  VindexLLM.Compute in '..\src\VindexLLM.Compute.pas',
  VindexLLM.Config in '..\src\VindexLLM.Config.pas',
  VindexLLM.FFNWeights in '..\src\VindexLLM.FFNWeights.pas',
  VindexLLM.GGUFReader in '..\src\VindexLLM.GGUFReader.pas',
  VindexLLM.Inference in '..\src\VindexLLM.Inference.pas',
  VindexLLM.LayerNorm in '..\src\VindexLLM.LayerNorm.pas',
  VindexLLM.Resources in '..\src\VindexLLM.Resources.pas',
  VindexLLM.Shaders in '..\src\VindexLLM.Shaders.pas',
  VindexLLM.Tokenizer in '..\src\VindexLLM.Tokenizer.pas',
  VindexLLM.TOML in '..\src\VindexLLM.TOML.pas',
  VindexLLM.TurboQuant in '..\src\VindexLLM.TurboQuant.pas',
  VindexLLM.Utils in '..\src\VindexLLM.Utils.pas',
  VindexLLM.VirtualBuffer in '..\src\VindexLLM.VirtualBuffer.pas',
  VindexLLM.Vulkan in '..\src\VindexLLM.Vulkan.pas';

begin
  RunVdxTestbed();
end.
