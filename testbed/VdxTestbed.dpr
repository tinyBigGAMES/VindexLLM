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
  UTest.Attention in 'UTest.Attention.pas',
  UTest.Compute in 'UTest.Compute.pas',
  UTest.FFNWeights in 'UTest.FFNWeights.pas',
  UTest.GGUFReader in 'UTest.GGUFReader.pas',
  UTest.LayerNorm in 'UTest.LayerNorm.pas',
  UTest.Tokenizer in 'UTest.Tokenizer.pas',
  UTest.TurboQuant in 'UTest.TurboQuant.pas',
  UTest.VirtualBuffer in 'UTest.VirtualBuffer.pas',
  UTest.VirtualFile in 'UTest.VirtualFile.pas',
  UVdxTestbed in 'UVdxTestbed.pas',
  UTest.Sampler in 'UTest.Sampler.pas',
  VindexLLM.Attention in '..\src\VindexLLM.Attention.pas',
  VindexLLM.Compute in '..\src\VindexLLM.Compute.pas',
  VindexLLM.Config in '..\src\VindexLLM.Config.pas',
  VindexLLM.FFNWeights in '..\src\VindexLLM.FFNWeights.pas',
  VindexLLM.GGUFReader in '..\src\VindexLLM.GGUFReader.pas',
  VindexLLM.LayerNorm in '..\src\VindexLLM.LayerNorm.pas',
  VindexLLM.Resources in '..\src\VindexLLM.Resources.pas',
  VindexLLM.Sampler in '..\src\VindexLLM.Sampler.pas',
  VindexLLM.Shaders in '..\src\VindexLLM.Shaders.pas',
  VindexLLM.TestCase in '..\src\VindexLLM.TestCase.pas',
  VindexLLM.Tokenizer in '..\src\VindexLLM.Tokenizer.pas',
  VindexLLM.TokenWriter in '..\src\VindexLLM.TokenWriter.pas',
  VindexLLM.TOML in '..\src\VindexLLM.TOML.pas',
  VindexLLM.TurboQuant in '..\src\VindexLLM.TurboQuant.pas',
  VindexLLM.Utils in '..\src\VindexLLM.Utils.pas',
  VindexLLM.VirtualBuffer in '..\src\VindexLLM.VirtualBuffer.pas',
  VindexLLM.VirtualFile in '..\src\VindexLLM.VirtualFile.pas',
  VindexLLM.Vulkan in '..\src\VindexLLM.Vulkan.pas';

begin
  RunVdxTestbed();
end.
