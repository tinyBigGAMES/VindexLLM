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
  UTest.Common in 'UTest.Common.pas',
  UTest.Compute in 'UTest.Compute.pas',
  UTest.Demo in 'UTest.Demo.pas',
  UTest.FFNWeights in 'UTest.FFNWeights.pas',
  UTest.GGUFReader in 'UTest.GGUFReader.pas',
  UTest.Inference in 'UTest.Inference.pas',
  UTest.LayerNorm in 'UTest.LayerNorm.pas',
  UTest.Model.Gemma3 in 'UTest.Model.Gemma3.pas',
  UTest.Sampler in 'UTest.Sampler.pas',
  UTest.Tokenizer in 'UTest.Tokenizer.pas',
  UTest.TurboQuant in 'UTest.TurboQuant.pas',
  UTest.VirtualBuffer in 'UTest.VirtualBuffer.pas',
  UTest.VirtualFile in 'UTest.VirtualFile.pas',
  UVdxTestbed in 'UVdxTestbed.pas',
  VindexLLM.Attention in '..\src\VindexLLM.Attention.pas',
  VindexLLM.Chat in '..\src\VindexLLM.Chat.pas',
  VindexLLM.Compute in '..\src\VindexLLM.Compute.pas',
  VindexLLM.Config in '..\src\VindexLLM.Config.pas',
  VindexLLM.ConsoleChat in '..\src\VindexLLM.ConsoleChat.pas',
  VindexLLM.Embeddings in '..\src\VindexLLM.Embeddings.pas',
  VindexLLM.FFN in '..\src\VindexLLM.FFN.pas',
  VindexLLM.GGUFReader in '..\src\VindexLLM.GGUFReader.pas',
  VindexLLM.Inference in '..\src\VindexLLM.Inference.pas',
  VindexLLM.LayerNorm in '..\src\VindexLLM.LayerNorm.pas',
  VindexLLM.Memory in '..\src\VindexLLM.Memory.pas',
  VindexLLM.Model.Gemma3 in '..\src\VindexLLM.Model.Gemma3.pas',
  VindexLLM.Model in '..\src\VindexLLM.Model.pas',
  VindexLLM.Model.Registry in '..\src\VindexLLM.Model.Registry.pas',
  VindexLLM.Resources in '..\src\VindexLLM.Resources.pas',
  VindexLLM.Sampler in '..\src\VindexLLM.Sampler.pas',
  VindexLLM.Session in '..\src\VindexLLM.Session.pas',
  VindexLLM.Shaders in '..\src\VindexLLM.Shaders.pas',
  VindexLLM.TestCase in '..\src\VindexLLM.TestCase.pas',
  VindexLLM.Tokenizer in '..\src\VindexLLM.Tokenizer.pas',
  VindexLLM.TokenWriter in '..\src\VindexLLM.TokenWriter.pas',
  VindexLLM.TOML in '..\src\VindexLLM.TOML.pas',
  VindexLLM.TurboQuant in '..\src\VindexLLM.TurboQuant.pas',
  VindexLLM.Utils in '..\src\VindexLLM.Utils.pas',
  VindexLLM.VirtualBuffer in '..\src\VindexLLM.VirtualBuffer.pas',
  VindexLLM.VirtualFile in '..\src\VindexLLM.VirtualFile.pas',
  VindexLLM.Vulkan in '..\src\VindexLLM.Vulkan.pas',
  VindexLLM.Common in '..\src\VindexLLM.Common.pas';

begin
  RunVdxTestbed();
end.
