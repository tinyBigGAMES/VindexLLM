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
  UTest.VirtualBuffer in 'UTest.VirtualBuffer.pas',
  UTest.VirtualFile in 'UTest.VirtualFile.pas',
  UVdxTestbed in 'UVdxTestbed.pas',
  UTest.Compute in 'UTest.Compute.pas',
  VindexLLM.Compute in '..\src\VindexLLM.Compute.pas',
  VindexLLM.Config in '..\src\VindexLLM.Config.pas',
  VindexLLM.GGUFReader in '..\src\VindexLLM.GGUFReader.pas',
  VindexLLM.Resources in '..\src\VindexLLM.Resources.pas',
  VindexLLM.Shaders in '..\src\VindexLLM.Shaders.pas',
  VindexLLM.TestCase in '..\src\VindexLLM.TestCase.pas',
  VindexLLM.TOML in '..\src\VindexLLM.TOML.pas',
  VindexLLM.Utils in '..\src\VindexLLM.Utils.pas',
  VindexLLM.VirtualBuffer in '..\src\VindexLLM.VirtualBuffer.pas',
  VindexLLM.VirtualFile in '..\src\VindexLLM.VirtualFile.pas',
  VindexLLM.Vulkan in '..\src\VindexLLM.Vulkan.pas',
  VindexLLM.Tokenizer in '..\src\VindexLLM.Tokenizer.pas';

begin
  RunVdxTestbed();
end.
