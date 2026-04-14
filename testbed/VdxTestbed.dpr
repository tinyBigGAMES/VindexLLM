program VdxTestbed;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  System.SysUtils,
  UVdxTestbed in 'UVdxTestbed.pas',
  VindexLLM.Attention in '..\src\VindexLLM.Attention.pas',
  VindexLLM.ChatTemplate in '..\src\VindexLLM.ChatTemplate.pas',
  VindexLLM.Config in '..\src\VindexLLM.Config.pas',
  VindexLLM.GGUFReader in '..\src\VindexLLM.GGUFReader.pas',
  VindexLLM.Inference in '..\src\VindexLLM.Inference.pas',
  VindexLLM.KNNWalk in '..\src\VindexLLM.KNNWalk.pas',
  VindexLLM.LayerNorm in '..\src\VindexLLM.LayerNorm.pas',
  VindexLLM.Resources in '..\src\VindexLLM.Resources.pas',
  VindexLLM.Tokenizer in '..\src\VindexLLM.Tokenizer.pas',
  VindexLLM.TOML in '..\src\VindexLLM.TOML.pas',
  VindexLLM.Utils in '..\src\VindexLLM.Utils.pas',
  VindexLLM.Vindex in '..\src\VindexLLM.Vindex.pas',
  VindexLLM.VirtualBuffer in '..\src\VindexLLM.VirtualBuffer.pas',
  VindexLLM.VulkanCompute in '..\src\VindexLLM.VulkanCompute.pas';

begin
  RunVdxTestbed();
end.
