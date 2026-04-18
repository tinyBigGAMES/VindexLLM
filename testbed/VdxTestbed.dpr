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
  VindexLLM.Utils in '..\src\VindexLLM.Utils.pas';

begin
  RunVdxTestbed();
end.
