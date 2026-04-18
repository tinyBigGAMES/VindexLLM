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
  VindexLLM.Utils;


procedure Test_XXXX();
begin
end;

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
    TVdxUtils.Pause('Press any key to start...');

    LIndex := 1;

    case LIndex of
      1: Test_XXXX();
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
