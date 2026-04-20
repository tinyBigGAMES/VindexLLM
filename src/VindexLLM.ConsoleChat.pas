{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.ConsoleChat;

{$I VindexLLM.Defines.inc}

interface

uses
  VindexLLM.Utils,
  VindexLLM.TokenWriter,
  VindexLLM.Chat;

type
  { TVdxConsoleChat }
  TVdxConsoleChat = class(TVdxChat)
  private
    FTokenWriter: TVdxConsoleTokenWriter;
  protected
    function  DoGetInput(): string; override;
    procedure DoOutput(const AText: string); override;
    procedure DoToken(const AToken: string); override;
    function  DoCancel(): Boolean; override;
    procedure DoStatus(const AText: string); override;
    procedure DoError(const AText: string); override;
    procedure DoInfo(const AText: string); override;
    procedure DoStartup(); override;
    procedure DoShutdown(); override;
    procedure DoGenerationComplete(); override;
  public
    constructor Create(); override;
    destructor Destroy(); override;
  end;

implementation

uses
  WinAPI.Windows,
  System.SysUtils;

{ TVdxConsoleChat }

constructor TVdxConsoleChat.Create();
begin
  inherited Create();
  FTokenWriter := TVdxConsoleTokenWriter.Create();
  FTokenWriter.MaxWidth := 118;
end;

destructor TVdxConsoleChat.Destroy();
begin
  FreeAndNil(FTokenWriter);
  inherited Destroy();
end;

function TVdxConsoleChat.DoGetInput(): string;
var
  LInput: string;
begin
  TVdxUtils.PrintLn();
  TVdxUtils.Print(COLOR_GREEN + 'You> ' + COLOR_RESET);
  ReadLn(LInput);
  Result := Trim(LInput);
end;

procedure TVdxConsoleChat.DoOutput(const AText: string);
begin
  TVdxUtils.PrintLn(AText);
end;

procedure TVdxConsoleChat.DoToken(const AToken: string);
begin
  FTokenWriter.Write(AToken);
end;

function TVdxConsoleChat.DoCancel(): Boolean;
begin
  Result := (GetAsyncKeyState(VK_ESCAPE) and $8000) <> 0;
end;

procedure TVdxConsoleChat.DoStatus(const AText: string);
begin
  TVdxUtils.PrintLn(AText);
end;

procedure TVdxConsoleChat.DoError(const AText: string);
begin
  TVdxUtils.PrintLn(COLOR_RED + AText + COLOR_RESET);
end;

procedure TVdxConsoleChat.DoInfo(const AText: string);
begin
  TVdxUtils.PrintLn(COLOR_CYAN + AText + COLOR_RESET);
end;

procedure TVdxConsoleChat.DoStartup();
begin
  TVdxUtils.PrintLn(COLOR_BOLD + COLOR_CYAN +
    'VindexLLM Chat' + COLOR_RESET);
  TVdxUtils.PrintLn(COLOR_CYAN +
    'Type /help for commands, /quit to exit.' + COLOR_RESET);
  TVdxUtils.PrintLn();
  TVdxUtils.PrintLn('Loading model...');
end;

procedure TVdxConsoleChat.DoShutdown();
begin
  TVdxUtils.PrintLn();
  TVdxUtils.PrintLn(COLOR_CYAN + 'Goodbye!' + COLOR_RESET);
end;

procedure TVdxConsoleChat.DoGenerationComplete();
begin
  FTokenWriter.Reset();
  TVdxUtils.PrintLn();
end;

end.