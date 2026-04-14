unit UVdxTestbed;

interface

procedure RunVdxTestbed();

implementation

uses
  System.SysUtils,
  System.IOUtils,
  System.Math,
  VindexLLM.Utils,
  VindexLLM.Inference;


procedure StatusCallback(const AText: string; const AUserData: Pointer);
begin
  TVdxUtils.PrintLn(AText);
end;

function PrintToken(const AToken: string; const AUserData: Pointer): Boolean;
begin
  Write(AToken);
  Result := True;
end;

procedure Test01();
var
  LInference: TVdxInference;
begin
  TVdxUtils.Pause();
  LInference := TVdxInference.Create();
  try
    LInference.SetStatusCallback(StatusCallback, nil);
    LInference.LoadModel('C:\Dev\LLM\GGUF\gemma-3-4b-it-f16.gguf');
    LInference.SetTokenCallback(PrintToken, nil);
    //WriteLn(LInference.Generate('What is the capital of France?'));
    LInference.Generate('What is the capital of France?');
    LInference.UnloadModel();
  finally
    LInference.Free();
  end;
end;

procedure RunVdxTestbed();
var
  LIndex: Integer;
begin
  try
    LIndex := 1;

    case LIndex of
      1: Test01();
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
