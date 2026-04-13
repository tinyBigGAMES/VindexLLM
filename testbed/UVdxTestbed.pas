unit UVdxTestbed;

interface

procedure RunVdxTestbed();

implementation

uses
  System.SysUtils,
  VindexLLM.Utils,
  VindexLLM.VulkanCompute;


procedure StatusCallback(const AText: string; const AUserData: Pointer);
begin
  TVdxUtils.PrintLn(AText);
end;

procedure Test01();
var
  LCompute: TVdxVulkanCompute;
begin
  LCompute := TVdxVulkanCompute.Create();
  try
    LCompute.SetStatusCallback(StatusCallback);
    LCompute.Init();
    TvdxUtils.Pause();
  finally
    LCompute.Free();
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
