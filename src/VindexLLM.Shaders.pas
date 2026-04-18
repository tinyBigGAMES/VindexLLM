{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Shaders;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils;

// Load a compiled SPIR-V shader from embedded resources.
// AName is the resource identifier (e.g., 'MATVEC_F16', 'RMSNORM').
// Returns the raw .spv bytes. Raises Exception if the shader is not
// present in the binary — execution cannot continue (the .res is
// baked in at compile time, so a missing shader is a build/deployment
// failure with no sensible runtime recovery).
function VdxLoadShader(const AName: string): TBytes;

implementation

uses
  System.Classes,
  WinAPI.Windows,
  VindexLLM.Utils,
  VindexLLM.Resources;

{$R VindexLLM.Shaders.res}

function VdxLoadShader(const AName: string): TBytes;
var
  LStream: TResourceStream;
begin
  SetLength(Result, 0);

  if not TVdxUtils.ResourceExist(AName) then
    raise Exception.CreateFmt(RSShNotFound, [AName]);

  LStream := TResourceStream.Create(HInstance, AName, RT_RCDATA);
  try
    SetLength(Result, LStream.Size);
    if LStream.Size > 0 then
      LStream.ReadBuffer(Result[0], LStream.Size);
  finally
    LStream.Free();
  end;
end;

end.
