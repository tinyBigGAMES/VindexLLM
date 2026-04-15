{===============================================================================
  VindexLLM - Graph-Walk LLM Inference Engine

  Copyright (c) 2026-present tinyBigGAMES LLC
  All Rights Reserved.

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Shaders;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils;

// Load a compiled SPIR-V shader from embedded resources.
// AName is the resource identifier (e.g., 'MATVEC_F16', 'RMSNORM').
// Returns the raw .spv bytes. Raises exception if not found.
function VdxLoadShader(const AName: string): TBytes;

implementation

uses
  System.Types,
  System.Classes,
  WinAPI.Windows;

{$R VindexLLM.Shaders.res}

function VdxLoadShader(const AName: string): TBytes;
var
  LStream: TResourceStream;
begin
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
