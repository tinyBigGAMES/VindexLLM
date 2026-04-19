{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Model.Registry;

interface

uses
  System.Generics.Collections,
  VindexLLM.Model;

type
  // ---------------------------------------------------------------------------
  // TVdxModelRegistry
  //
  // Internal machinery that maps GGUF `general.architecture` strings to the
  // concrete TVdxModel descendant that handles them. Concrete model units
  // self-register in their `initialization` section — consumers never touch
  // the registry directly, they go through TVdxModel.LoadModel.
  // ---------------------------------------------------------------------------
  TVdxModelRegistry = class
  private
    // Architecture name (lower-case) → concrete class reference.
    class var FMap: TDictionary<string, TVdxModelClass>;

    class constructor ClassCreate();
    class destructor  ClassDestroy();
  public
    // Register AClass under every name returned by its
    // SupportedArchitectures() class method. Safe to call from unit
    // initialization. Duplicate registrations are last-wins and
    // intentionally silent — different builds may swap implementations.
    class procedure RegisterClass(const AClass: TVdxModelClass);

    // Returns the concrete class for AArchitecture, or nil if nothing is
    // registered under that name. Matching is case-insensitive.
    class function ResolveClass(const AArchitecture: string): TVdxModelClass;

    // Snapshot of every architecture currently registered. Used to build
    // a readable error message when resolution fails.
    class function ListArchitectures(): TArray<string>;
  end;

implementation

uses
  System.SysUtils;

{ TVdxModelRegistry }

class constructor TVdxModelRegistry.ClassCreate();
begin
  FMap := TDictionary<string, TVdxModelClass>.Create();
end;

class destructor TVdxModelRegistry.ClassDestroy();
begin
  FMap.Free();
end;

// ---------------------------------------------------------------------------
// Register AClass under every name in its SupportedArchitectures() set.
// Names are normalized to lower-case so lookup is case-insensitive.
// ---------------------------------------------------------------------------
class procedure TVdxModelRegistry.RegisterClass(const AClass: TVdxModelClass);
var
  LArchitectures: TArray<string>;
  LArch: string;
begin
  if AClass = nil then Exit;

  LArchitectures := AClass.SupportedArchitectures();
  for LArch in LArchitectures do
  begin
    if LArch = '' then Continue;
    FMap.AddOrSetValue(LowerCase(LArch), AClass);
  end;
end;

// ---------------------------------------------------------------------------
// Case-insensitive lookup. Returns nil for unknown architectures — the
// caller (TVdxModel.LoadFromGGUF) is responsible for turning that into a
// user-visible error using ListArchitectures for context.
// ---------------------------------------------------------------------------
class function TVdxModelRegistry.ResolveClass(
  const AArchitecture: string): TVdxModelClass;
begin
  if not FMap.TryGetValue(LowerCase(AArchitecture), Result) then
    Result := nil;
end;

// ---------------------------------------------------------------------------
// Snapshot of registered architecture names. Order is the dictionary's
// enumeration order — consumers should sort it if they need stable output.
// ---------------------------------------------------------------------------
class function TVdxModelRegistry.ListArchitectures(): TArray<string>;
var
  LList: TList<string>;
  LArch: string;
begin
  LList := TList<string>.Create();
  try
    for LArch in FMap.Keys do
      LList.Add(LArch);
    Result := LList.ToArray();
  finally
    LList.Free();
  end;
end;

end.
