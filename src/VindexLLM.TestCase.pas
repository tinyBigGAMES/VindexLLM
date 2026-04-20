{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.TestCase;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Generics.Collections,
  VindexLLM.Utils;

type

  { TVdxTestCase }
  TVdxTestCase = class(TVdxBaseObject)
  private
    FTitle: string;
    FAllPassed: Boolean;
    FSectionIndex: Integer;
  protected
    // Subclasses implement the actual test body here. Called by
    // Execute between banner and summary. Inside Run the subclass calls
    // Section / Check / FlushErrors freely.
    procedure Run(); virtual; abstract;
  public
    constructor Create(); override;

    // One-shot entry point: prints the banner, invokes Run, prints the
    // pass/fail summary. Resets FAllPassed / FSectionIndex up front so
    // the same instance can be re-executed if the caller wants.
    procedure Execute();

    // Prints a dim, auto-numbered sub-section header. Each call
    // increments FSectionIndex, so the caller never writes numbers.
    procedure Section(const ATitle: string);

    // Records one assertion. Prints [PASS] green / [FAIL] red and
    // flips FAllPassed to False on any failure. The test's overall
    // result is the AND of every Check call in Run.
    procedure Check(const ACondition: Boolean; const ALabel: string);

    // Prints every entry in AErrors with color-coded severity
    // (HINT / WARN / ERROR / FATAL). Nil-safe, empty-safe.
    procedure PrintErrors(const AErrors: TVdxErrors);

    // PrintErrors followed by AErrors.Clear — use this at the end of
    // each object's lifetime in a test so subsequent checks aren't
    // polluted by stale entries from a prior operation.
    procedure FlushErrors(const AErrors: TVdxErrors);

    property Title: string read FTitle write FTitle;
    property AllPassed: Boolean read FAllPassed;
  end;

  { TVdxTestCaseClass }
  // Metaclass reference — lets VdxRunTestCase accept the subclass
  // type (e.g. TVirtualBufferTest) without needing an instance.
  TVdxTestCaseClass = class of TVdxTestCase;

// Instantiates ATestClass, runs its Execute, frees it. Returns the
// test's overall pass flag so a caller can chain or aggregate
// multiple test runs. Safe to call repeatedly.
function VdxRunTestCase(const ATestClass: TVdxTestCaseClass): Boolean;

implementation

{ TVdxTestCase }

constructor TVdxTestCase.Create();
begin
  inherited;
  FTitle := '';
  FAllPassed := True;
  FSectionIndex := 0;
end;

procedure TVdxTestCase.Execute();
begin
  // Reset so the same instance can be Execute'd multiple times.
  FAllPassed := True;
  FSectionIndex := 0;

  // Banner
  TVdxUtils.PrintLn('');
  TVdxUtils.PrintLn(COLOR_CYAN + COLOR_BOLD + '--- %s ---' + COLOR_RESET,
    [FTitle]);

  Run();

  // Summary
  if FAllPassed then
    TVdxUtils.PrintLn(COLOR_GREEN + COLOR_BOLD +
      '=== %s: ALL PASSED ===' + COLOR_RESET, [FTitle])
  else
    TVdxUtils.PrintLn(COLOR_RED + COLOR_BOLD +
      '=== %s: FAILED ===' + COLOR_RESET, [FTitle]);
end;

procedure TVdxTestCase.Section(const ATitle: string);
begin
  Inc(FSectionIndex);
  TVdxUtils.PrintLn('');
  TVdxUtils.PrintLn(COLOR_BLUE + '  [ %d. %s ]' + COLOR_RESET,
    [FSectionIndex, ATitle]);
end;

procedure TVdxTestCase.Check(const ACondition: Boolean; const ALabel: string);
begin
  if ACondition then
    TVdxUtils.PrintLn(COLOR_GREEN + '  [PASS] ' + COLOR_RESET + '%s',
      [ALabel])
  else
  begin
    TVdxUtils.PrintLn(COLOR_RED + '  [FAIL] ' + COLOR_RESET + '%s',
      [ALabel]);
    FAllPassed := False;
  end;
end;

procedure TVdxTestCase.PrintErrors(const AErrors: TVdxErrors);
var
  LItems: TList<TVdxError>;
  LI: Integer;
  LErr: TVdxError;
  LColor: string;
  LLabel: string;
begin
  if AErrors = nil then
    Exit;
  LItems := AErrors.GetItems();
  if LItems.Count = 0 then
    Exit;

  TVdxUtils.PrintLn('');
  for LI := 0 to LItems.Count - 1 do
  begin
    LErr := LItems[LI];
    case LErr.Severity of
      esHint:
      begin
        LColor := COLOR_CYAN;
        LLabel := 'HINT';
      end;
      esWarning:
      begin
        LColor := COLOR_YELLOW;
        LLabel := 'WARN';
      end;
      esError:
      begin
        LColor := COLOR_RED;
        LLabel := 'ERROR';
      end;
      esFatal:
      begin
        LColor := COLOR_MAGENTA;
        LLabel := 'FATAL';
      end;
    else
      LColor := COLOR_WHITE;
      LLabel := '?';
    end;

    if LErr.Code <> '' then
      TVdxUtils.PrintLn(LColor + '[%s] %s: %s',
        [LLabel, LErr.Code, LErr.Message])
    else
      TVdxUtils.PrintLn(LColor + '[%s] %s', [LLabel, LErr.Message]);
  end;
end;

procedure TVdxTestCase.FlushErrors(const AErrors: TVdxErrors);
begin
  PrintErrors(AErrors);
  if AErrors <> nil then
    AErrors.Clear();
end;

{ VdxRunTestCase }

function VdxRunTestCase(const ATestClass: TVdxTestCaseClass): Boolean;
var
  LTest: TVdxTestCase;
begin
  Result := False;
  if ATestClass = nil then
    Exit;

  // Virtual constructor dispatch — ATestClass.Create() runs the
  // most-derived override, so we actually get a fully-initialized
  // subclass instance even though LTest is declared as the base.
  LTest := ATestClass.Create();
  try
    LTest.Execute();
    Result := LTest.AllPassed;
  finally
    LTest.Free();
  end;
end;

end.
