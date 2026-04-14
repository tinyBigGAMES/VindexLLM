@echo off
setlocal

set GLSLC=..\.claude\tools\glslangValidator.exe

echo Compiling shaders...

%GLSLC% -V double_floats.comp -o double_floats.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: double_floats.comp
    exit /b 1
)

%GLSLC% -V gate_scan.comp -o gate_scan.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: gate_scan.comp
    exit /b 1
)

%GLSLC% -V accumulate.comp -o accumulate.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: accumulate.comp
    exit /b 1
)

%GLSLC% -V rmsnorm.comp -o rmsnorm.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: rmsnorm.comp
    exit /b 1
)

echo Done.
