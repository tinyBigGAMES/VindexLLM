@echo off
setlocal

set GLSLC=..\.claude\tools\glslangValidator.exe

echo Compiling shaders...

%GLSLC% -V double_floats.comp -o double_floats.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: double_floats.comp
    exit /b 1
)

echo Done.
