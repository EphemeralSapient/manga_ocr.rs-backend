@echo off
REM Batch launcher script for manga_workflow
REM Ensures all required DLLs are in PATH before running the Windows executable

setlocal enabledelayedexpansion

REM Get the script's directory (project root)
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Define DLL directories
set "DLL_DIR1=%SCRIPT_DIR%"
set "DLL_DIR2=%SCRIPT_DIR%\libs\windows\x64"

REM Verify required DLLs exist
echo Checking for required DLLs...

set "DLL1=%SCRIPT_DIR%\opencv_world4120.dll"
set "DLL2=%SCRIPT_DIR%\libs\windows\x64\DirectML.dll"

if not exist "%DLL1%" (
    echo ERROR: Missing required DLL: %DLL1%
    exit /b 1
)
echo   [OK] opencv_world4120.dll

if not exist "%DLL2%" (
    echo ERROR: Missing required DLL: %DLL2%
    exit /b 1
)
echo   [OK] DirectML.dll

REM Verify required directories exist
echo Checking for required directories...

if not exist "%SCRIPT_DIR%\models" (
    echo ERROR: Missing required directory: %SCRIPT_DIR%\models
    exit /b 1
)
echo   [OK] models\

if not exist "%SCRIPT_DIR%\fonts" (
    echo ERROR: Missing required directory: %SCRIPT_DIR%\fonts
    exit /b 1
)
echo   [OK] fonts\

REM Find the DirectML executable
set "EXE_PATH="
for %%F in ("%SCRIPT_DIR%\binary\*directml.exe") do (
    set "EXE_PATH=%%F"
    set "EXE_NAME=%%~nxF"
    goto :found_exe
)

:found_exe
if not defined EXE_PATH (
    echo ERROR: No executable found in binary\ directory
    exit /b 1
)

echo Found executable: %EXE_NAME%
echo.

REM Save original PATH
set "ORIGINAL_PATH=%PATH%"

REM Add DLL directories to PATH
set "PATH=%DLL_DIR1%;%DLL_DIR2%;%PATH%"

REM Change to project root directory
pushd "%SCRIPT_DIR%"

REM Run the executable
echo Starting %EXE_NAME%...
echo.

"%EXE_PATH%"

REM Restore original directory and PATH
popd
set "PATH=%ORIGINAL_PATH%"
