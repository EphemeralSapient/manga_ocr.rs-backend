@echo off
REM Batch launcher script for manga_workflow (NVIDIA CUDA version)
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

if not exist "%DLL1%" (
    echo ERROR: Missing required DLL: %DLL1%
    exit /b 1
)
echo   [OK] opencv_world4120.dll

REM Check for CUDA DLLs (may be in system PATH or libs directory)
echo.
echo Checking for CUDA runtime libraries...
echo NOTE: CUDA DLLs may be installed system-wide via CUDA Toolkit

REM Check common CUDA DLL locations
set "CUDA_FOUND=0"

REM Check if CUDA DLLs are in libs\windows\x64\
if exist "%SCRIPT_DIR%\libs\windows\x64\cudart64_*.dll" (
    echo   [OK] Found CUDA Runtime in libs\windows\x64\
    set "CUDA_FOUND=1"
)

REM Check if CUDA is in system PATH
where cudart64_12.dll >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Found CUDA Runtime in system PATH
    set "CUDA_FOUND=1"
)

where cudart64_11.dll >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Found CUDA Runtime in system PATH
    set "CUDA_FOUND=1"
)

if "%CUDA_FOUND%"=="0" (
    echo   [WARNING] CUDA Runtime DLLs not found in libs\ or system PATH
    echo   [WARNING] Make sure CUDA Toolkit is installed or CUDA DLLs are in libs\windows\x64\
    echo.
    echo   Required CUDA DLLs typically include:
    echo     - cudart64_*.dll (CUDA Runtime^)
    echo     - cublas64_*.dll or cublasLt64_*.dll (CUDA BLAS^)
    echo     - cudnn64_*.dll (cuDNN^)
    echo.
    set /p "CONTINUE=Continue anyway? (y/n): "
    if /i not "!CONTINUE!"=="y" (
        echo Aborted by user
        exit /b 1
    )
)

REM Verify required directories exist
echo.
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

REM Find the CUDA executable
set "EXE_PATH="
for %%F in ("%SCRIPT_DIR%\binary\*cuda.exe") do (
    set "EXE_PATH=%%F"
    set "EXE_NAME=%%~nxF"
    goto :found_exe
)

:found_exe
if not defined EXE_PATH (
    echo ERROR: No CUDA executable found in binary\ directory
    echo Expected: binary\*cuda.exe
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
