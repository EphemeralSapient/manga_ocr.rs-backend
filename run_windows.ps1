#!/usr/bin/env pwsh
# PowerShell launcher script for manga_workflow
# Ensures all required DLLs are in PATH before running the Windows executable

# Stop on any error
$ErrorActionPreference = "Stop"

# Get the script's directory (project root)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Define DLL directories relative to script location
$DllDirs = @(
    $ScriptDir,                              # opencv_world4120.dll
    "$ScriptDir\libs\windows\x64"            # DirectML.dll
)

# Verify DLLs exist
$RequiredDlls = @(
    "$ScriptDir\opencv_world4120.dll",
    "$ScriptDir\libs\windows\x64\DirectML.dll"
)

Write-Host "Checking for required DLLs..." -ForegroundColor Cyan
foreach ($dll in $RequiredDlls) {
    if (-not (Test-Path $dll)) {
        Write-Host "ERROR: Missing required DLL: $dll" -ForegroundColor Red
        exit 1
    }
    $dllName = Split-Path -Leaf $dll
    $dllSize = [math]::Round((Get-Item $dll).Length / 1MB, 1)
    Write-Host "  ✓ $dllName ($dllSize MB)" -ForegroundColor Green
}

# Verify required directories exist
$RequiredDirs = @(
    "$ScriptDir\models",
    "$ScriptDir\fonts"
)

Write-Host "Checking for required directories..." -ForegroundColor Cyan
foreach ($dir in $RequiredDirs) {
    if (-not (Test-Path $dir)) {
        Write-Host "ERROR: Missing required directory: $dir" -ForegroundColor Red
        exit 1
    }
    $dirName = Split-Path -Leaf $dir
    $fileCount = (Get-ChildItem $dir -File).Count
    Write-Host "  ✓ $dirName\ ($fileCount files)" -ForegroundColor Green
}

# Find the executable
$ExePath = Get-ChildItem "$ScriptDir\binary\*.exe" | Select-Object -First 1

if (-not $ExePath) {
    Write-Host "ERROR: No executable found in binary\ directory" -ForegroundColor Red
    exit 1
}

Write-Host "Found executable: $($ExePath.Name)" -ForegroundColor Cyan

# Temporarily modify PATH for this process
$OriginalPath = $env:PATH
$env:PATH = ($DllDirs -join ";") + ";" + $env:PATH

try {
    Write-Host "Starting $($ExePath.Name)..." -ForegroundColor Green
    Write-Host ""

    # Change to project root directory so relative paths (models/, fonts/) work
    Push-Location $ScriptDir

    # Run the executable with all passed arguments
    & $ExePath.FullName $args

    # Capture exit code
    $ExitCode = $LASTEXITCODE

    Write-Host ""
    if ($ExitCode -eq 0) {
        Write-Host "Process completed successfully" -ForegroundColor Green
    } else {
        Write-Host "Process exited with code: $ExitCode" -ForegroundColor Yellow
    }

    exit $ExitCode
}
catch {
    Write-Host "ERROR: Failed to run executable" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
finally {
    # Restore original directory and PATH
    Pop-Location
    $env:PATH = $OriginalPath
}
