# PowerShell script to verify ONNX model checksums on Windows
# Run this script in the project root directory

Write-Host "=== ONNX Model Verification ===" -ForegroundColor Cyan
Write-Host ""

$expectedChecksums = @{
    "models\detector.onnx" = "065744e91c0594ad8663aa8b870ce3fb27222942eded5a3cc388ce23421bd195"
    "models\mask.onnx"     = "36c26bdefe150226acd9669772e9ff5a011fa0dd4622469b49d3d5e359f3251c"
}

$allValid = $true

foreach ($file in $expectedChecksums.Keys) {
    if (-not (Test-Path $file)) {
        Write-Host "[MISSING] $file" -ForegroundColor Red
        $allValid = $false
        continue
    }

    Write-Host "Checking $file..." -NoNewline
    $hash = (Get-FileHash -Algorithm SHA256 $file).Hash.ToLower()
    $expected = $expectedChecksums[$file]

    if ($hash -eq $expected) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "  Expected: $expected" -ForegroundColor Yellow
        Write-Host "  Got:      $hash" -ForegroundColor Yellow
        $allValid = $false
    }
}

Write-Host ""
if ($allValid) {
    Write-Host "All model files verified successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Some model files failed verification!" -ForegroundColor Red
    Write-Host "Please re-download or re-transfer the models." -ForegroundColor Yellow
    exit 1
}
