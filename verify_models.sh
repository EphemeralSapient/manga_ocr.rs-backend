#!/bin/bash
# Shell script to verify ONNX model checksums on Linux/Mac
# Run this script in the project root directory

set -e

echo "=== ONNX Model Verification ==="
echo ""

cd models
if sha256sum -c CHECKSUMS.txt --quiet 2>/dev/null; then
    echo "✓ All model files verified successfully!"
    exit 0
else
    echo "✗ Some model files failed verification!"
    echo "Please re-download or re-transfer the models."
    sha256sum -c CHECKSUMS.txt
    exit 1
fi
