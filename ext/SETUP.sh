#!/bin/bash
# Setup script for Webtoon Translator Chrome Extension (Client-Side)

echo "=========================================="
echo "Webtoon Translator Extension Setup"
echo "=========================================="
echo ""

# Download ONNX Runtime Web
echo "1. Downloading ONNX Runtime Web..."
if [ -f "ort.min.js" ]; then
    echo "   ✓ ort.min.js already exists"
else
    if command -v curl &> /dev/null; then
        curl -L -o ort.min.js https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/ort.min.js
        echo "   ✓ ort.min.js downloaded"
    elif command -v wget &> /dev/null; then
        wget -O ort.min.js https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/ort.min.js
        echo "   ✓ ort.min.js downloaded"
    else
        echo "   ✗ Error: curl or wget not found"
        echo "   Please manually download:"
        echo "   https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/ort.min.js"
        echo "   and save it as ort.min.js in this directory"
    fi
fi
echo ""

# Note about RT-DETR model
echo "2. RT-DETR Model Information"
echo "   ℹ Model is loaded from external URL (not bundled)"
echo "   ℹ Location: HuggingFace (ogkalu/comic-text-and-bubble-detector)"
echo "   ℹ Size: 161MB (downloaded on first use)"
echo "   ℹ Cached by browser after first download"
echo "   ℹ No action needed - automatic on first translation"
echo ""

# Check file sizes
echo "3. Checking extension files..."
if [ -f "ort.min.js" ]; then
    SIZE=$(du -h ort.min.js | cut -f1)
    echo "   ✓ ort.min.js: $SIZE"
else
    echo "   ✗ ort.min.js not found"
fi

if [ -f "manifest.json" ]; then
    echo "   ✓ manifest.json found"
else
    echo "   ✗ manifest.json not found"
fi

if [ -f "content.js" ]; then
    echo "   ✓ content.js found"
else
    echo "   ✗ content.js not found"
fi

if [ -f "detector.js" ]; then
    echo "   ✓ detector.js found"
else
    echo "   ✗ detector.js not found"
fi

if [ -f "gemini.js" ]; then
    echo "   ✓ gemini.js found"
else
    echo "   ✗ gemini.js not found"
fi

if [ -f "renderer.js" ]; then
    echo "   ✓ renderer.js found"
else
    echo "   ✗ renderer.js not found"
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
echo ""

READY=true

if [ ! -f "ort.min.js" ]; then
    echo "❌ Missing: ort.min.js"
    READY=false
fi

if [ ! -f "manifest.json" ] || [ ! -f "content.js" ] || [ ! -f "detector.js" ] || [ ! -f "gemini.js" ] || [ ! -f "renderer.js" ]; then
    echo "❌ Missing: Core extension files"
    READY=false
fi

if [ ! -f "icon16.png" ] || [ ! -f "icon48.png" ] || [ ! -f "icon128.png" ]; then
    echo "⚠ Warning: Extension icons not found"
    echo "   Extension will work but won't have icons"
fi

echo ""

if [ "$READY" = true ]; then
    echo "✅ Extension is ready!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Get Gemini API key: https://aistudio.google.com/app/apikey"
    echo "2. Open chrome://extensions/ in Chrome"
    echo "3. Enable 'Developer mode' (top-right toggle)"
    echo "4. Click 'Load unpacked' and select this folder"
    echo "5. Click the extension icon and save your API key"
    echo "6. Go to a webtoon site and press Alt+T!"
    echo ""
    echo "⚠️ First use: Model will download automatically (~20-30 seconds)"
    echo "   After that, translations are fast (5-8 seconds)"
else
    echo "⚠ Extension setup incomplete"
    echo ""
    echo "Please complete the missing steps above"
fi

echo ""
echo "📚 For detailed instructions, see INSTALL.md"
echo "=========================================="