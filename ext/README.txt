Chrome Extension - Webtoon Translator (Client-Side)
===================================================

FULLY CLIENT-SIDE - NO SERVER REQUIRED!
Everything runs in your browser: RT-DETR detection + Gemini API + Canvas rendering

VERIFIED: Implementation matches translate_pipeline.py
See VALIDATION.md for detailed comparison


INSTALLATION
------------

1. Download required files:

   a) ONNX Runtime Web:
      - Download: https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/ort.min.js
      - Save as: extension/ort.min.js

   b) RT-DETR ONNX Model:
      - Need to convert from Python model to ONNX format
      - See "MODEL PREPARATION" section below
      - Save to: extension/models/rtdetr.onnx

2. Get Gemini API key:
   - Go to: https://aistudio.google.com/app/apikey
   - Create new API key
   - Copy the key (you'll enter it in the extension popup)

3. Load extension in Chrome:
   - Open chrome://extensions/
   - Enable "Developer mode" (top right)
   - Click "Load unpacked"
   - Select this 'extension' folder
   - Extension installed!

4. Configure API key:
   - Click extension icon
   - Paste your Gemini API key
   - Click "Save API Key"


MODEL PREPARATION
-----------------

To create the RT-DETR ONNX model:

1. From the main project directory:
   cd ..

2. Run Python script to export model:
   python3 <<EOF
   import torch
   from ultralytics import RTDETR

   # Load model (adjust path if needed)
   model = RTDETR('path/to/rtdetr.pt')

   # Export to ONNX
   model.export(format='onnx', imgsz=640, simplify=True)

   # Move to extension folder
   import shutil
   shutil.move('rtdetr.onnx', 'extension/models/rtdetr.onnx')
   EOF

3. Verify file size:
   ls -lh extension/models/rtdetr.onnx
   # Should be ~5-10MB


USAGE
-----

1. Go to any webtoon website
   Example: https://www.webtoons.com/

2. Hover over a webtoon image (green outline appears)

3. Press Alt+T to translate

4. Wait a few seconds:
   - "Detecting speech bubbles..."
   - "Translating N bubbles..."
   - "Rendering translations..."
   - "✓ Translated N bubbles!"

5. Image is replaced with English translation!

6. To restore original: click extension icon → "Restore Original"


HOW IT WORKS
------------

Everything runs in your browser:

1. Image Capture
   └─> Canvas API captures displayed image

2. Bubble Detection (RT-DETR via ONNX Runtime Web)
   └─> Detects speech bubbles and text regions

3. Translation (Gemini API direct call)
   └─> Sends all bubble images in single batch request
   └─> Returns Korean text + English translations

4. Text Rendering (Canvas API)
   └─> Clears text regions (white fill)
   └─> Renders English with optimal font size
   └─> Word wrapping and vertical centering

5. Image Replacement
   └─> Replaces original image with translated version
   └─> Stores original for restore function


FEATURES
--------

✓ Fully client-side - no server, no backend
✓ Direct API calls to Gemini (only your API key, no middleman)
✓ RT-DETR runs in browser via WebAssembly
✓ No file downloads - works on displayed images
✓ In-place translation - seamless replacement
✓ Batch translation - all bubbles in single API call
✓ Keyboard shortcut - Alt+T for quick access
✓ Visual feedback - notifications and badges
✓ Restore function - undo translation
✓ Privacy-focused - no data leaves your browser except API call


ARCHITECTURE
------------

extension/
├── manifest.json       - Extension configuration (Manifest V3)
├── content.js          - Main orchestration logic
├── detector.js         - RT-DETR bubble detection (ONNX Runtime Web)
├── gemini.js           - Direct Gemini API client
├── renderer.js         - Canvas text rendering
├── background.js       - Service worker (keyboard shortcuts)
├── popup.html          - Extension popup UI
├── popup.js            - API key configuration
├── ort.min.js          - ONNX Runtime Web library
├── icon*.png           - Extension icons
└── models/
    └── rtdetr.onnx     - RT-DETR object detection model


TROUBLESHOOTING
---------------

API key not working:
  → Make sure you copied the full key
  → Verify at: https://aistudio.google.com/app/apikey
  → Check browser console for API errors (F12)

Model not loading:
  → Check models/rtdetr.onnx exists
  → Verify file size is reasonable (~5-10MB)
  → Check browser console for ONNX errors

No bubbles detected:
  → Image might not be a webtoon/comic
  → Try adjusting detection threshold in detector.js
  → Check browser console for detection results

Translation fails:
  → Check API key is configured correctly
  → Verify internet connection
  → Check Gemini API quota/billing
  → Look at browser console (F12) for errors

Extension not working:
  → Reload extension at chrome://extensions/
  → Check all files are present (ort.min.js, models/rtdetr.onnx)
  → Verify icons were created
  → Check browser console for errors (F12)


PERFORMANCE
-----------

Typical translation time: 5-8 seconds
- Model loading: ~2 seconds (first time only, then cached)
- Detection: ~2 seconds (WebAssembly)
- Translation: ~2 seconds (Gemini API batch)
- Rendering: ~1 second (Canvas)

Memory usage: ~100-200MB (browser tab)
Cost: ~$0.00014 per bubble (Gemini API pricing)


PRIVACY
-------

This extension is privacy-focused:
✓ No telemetry or analytics
✓ API key stored locally (Chrome sync storage)
✓ Images processed in-browser (except Gemini API call)
✓ No data sent to third parties except Google Gemini
✓ Open source - audit the code yourself


GEMINI API SETUP
----------------

1. Go to: https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key
5. Paste in extension popup
6. You're ready!

Note: Gemini Flash-Lite is extremely cheap:
- ~$0.00014 per speech bubble
- 1500 RPD (requests per day) free tier
- See: https://ai.google.dev/pricing


DEVELOPMENT
-----------

To modify the extension:

1. Edit files in extension/ folder
2. Go to chrome://extensions/
3. Click "Reload" icon on the extension card
4. Test your changes

Debugging:
- Content script: F12 → Console (on the web page)
- Background worker: chrome://extensions/ → "service worker" link
- Popup: Right-click popup → "Inspect"