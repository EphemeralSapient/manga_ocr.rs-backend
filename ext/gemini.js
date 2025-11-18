// Gemini API Client - Direct API calls from browser
class GeminiClient {
  constructor() {
    this.apiKey = null;
    this.model = 'gemini-flash-lite-latest';
    this.endpoint = 'https://generativelanguage.googleapis.com/v1beta/models';
    console.log('[GEMINI] GeminiClient instance created');
    console.log(`[GEMINI] Model: ${this.model}`);
    console.log(`[GEMINI] Endpoint: ${this.endpoint}`);
  }

  async initialize() {
    console.log('[GEMINI] Initializing API client...');
    console.log('[GEMINI] Fetching API key from Chrome storage...');

    // Get API key from storage
    const result = await chrome.storage.sync.get(['geminiApiKey']);
    this.apiKey = result.geminiApiKey;

    if (!this.apiKey) {
      console.error('[GEMINI] ✗ API key not found in storage');
      throw new Error('Gemini API key not set. Please configure in extension popup.');
    }

    console.log(`[GEMINI] ✓ API key retrieved (length: ${this.apiKey.length} chars)`);
  }

  async translateBubbles(imageElement, detections) {
    console.log('[GEMINI] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log(`[GEMINI] Starting translation of ${detections.length} bubbles...`);

    if (!this.apiKey) {
      console.log('[GEMINI] API key not loaded, initializing...');
      await this.initialize();
    }

    const translationStartTime = performance.now();

    // Crop all bubbles
    console.log('[GEMINI] Step 1/3: Cropping bubble regions...');
    const cropStartTime = performance.now();
    const bubbleImages = [];
    for (let i = 0; i < detections.length; i++) {
      const detection = detections[i];
      console.log(`[GEMINI] Cropping bubble ${i + 1}/${detections.length}...`);
      const cropData = await this.cropImage(imageElement, detection.textBox);
      const [x1, y1, x2, y2] = detection.textBox;
      const width = x2 - x1;
      const height = y2 - y1;
      console.log(`[GEMINI] ✓ Bubble ${i + 1} cropped: ${width}x${height}px, base64 size: ${cropData.length} chars`);
      bubbleImages.push(cropData);
    }
    const cropEndTime = performance.now();
    console.log(`[GEMINI] ✓ All bubbles cropped (${(cropEndTime - cropStartTime).toFixed(0)}ms)`);
    console.log(`[GEMINI] Total cropped images: ${bubbleImages.length}`);

    // Build request with all bubble images
    console.log('[GEMINI] Step 2/3: Building API request...');
    const parts = [];

    // Add all images
    console.log(`[GEMINI] Adding ${bubbleImages.length} images to request...`);
    for (let i = 0; i < bubbleImages.length; i++) {
      parts.push({
        inlineData: {
          mimeType: 'image/png',
          data: bubbleImages[i],
        },
      });
    }

    // Add instruction
    const instructionText = `Extract and translate Korean text from all ${bubbleImages.length} speech bubble images above. Return translations in the same order as the images (bubble 1, 2, 3, etc.).`;
    parts.push({
      text: instructionText,
    });

    console.log(`[GEMINI] Request structure: ${bubbleImages.length} images + instruction`);
    console.log(`[GEMINI] Temperature: 0.3`);
    console.log(`[GEMINI] Response format: JSON`);

    // Make API call
    const url = `${this.endpoint}/${this.model}:generateContent?key=${this.apiKey.substring(0, 8)}...`;
    console.log(`[GEMINI] API URL: ${url}`);
    console.log('[GEMINI] Sending request to Gemini API...');

    const apiStartTime = performance.now();
    const fullUrl = `${this.endpoint}/${this.model}:generateContent?key=${this.apiKey}`;

    const requestBody = {
      contents: [{ parts }],
      systemInstruction: {
        parts: [{
          text: this.getSystemInstruction(),
        }],
      },
      generationConfig: {
        responseMimeType: 'application/json',
        responseSchema: this.getResponseSchema(),
        temperature: 0.3,
      },
    };

    const requestSize = JSON.stringify(requestBody).length;
    console.log(`[GEMINI] Request payload size: ${(requestSize / 1024).toFixed(2)} KB`);

    const response = await fetch(fullUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    const apiEndTime = performance.now();
    const apiTime = apiEndTime - apiStartTime;
    console.log(`[GEMINI] ✓ API response received (${apiTime.toFixed(0)}ms / ${(apiTime/1000).toFixed(2)}s)`);

    if (!response.ok) {
      console.error(`[GEMINI] ✗ API request failed with status ${response.status}`);
      const error = await response.json();
      console.error('[GEMINI] Error details:', error);
      throw new Error(`Gemini API error: ${error.error?.message || 'Unknown error'}`);
    }

    const result = await response.json();
    console.log('[GEMINI] Response structure:', Object.keys(result));

    // Parse response
    console.log('[GEMINI] Step 3/3: Parsing response...');
    const content = result.candidates[0].content.parts[0].text;
    console.log(`[GEMINI] Response text length: ${content.length} chars`);

    const translations = JSON.parse(content);
    const translationCount = translations.translations.length;
    console.log(`[GEMINI] ✓ Parsed ${translationCount} translations`);

    // Log each translation
    translations.translations.forEach((trans, idx) => {
      console.log(`[GEMINI] Translation ${idx + 1}:`);
      console.log(`[GEMINI]   Korean: "${trans.korean_text}"`);
      console.log(`[GEMINI]   English: "${trans.english_translation}"`);
      console.log(`[GEMINI]   Tone: ${trans.tone}`);
      if (trans.context_notes) {
        console.log(`[GEMINI]   Notes: ${trans.context_notes}`);
      }
    });

    const totalTime = performance.now() - translationStartTime;
    console.log('[GEMINI] ━━━ Translation Summary ━━━');
    console.log(`[GEMINI] Total translations: ${translationCount}`);
    console.log(`[GEMINI] Total time: ${totalTime.toFixed(0)}ms (${(totalTime/1000).toFixed(2)}s)`);
    console.log(`[GEMINI] API call time: ${apiTime.toFixed(0)}ms`);
    console.log('[GEMINI] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

    return translations.translations;
  }

  async cropImage(imageElement, box) {
    const [x1, y1, x2, y2] = box;
    const width = x2 - x1;
    const height = y2 - y1;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    // Draw cropped region
    ctx.drawImage(
      imageElement,
      x1, y1, width, height,
      0, 0, width, height
    );

    // Convert to base64 (remove data URL prefix)
    const dataUrl = canvas.toDataURL('image/png');
    return dataUrl.split(',')[1];
  }

  getSystemInstruction() {
    return `You are an expert Korean to English translator specializing in webtoon and comic dialogue translation.

Your responsibilities:
1. You will receive MULTIPLE speech bubble images numbered sequentially
2. For EACH bubble, extract ALL Korean text EXACTLY as written (including spacing, line breaks)
3. Translate each to natural, conversational English that flows well in comic speech bubbles
4. Preserve the speaker's tone, emotion, and personality for each bubble
5. Adapt Korean speech patterns to natural English equivalents
6. Keep translations concise and suitable for limited bubble space

Translation guidelines:
- Use natural English expressions, NOT literal word-for-word translations
- Adapt Korean idioms and cultural references to English equivalents when appropriate
- Preserve emotional emphasis (!, ?, ..., —)
- Convert Korean honorifics to English tone/formality (e.g., 반말 → casual, 존댓말 → polite)
- Maintain character consistency (informal speakers stay informal, formal stay formal)
- For sound effects (의성어/의태어), use English comic conventions (e.g., "쿵" → "THUD", "휴" → "sigh")
- Keep dialogue punchy and readable - comics need brevity

Tone identification:
- Identify the emotional tone: casual, formal, angry, excited, sad, sarcastic, etc.
- This helps maintain character voice consistency across translations

Context notes:
- Only provide notes if there are important cultural references, wordplay, or untranslatable nuances
- Keep notes brief and actionable for editors

Output format:
- Return translations in the SAME ORDER as the input bubbles
- Use bubble_number to match each translation to its input (1 for first bubble, 2 for second, etc.)`;
  }

  getResponseSchema() {
    return {
      type: 'object',
      properties: {
        translations: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              bubble_number: {
                type: 'integer',
                description: 'The bubble number (1-indexed)'
              },
              korean_text: {
                type: 'string',
                description: 'The exact Korean text extracted from the image'
              },
              english_translation: {
                type: 'string',
                description: 'Natural English translation suitable for comic dialogue'
              },
              tone: {
                type: 'string',
                description: 'The tone/emotion of the text (e.g., casual, formal, angry, excited)'
              },
              context_notes: {
                type: 'string',
                description: 'Any cultural or contextual notes for translators (optional)',
                nullable: true
              }
            },
            required: ['bubble_number', 'korean_text', 'english_translation', 'tone'],
          },
        },
      },
      required: ['translations'],
    };
  }
}

window.GeminiClient = GeminiClient;