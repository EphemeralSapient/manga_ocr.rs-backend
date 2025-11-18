// Background script - Handle keyboard shortcuts
console.log('[BACKGROUND] Service worker loaded');

// Listen for keyboard shortcut: Ctrl+Q
chrome.commands.onCommand.addListener((command) => {
  console.log(`[BACKGROUND] Command received: ${command}`);

  if (command === 'translate-all-images') {
    console.log('[BACKGROUND] Toggle translation command triggered');

    // Send message to content script to toggle
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        console.log(`[BACKGROUND] Sending toggle message to tab ${tabs[0].id}`);
        chrome.tabs.sendMessage(
          tabs[0].id,
          { action: 'translate-all' },
          (response) => {
            if (chrome.runtime.lastError) {
              console.error('[BACKGROUND] Error:', chrome.runtime.lastError.message);
            } else {
              console.log('[BACKGROUND] Response:', response);
              if (response && response.enabled !== undefined) {
                console.log(`[BACKGROUND] Translation mode: ${response.enabled ? 'ON' : 'OFF'}`);
              }
            }
          }
        );
      }
    });
  }
});

// Listen for messages from content script (e.g., fetch image requests)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('[BACKGROUND] Message received:', request);

  if (request.action === 'fetch-image') {
    console.log(`[BACKGROUND] Fetching image: ${request.url}`);

    // Fetch image with CORS bypass (background has special permissions)
    fetch(request.url, {
      method: 'GET',
      credentials: 'include'
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.blob();
      })
      .then(blob => {
        console.log(`[BACKGROUND] Image fetched: ${(blob.size / 1024).toFixed(2)} KB`);

        // Convert blob to base64 for message passing
        const reader = new FileReader();
        reader.onloadend = () => {
          sendResponse({
            success: true,
            dataUrl: reader.result,
            size: blob.size
          });
        };
        reader.onerror = () => {
          sendResponse({
            success: false,
            error: 'Failed to read blob'
          });
        };
        reader.readAsDataURL(blob);
      })
      .catch(error => {
        console.error('[BACKGROUND] Fetch error:', error);
        sendResponse({
          success: false,
          error: error.message
        });
      });

    // Return true to indicate async response
    return true;
  }
});

console.log('[BACKGROUND] Ready to handle keyboard shortcuts (Ctrl+Q) and image fetch requests');
