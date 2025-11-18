// Popup script
document.addEventListener('DOMContentLoaded', () => {
  const statusEl = document.getElementById('status');
  const apiKeyInput = document.getElementById('apiKeyInput');
  const saveKeyBtn = document.getElementById('saveKeyBtn');
  const translateBtn = document.getElementById('translateBtn');
  const restoreBtn = document.getElementById('restoreBtn');

  // Load saved API key
  loadApiKey();

  // Save API key button
  saveKeyBtn.addEventListener('click', () => {
    const apiKey = apiKeyInput.value.trim();

    if (!apiKey) {
      statusEl.textContent = '❌ Please enter an API key';
      statusEl.className = 'status not-ready';
      return;
    }

    chrome.storage.sync.set({ geminiApiKey: apiKey }, () => {
      statusEl.textContent = '✓ API key saved - Ready to translate!';
      statusEl.className = 'status ready';
      translateBtn.disabled = false;
      apiKeyInput.value = '';
    });
  });

  // Translate button
  translateBtn.addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, { action: 'translate' });
      window.close();
    });
  });

  // Restore button
  restoreBtn.addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, { action: 'restore' });
      window.close();
    });
  });

  async function loadApiKey() {
    const result = await chrome.storage.sync.get(['geminiApiKey']);

    if (result.geminiApiKey) {
      statusEl.textContent = '✓ API key configured - Ready!';
      statusEl.className = 'status ready';
      translateBtn.disabled = false;
    } else {
      statusEl.textContent = '⚠ Please configure Gemini API key';
      statusEl.className = 'status not-ready';
      translateBtn.disabled = true;
    }
  }
});