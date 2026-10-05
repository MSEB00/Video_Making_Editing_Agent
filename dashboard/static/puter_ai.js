/* Puter.js bridge — free, keyless creative AI executed in the user's browser.
 *
 * The Flask/SocketIO backend asks this page to run creative model calls
 * (edit planning, music ranking, render review, revision) through the user's
 * own Puter session. No API keys ever touch this project. On the first call
 * Puter.js opens its sign-in popup automatically; if the user declines, the
 * backend falls back to the measured local planner.
 */
(function () {
  'use strict';

  if (typeof socket === 'undefined') {
    console.warn('[puter_ai] socket not available; bridge disabled');
    return;
  }

  const DEFAULT_MODEL = 'qwen/qwen3.7-plus';

  async function dataUrlToFile(dataUrl, name) {
    const response = await fetch(dataUrl);
    const blob = await response.blob();
    return new File([blob], name, { type: blob.type || 'image/jpeg' });
  }

  function extractText(response) {
    if (typeof response === 'string') return response;
    const content = response && response.message ? response.message.content : undefined;
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
      return content.map((part) => (part && part.text) || '').join('');
    }
    if (response && typeof response.text === 'string') return response.text;
    return response == null ? '' : String(response);
  }

  function addSystemNote(text) {
    try {
      if (typeof addStep === 'function') addStep('🧠', text);
    } catch (e) { /* chat UI optional */ }
  }

  socket.on('puter_ai_request', async (data, ack) => {
    const reply = (payload) => { try { if (typeof ack === 'function') ack(payload); } catch (e) {} };
    try {
      if (typeof puter === 'undefined') {
        return reply({ ok: false, error: 'puter.js is not loaded (check network access to js.puter.com)' });
      }
      const messages = [
        { role: 'system', content: String(data.system || '') },
        { role: 'user', content: JSON.stringify(data.payload || {}) },
      ];
      const options = {
        model: data.model || DEFAULT_MODEL,
        temperature: 0.3,
      };
      if (data.max_tokens) options.max_tokens = data.max_tokens;

      const files = [];
      for (let i = 0; i < (data.frames || []).length; i++) {
        const frame = data.frames[i];
        if (frame && typeof frame.data_url === 'string' && frame.data_url.startsWith('data:')) {
          try { files.push(await dataUrlToFile(frame.data_url, 'frame_' + i + '.jpg')); }
          catch (e) { console.warn('[puter_ai] frame conversion failed', e); }
        }
      }

      addSystemNote('Asking Puter AI (' + options.model + ') — sign in to Puter if prompted...');
      let response;
      if (files.length) {
        try {
          response = await puter.ai.chat(messages, files, false, options);
        } catch (mediaError) {
          console.warn('[puter_ai] multimodal call failed; retrying text-only:', mediaError);
          response = await puter.ai.chat(messages, false, options);
        }
      } else {
        response = await puter.ai.chat(messages, false, options);
      }
      const text = extractText(response);
      if (!text || !text.trim()) {
        return reply({ ok: false, error: 'Puter returned an empty response' });
      }
      reply({ ok: true, text: text });
    } catch (err) {
      const message = err && err.message ? err.message : String(err);
      console.error('[puter_ai] request failed:', message);
      const hint = /auth|sign.?in|login|permission|popup|user/i.test(message)
        ? ' Tip: allow popups for localhost, open https://puter.com in a normal tab and sign in there first (or run puter.auth.signIn() in the browser console), then retry the request.'
        : '';
      reply({ ok: false, error: String(message).slice(0, 300) + hint });
    }
  });

  console.info('[puter_ai] Puter.js bridge ready (free, keyless creative AI)');
})();
