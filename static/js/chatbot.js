/* ================================================================
   AgroMate – AgroBot Popup Chatbot  (Groq / llama-3.3-70b)
================================================================ */

(function () {
  'use strict';

  /* ── DOM refs ──────────────────────────────────────────────── */
  const fab      = document.getElementById('agrobot-fab');
  const win      = document.getElementById('agrobot-window');
  const closeBtn = document.getElementById('agrobot-close');
  const msgArea  = document.getElementById('agrobot-messages');
  const input    = document.getElementById('agrobot-input');
  const sendBtn  = document.getElementById('agrobot-send');
  const chipsEl  = document.getElementById('agrobot-chips');
  const badge    = document.getElementById('agrobot-badge');

  if (!fab || !win) return;

  // Unique session id per browser tab (persists while tab is open)
  const SESSION_ID = 'agro_' + Math.random().toString(36).slice(2, 10);

  let isOpen         = false;
  let isTyping       = false;
  let sessionGreeted = false;

  /* ── Open / Close ──────────────────────────────────────────── */
  function openChat() {
    isOpen = true;
    fab.classList.add('open');
    win.classList.add('open');
    if (badge) badge.style.opacity = '0';
    input.focus();

    if (!sessionGreeted) {
      sessionGreeted = true;
      setTimeout(() => {
        appendBotMsg(
          "👋 Hi! I'm **AgroBot**, powered by Llama 3.3-70B!\n\n" +
          "Ask me anything about agriculture — crops, soil, diseases, fertilizers, pests, irrigation, weather, and more. 🌱"
        );
      }, 200);
    }
  }

  function closeChat() {
    isOpen = false;
    fab.classList.remove('open');
    win.classList.remove('open');
  }

  fab.addEventListener('click', () => (isOpen ? closeChat() : openChat()));
  closeBtn.addEventListener('click', closeChat);

  /* ── Chip shortcuts ────────────────────────────────────────── */
  if (chipsEl) {
    chipsEl.addEventListener('click', (e) => {
      const chip = e.target.closest('.chip');
      if (!chip) return;
      const msg = chip.dataset.msg;
      if (msg) sendMessage(msg);
    });
  }

  /* ── Send on button click or Enter ────────────────────────── */
  sendBtn.addEventListener('click', () => sendMessage(input.value));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input.value);
    }
  });

  /* Auto-resize textarea */
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 80) + 'px';
  });

  /* ── Core send function ────────────────────────────────────── */
  function sendMessage(text) {
    text = text.trim();
    if (!text || isTyping) return;

    appendUserMsg(text);
    input.value = '';
    input.style.height = 'auto';

    const typingId = showTyping();
    isTyping = true;
    sendBtn.disabled = true;

    fetch('/chatbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: SESSION_ID })
    })
      .then((res) => {
        if (!res.ok) throw new Error('Network error ' + res.status);
        return res.json();
      })
      .then((data) => {
        removeTyping(typingId);
        isTyping = false;
        sendBtn.disabled = false;
        setTimeout(() => appendBotMsg(data.reply || '…'), 150);
      })
      .catch(() => {
        removeTyping(typingId);
        isTyping = false;
        sendBtn.disabled = false;
        appendBotMsg('⚠️ Connection error. Please check the server and try again.');
      });
  }

  /* ── Message helpers ───────────────────────────────────────── */
  function appendUserMsg(text) {
    const row    = document.createElement('div');
    row.className = 'msg-row user';
    const bubble  = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;
    row.appendChild(bubble);
    msgArea.appendChild(row);
    scrollBottom();
  }

  function appendBotMsg(text) {
    const row    = document.createElement('div');
    row.className = 'msg-row bot';
    const avatar  = document.createElement('div');
    avatar.className = 'bot-mini-avatar';
    avatar.textContent = '🌿';
    const bubble  = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = parseMarkdown(text);
    row.appendChild(avatar);
    row.appendChild(bubble);
    msgArea.appendChild(row);
    scrollBottom();
  }

  function showTyping() {
    const id  = 'typing-' + Date.now();
    const row = document.createElement('div');
    row.className = 'msg-row bot';
    row.id = id;
    const avatar = document.createElement('div');
    avatar.className = 'bot-mini-avatar';
    avatar.textContent = '🌿';
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    row.appendChild(avatar);
    row.appendChild(bubble);
    msgArea.appendChild(row);
    scrollBottom();
    return id;
  }

  function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function scrollBottom() {
    msgArea.scrollTop = msgArea.scrollHeight;
  }

  /* ── Markdown → HTML (bold, code, line breaks) ─────────────── */
  function parseMarkdown(text) {
    let html = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    // **bold**
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // `inline code`
    html = html.replace(/`(.+?)`/g, '<code style="background:#e8f5e9;padding:1px 5px;border-radius:4px;font-size:.8em;">$1</code>');
    // newlines → <br>
    html = html.replace(/\n/g, '<br>');
    return html;
  }

})();


(function () {
  'use strict';

  /* ── DOM refs ──────────────────────────────────────────────── */
  const fab      = document.getElementById('agrobot-fab');
  const win      = document.getElementById('agrobot-window');
  const closeBtn = document.getElementById('agrobot-close');
  const msgArea  = document.getElementById('agrobot-messages');
  const input    = document.getElementById('agrobot-input');
  const sendBtn  = document.getElementById('agrobot-send');
  const chipsEl  = document.getElementById('agrobot-chips');
  const badge    = document.getElementById('agrobot-badge');

  if (!fab || !win) return;          // guard if elements missing

  let isOpen        = false;
  let isTyping      = false;
  let sessionGreeted = false;

  /* ── Open / Close ──────────────────────────────────────────── */
  function openChat() {
    isOpen = true;
    fab.classList.add('open');
    win.classList.add('open');
    if (badge) badge.style.opacity = '0';
    input.focus();

    // Show greeting on first open
    if (!sessionGreeted) {
      sessionGreeted = true;
      setTimeout(() => {
        appendBotMsg(
          "👋 Hi! I'm **AgroBot**, your AI agriculture assistant!\n\n" +
          "I can help you with crops, fertilizers, soil health, plant diseases, " +
          "pest control, irrigation, and more.\n\nWhat's on your farming mind? 🌱"
        );
      }, 200);
    }
  }

  function closeChat() {
    isOpen = false;
    fab.classList.remove('open');
    win.classList.remove('open');
  }

  fab.addEventListener('click', () => (isOpen ? closeChat() : openChat()));
  closeBtn.addEventListener('click', closeChat);

  /* ── Chip shortcuts ────────────────────────────────────────── */
  if (chipsEl) {
    chipsEl.addEventListener('click', (e) => {
      const chip = e.target.closest('.chip');
      if (!chip) return;
      const msg = chip.dataset.msg;
      if (msg) sendMessage(msg);
    });
  }

  /* ── Send on button click or Enter (Shift+Enter = newline) ─── */
  sendBtn.addEventListener('click', () => sendMessage(input.value));

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input.value);
    }
  });

  /* Auto-resize textarea */
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 80) + 'px';
  });

  /* ── Core send function ────────────────────────────────────── */
  function sendMessage(text) {
    text = text.trim();
    if (!text || isTyping) return;

    // Append user bubble
    appendUserMsg(text);
    input.value = '';
    input.style.height = 'auto';

    // Show typing indicator
    const typingId = showTyping();
    isTyping = true;

    // POST to Flask /chatbot
    fetch('/chatbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    })
      .then((res) => {
        if (!res.ok) throw new Error('Network response was not ok.');
        return res.json();
      })
      .then((data) => {
        removeTyping(typingId);
        isTyping = false;
        // Small delay for realism
        setTimeout(() => appendBotMsg(data.reply || '…'), 200);
      })
      .catch(() => {
        removeTyping(typingId);
        isTyping = false;
        appendBotMsg(
          "⚠️ Sorry, I'm having trouble connecting right now. " +
          "Please try again in a moment!"
        );
      });
  }

  /* ── Append helpers ────────────────────────────────────────── */
  function appendUserMsg(text) {
    const row = document.createElement('div');
    row.className = 'msg-row user';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;

    row.appendChild(bubble);
    msgArea.appendChild(row);
    scrollBottom();
  }

  function appendBotMsg(text) {
    const row = document.createElement('div');
    row.className = 'msg-row bot';

    const avatar = document.createElement('div');
    avatar.className = 'bot-mini-avatar';
    avatar.textContent = '🌿';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    // Render basic markdown: **bold**, bullet points preserved via pre-line CSS
    bubble.innerHTML = parseMarkdown(text);

    row.appendChild(avatar);
    row.appendChild(bubble);
    msgArea.appendChild(row);
    scrollBottom();
  }

  function showTyping() {
    const id = 'typing-' + Date.now();
    const row = document.createElement('div');
    row.className = 'msg-row bot';
    row.id = id;

    const avatar = document.createElement('div');
    avatar.className = 'bot-mini-avatar';
    avatar.textContent = '🌿';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML =
      '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    row.appendChild(avatar);
    row.appendChild(bubble);
    msgArea.appendChild(row);
    scrollBottom();
    return id;
  }

  function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  /* ── Scroll to bottom ──────────────────────────────────────── */
  function scrollBottom() {
    msgArea.scrollTop = msgArea.scrollHeight;
  }

  /* ── Minimal markdown parser (bold + line breaks) ─────────── */
  function parseMarkdown(text) {
    // Escape HTML first
    let html = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // **bold**
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Line breaks → <br>
    html = html.replace(/\n/g, '<br>');

    return html;
  }

})();
