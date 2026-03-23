(function () {
  const body = document.body;
  if (!body) return;

  const targetLang = body.dataset.currentLang || 'en';
  const defaultLang = body.dataset.defaultLang || 'en';

  if (targetLang === defaultLang) {
    return;
  }

  const CACHE_PREFIX = 'agromate_i18n_googletrans_v3_';
  const ATTRS = ['placeholder', 'title', 'aria-label'];

  function isTranslatableText(text) {
    const trimmed = (text || '').trim();
    if (!trimmed) return false;
    if (/^[\d\s.,:%+\-()]+$/.test(trimmed)) return false;
    if (trimmed.length < 2) return false;
    // Very long blocks are expensive and often fail in free translators.
    if (trimmed.length > 220) return false;
    return true;
  }

  function collectTextNodes() {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.parentElement) return NodeFilter.FILTER_REJECT;

        const parent = node.parentElement;
        const tag = parent.tagName;
        if (['SCRIPT', 'STYLE', 'NOSCRIPT', 'CODE', 'PRE', 'TEXTAREA'].includes(tag)) {
          return NodeFilter.FILTER_REJECT;
        }

        if (parent.closest('[data-translate-skip="true"]')) {
          return NodeFilter.FILTER_REJECT;
        }

        if (!isTranslatableText(node.nodeValue)) {
          return NodeFilter.FILTER_REJECT;
        }

        return NodeFilter.FILTER_ACCEPT;
      }
    });

    const refs = [];
    let node;
    while ((node = walker.nextNode())) {
      refs.push(node);
    }
    return refs;
  }

  function collectAttributeNodes() {
    const refs = [];
    const selector = ATTRS.map((attr) => `[${attr}]`).join(',');

    document.querySelectorAll(selector).forEach((el) => {
      if (el.closest('[data-translate-skip="true"]')) {
        return;
      }

      ATTRS.forEach((attr) => {
        const value = el.getAttribute(attr);
        if (isTranslatableText(value)) {
          refs.push({ el, attr, value: value.trim() });
        }
      });
    });

    return refs;
  }

  function loadCache(lang) {
    try {
      const raw = localStorage.getItem(CACHE_PREFIX + lang);
      return raw ? JSON.parse(raw) : {};
    } catch (err) {
      return {};
    }
  }

  function saveCache(lang, cache) {
    try {
      localStorage.setItem(CACHE_PREFIX + lang, JSON.stringify(cache));
    } catch (err) {
      // Ignore storage errors.
    }
  }

  function chunkArray(arr, size) {
    const chunks = [];
    for (let i = 0; i < arr.length; i += size) {
      chunks.push(arr.slice(i, i + size));
    }
    return chunks;
  }

  async function requestTranslations(strings, lang) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    const response = await fetch('/api/translate-batch', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ texts: strings, target: lang }),
      signal: controller.signal
    });
    clearTimeout(timer);

    if (!response.ok) {
      throw new Error('Translation API request failed');
    }

    const data = await response.json();
    return data.translations || {};
  }

  async function translatePage() {
    const textNodes = collectTextNodes();
    const attrRefs = collectAttributeNodes();

    const allStrings = new Set();
    textNodes.forEach((node) => allStrings.add(node.nodeValue.trim()));
    attrRefs.forEach((ref) => allStrings.add(ref.value));

    if (allStrings.size === 0) {
      return;
    }

    const cache = loadCache(targetLang);
    const uncached = [...allStrings].filter((s) => typeof cache[s] !== 'string');

    if (uncached.length > 0) {
      const chunks = chunkArray(uncached, 25);
      for (const chunk of chunks) {
        try {
          const translatedMap = await requestTranslations(chunk, targetLang);
          Object.entries(translatedMap).forEach(([src, translated]) => {
            const out = (translated || '').trim();
            if (!out) return;
            if (targetLang !== defaultLang && out === src) return;
            cache[src] = out;
          });
        } catch (err) {
          // Skip failed chunk and continue translating remaining chunks.
          continue;
        }
      }
      saveCache(targetLang, cache);
    }

    textNodes.forEach((node) => {
      const original = node.nodeValue.trim();
      if (cache[original]) {
        const leading = node.nodeValue.match(/^\s*/)[0] || '';
        const trailing = node.nodeValue.match(/\s*$/)[0] || '';
        node.nodeValue = `${leading}${cache[original]}${trailing}`;
      }
    });

    attrRefs.forEach((ref) => {
      if (cache[ref.value]) {
        ref.el.setAttribute(ref.attr, cache[ref.value]);
      }
    });
  }

  translatePage().catch(() => {
    // Silent failover: keep original English UI.
  });

  // Some pages finalize content after initial paint; run a second pass.
  window.addEventListener('load', () => {
    setTimeout(() => {
      translatePage().catch(() => {
        // Keep original text if second pass fails.
      });
    }, 600);
  });
})();
