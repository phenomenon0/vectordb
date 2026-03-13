(function() {
  if (window.__deepdataDesktopInjected) return;
  window.__deepdataDesktopInjected = true;

  document.documentElement.style.background = '#0a0a0a';
  document.body.style.background = '#0a0a0a';
  document.body.style.margin = '0';

  function getAuthToken() {
    return (localStorage.getItem('deepdataApiAuthToken') || '').trim();
  }

  function formatAuthHeader(token) {
    if (!token) return '';
    return /^Bearer\s+/i.test(token) ? token : 'Bearer ' + token;
  }

  function setAuthToken(token) {
    const value = (token || '').trim();
    if (value) {
      localStorage.setItem('deepdataApiAuthToken', value);
    } else {
      localStorage.removeItem('deepdataApiAuthToken');
    }
  }

  function parseJSONBody(body) {
    if (!body || typeof body !== 'string') return null;
    try {
      return JSON.parse(body);
    } catch (_err) {
      return null;
    }
  }

  function isDeepDataURL(url) {
    try {
      const parsed = new URL(url, window.location.href);
      return parsed.origin === window.location.origin;
    } catch (_err) {
      return false;
    }
  }

  function rewriteRequest(url, init) {
    const parsed = new URL(url, window.location.href);
    const nextInit = Object.assign({}, init || {});
    const headers = new Headers(nextInit.headers || {});
    const authHeader = formatAuthHeader(getAuthToken());
    if (authHeader && !headers.has('Authorization') && isDeepDataURL(parsed.href)) {
      headers.set('Authorization', authHeader);
    }

    const contentType = headers.get('Content-Type') || headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      nextInit.headers = headers;
      return { url: parsed.href, init: nextInit };
    }

    const body = parseJSONBody(nextInit.body);
    if (!body) {
      nextInit.headers = headers;
      return { url: parsed.href, init: nextInit };
    }

    let rewrittenBody = body;

    if (parsed.pathname === '/insert') {
      rewrittenBody = Object.assign({}, body);
      if (rewrittenBody.text && !rewrittenBody.doc) {
        rewrittenBody.doc = rewrittenBody.text;
      }
      if (rewrittenBody.metadata && !rewrittenBody.meta) {
        rewrittenBody.meta = rewrittenBody.metadata;
      }
      delete rewrittenBody.text;
      delete rewrittenBody.metadata;
    } else if (parsed.pathname === '/delete') {
      rewrittenBody = Object.assign({}, body);
      if (Array.isArray(rewrittenBody.ids) && rewrittenBody.ids.length > 0 && !rewrittenBody.id) {
        rewrittenBody.id = rewrittenBody.ids[0];
      }
      delete rewrittenBody.ids;
      delete rewrittenBody.collection;
    }

    nextInit.headers = headers;
    nextInit.body = JSON.stringify(rewrittenBody);
    return { url: parsed.href, init: nextInit };
  }

  function normalizeResponsePayload(url, payload) {
    const parsed = new URL(url, window.location.href);
    if (!payload || typeof payload !== 'object') {
      return payload;
    }

    if (parsed.pathname === '/query' && Array.isArray(payload.ids) && Array.isArray(payload.docs)) {
      const ids = payload.ids || [];
      const docs = payload.docs || [];
      const scores = payload.scores || [];
      const meta = payload.meta || [];
      const results = ids.map(function(id, index) {
        return {
          id: id,
          doc_id: id,
          text: docs[index] || '',
          document: docs[index] || '',
          score: typeof scores[index] === 'number' ? scores[index] : null,
          metadata: meta[index] || null
        };
      });

      return Object.assign({}, payload, { results: results });
    }

    if (parsed.pathname === '/scroll' && Array.isArray(payload.docs)) {
      const next = Object.assign({}, payload);
      if (!Array.isArray(next.documents)) {
        next.documents = next.docs;
      }
      if (!Array.isArray(next.metadata) && Array.isArray(next.meta)) {
        next.metadata = next.meta;
      }
      return next;
    }

    return payload;
  }

  function rebuildJSONResponse(response, payload) {
    const headers = new Headers(response.headers);
    headers.set('content-type', 'application/json');
    return new Response(JSON.stringify(payload), {
      status: response.status,
      statusText: response.statusText,
      headers: headers
    });
  }

  const originalFetch = window.fetch.bind(window);
  let authPromptActive = false;

  window.fetch = async function(input, init) {
    const sourceURL = typeof input === 'string' ? input : input.url;
    const rewritten = rewriteRequest(sourceURL, init);
    let response = await originalFetch(rewritten.url, rewritten.init);

    if (
      response.status === 401 &&
      isDeepDataURL(rewritten.url) &&
      !getAuthToken() &&
      !authPromptActive
    ) {
      authPromptActive = true;
      const token = window.prompt('DeepData auth token or JWT');
      authPromptActive = false;
      if (token) {
        setAuthToken(token);
        const retried = rewriteRequest(sourceURL, init);
        response = await originalFetch(retried.url, retried.init);
      }
    }

    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      return response;
    }

    const payload = await response.clone().json().catch(function() {
      return null;
    });
    if (payload === null) {
      return response;
    }

    const normalized = normalizeResponsePayload(rewritten.url, payload);
    if (normalized === payload) {
      return response;
    }

    return rebuildJSONResponse(response, normalized);
  };

  const nav = document.querySelector('nav');
  if (!nav) return;

  nav.style.webkitAppRegion = 'drag';
  nav.querySelectorAll('a, button, input, select, span[class]').forEach(function(el) {
    el.style.webkitAppRegion = 'no-drag';
  });

  nav.addEventListener('dblclick', function(e) {
    if (e.target === nav || e.target.classList.contains('topnav-brand')) {
      window.__TAURI_INVOKE__('win_toggle_maximize');
    }
  });

  const controls = document.createElement('div');
  controls.id = '__dd_wc';
  controls.style.cssText = 'display:flex;align-items:center;margin-left:auto;padding-left:8px;-webkit-app-region:no-drag;flex-shrink:0;gap:4px;';

  function makeButton(label, onclick, hoverBg) {
    const button = document.createElement('button');
    button.textContent = label;
    button.style.cssText = 'background:none;border:none;color:#666;min-width:36px;height:28px;padding:0 8px;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:11px;transition:background 0.15s,color 0.15s;border-radius:4px;text-transform:uppercase;font-family:monospace;';
    button.addEventListener('click', function(e) {
      e.stopPropagation();
      onclick();
    });
    button.addEventListener('mouseenter', function() {
      button.style.background = hoverBg;
      button.style.color = '#fff';
    });
    button.addEventListener('mouseleave', function() {
      button.style.background = 'none';
      syncAuthButton();
    });
    return button;
  }

  const authButton = makeButton('auth', function() {
    const current = getAuthToken();
    const next = window.prompt('DeepData auth token or JWT', current);
    if (next === null) return;
    setAuthToken(next);
    syncAuthButton();
  }, 'rgba(59,130,246,0.2)');

  function syncAuthButton() {
    authButton.style.color = getAuthToken() ? '#3b82f6' : '#666';
  }

  controls.appendChild(authButton);
  controls.appendChild(makeButton('-', function() {
    window.__TAURI_INVOKE__('win_minimize');
  }, 'rgba(255,255,255,0.1)'));
  controls.appendChild(makeButton('[]', function() {
    window.__TAURI_INVOKE__('win_toggle_maximize');
  }, 'rgba(255,255,255,0.1)'));
  controls.appendChild(makeButton('x', function() {
    window.__TAURI_INVOKE__('win_close');
  }, 'rgba(239,68,68,0.7)'));

  syncAuthButton();
  nav.appendChild(controls);
})();
