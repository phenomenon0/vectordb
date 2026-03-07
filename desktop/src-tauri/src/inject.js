(function() {
  if (document.getElementById('__dd_wc')) return;

  document.documentElement.style.background = '#0a0a0a';
  document.body.style.background = '#0a0a0a';
  document.body.style.margin = '0';

  const nav = document.querySelector('nav');
  if (!nav) return;

  // Make nav bar draggable (empty space acts as drag handle)
  nav.style.webkitAppRegion = 'drag';
  // All interactive children must opt out of drag
  nav.querySelectorAll('a, button, input, select, span[class]').forEach(function(el) {
    el.style.webkitAppRegion = 'no-drag';
  });

  // Double-click nav to maximize/restore
  nav.addEventListener('dblclick', function(e) {
    if (e.target === nav || e.target.classList.contains('topnav-brand')) {
      window.__TAURI_INVOKE__('win_toggle_maximize');
    }
  });

  // Window controls (appended to the right side of existing nav)
  const controls = document.createElement('div');
  controls.id = '__dd_wc';
  controls.style.cssText = 'display:flex;align-items:center;margin-left:auto;padding-left:8px;-webkit-app-region:no-drag;flex-shrink:0;';

  function btn(label, onclick, hoverBg) {
    const b = document.createElement('button');
    b.innerHTML = label;
    b.style.cssText = 'background:none;border:none;color:#666;width:36px;height:28px;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:13px;transition:background 0.15s,color 0.15s;border-radius:4px;';
    b.addEventListener('click', function(e) { e.stopPropagation(); onclick(); });
    b.addEventListener('mouseenter', function() { b.style.background = hoverBg; b.style.color = '#fff'; });
    b.addEventListener('mouseleave', function() { b.style.background = 'none'; b.style.color = '#666'; });
    return b;
  }

  controls.appendChild(btn('\u2013', function(){ window.__TAURI_INVOKE__('win_minimize'); }, 'rgba(255,255,255,0.1)'));
  controls.appendChild(btn('\u25A1', function(){ window.__TAURI_INVOKE__('win_toggle_maximize'); }, 'rgba(255,255,255,0.1)'));
  controls.appendChild(btn('\u2715', function(){ window.__TAURI_INVOKE__('win_close'); }, 'rgba(239,68,68,0.7)'));

  nav.appendChild(controls);
})();
