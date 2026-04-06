// api.js — API 路由 + 工具函数
import state from './state.js';

export function apiUrl(path) {
  if (state.activeTarget) return '/fleet/' + state.activeTarget + path;
  return path;
}

export function esc(s) {
  var d = document.createElement('span');
  d.textContent = s;
  return d.innerHTML;
}

export function showToast(msg, isError) {
  var el = document.createElement('div');
  el.style.cssText = 'position:fixed;top:12px;right:12px;padding:8px 16px;border-radius:4px;font-size:12px;z-index:9999;animation:fadeIn 0.2s;' +
    (isError ? 'background:#3a1a1a;border:1px solid var(--warn);color:var(--warn)' : 'background:#0a3a2c;border:1px solid var(--ok);color:var(--ok)');
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(function() { el.remove(); }, 3000);
}

export function fmtTime(s) {
  if (s <= 0) return '';
  var h = Math.floor(s / 3600);
  var m = Math.floor((s % 3600) / 60);
  if (h > 0) return h + 'h' + m + 'm';
  if (m > 0) return m + 'm';
  return Math.round(s) + 's';
}
