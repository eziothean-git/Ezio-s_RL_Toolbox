// sse.js — SSE 连接 + 状态更新 + Console
import state from './state.js';
import { apiUrl, fmtTime } from './api.js';

// rebuildTargetSelector 通过动态 import 避免 sse↔training 循环依赖
function _rebuildTargetSelector() {
  import('./training.js').then(function(m) { m.rebuildTargetSelector(); });
}

// ── SSE ──

export function connectSSE() {
  if (state.sse) state.sse.close();
  var url = state.activeTarget ? apiUrl('/stream') : '/stream';
  state.sse = new EventSource(url);
  state.sse.onopen = function() {
    document.getElementById('connDot').className = 'dot on';
    document.getElementById('connText').textContent = state.activeTarget ? ('connected: ' + state.activeTarget) : 'connected';
  };
  state.sse.onerror = function() {
    document.getElementById('connDot').className = 'dot off';
    document.getElementById('connText').textContent = 'reconnecting...';
  };
  state.sse.onmessage = function(e) {
    try { var d = JSON.parse(e.data); } catch(_) { return; }
    if (d.type === 'status') updateState(d);
    else if (d.type === 'console') appendConsole(d.line);
    else if (d.type === 'train') updateTrain(d);
    else if (d.type === 'system') updateGPU(d);
    else if (d.type === 'fleet_health') onFleetHealth(d);
    else if (d.type === 'fleet_console') onFleetConsole(d);
    else if (d.type === 'fleet_op') onFleetOp(d);
  };
}

export function connectFleetSSE() {
  if (state.fleetSSE) state.fleetSSE.close();
  if (!state.activeTarget) { state.fleetSSE = null; return; }
  state.fleetSSE = new EventSource('/stream');
  state.fleetSSE.onmessage = function(e) {
    try { var d = JSON.parse(e.data); } catch(_) { return; }
    if (d.type === 'fleet_health') onFleetHealth(d);
    else if (d.type === 'fleet_console') onFleetConsole(d);
    else if (d.type === 'fleet_op') onFleetOp(d);
  };
}

export function fetchStatus() {
  fetch(apiUrl('/status')).then(function(r) { return r.json(); }).then(function(d) {
    updateState(d);
    updateGPU(d);
    if (d.iteration && d.tot_iter) updateTrain({iteration: d.iteration, extras: {tot_iter: d.tot_iter}});
  }).catch(function() {});
}

// ── State updates ──

function updateState(d) {
  var st = d.state || 'stopped';
  state.currentState = st;
  var el = document.getElementById('stateLabel');
  el.textContent = st;
  el.className = 'state state-' + st;
  var running = st === 'running' || st === 'halted';
  document.getElementById('btnStart').disabled = running;
  document.getElementById('btnStop').disabled = !running && st !== 'halted';
  document.getElementById('btnHalt').disabled = st !== 'running';
  document.getElementById('btnResume').disabled = st !== 'halted';
  document.getElementById('btnCkpt').disabled = st !== 'running';
  var port = document.getElementById('cfgSignalPort').value || '7002';
  var link = document.getElementById('scopeLink');
  if (running) {
    link.href = 'http://' + location.hostname + ':' + port;
    link.style.display = '';
    link.textContent = 'Oscilloscope :' + port;
  } else {
    link.style.display = 'none';
  }
}

function updateTrain(d) {
  var it = d.iteration || 0;
  var tot = (d.extras && d.extras.tot_iter) || 0;
  if (it > 0) document.getElementById('iterLabel').textContent = it + '/' + tot;
}

function updateGPU(d) {
  if (!d.gpus || !d.gpus.length) return;
  var parts = [];
  d.gpus.forEach(function(g) {
    parts.push('GPU' + g.idx + ': ' + g.util + '% ' + g.mem_used + '/' + g.mem_total + 'M ' + g.temp + 'C');
  });
  document.getElementById('gpuLabel').textContent = parts.join(' | ');
  if (d.eta_s > 0) document.getElementById('etaLabel').textContent = 'ETA: ' + fmtTime(d.eta_s);
}

// ── Console ──

export function loadConsoleHistory() {
  fetch(apiUrl('/console?n=200')).then(function(r) { return r.json(); }).then(function(d) {
    if (d.lines) d.lines.forEach(function(l) { appendConsole(l); });
  }).catch(function() {});
}

export function clearConsole() {
  document.getElementById('console').innerHTML = '';
  state.consoleLines = 0;
  document.getElementById('consoleCount').textContent = '0 lines';
}

export function appendConsole(text, isErr) {
  var el = document.getElementById('console');
  var div = document.createElement('div');
  div.className = 'line' + (isErr ? ' line-err' : '');
  div.textContent = text;
  el.appendChild(div);
  state.consoleLines++;
  el.scrollTop = el.scrollHeight;
  while (el.children.length > 2000) el.removeChild(el.firstChild);
  document.getElementById('consoleCount').textContent = state.consoleLines + ' lines';
}

// ── Fleet SSE Handlers ──

function onFleetHealth(d) {
  if (!d.servers) return;
  for (var sid in d.servers) {
    if (state.fleetServers[sid]) state.fleetServers[sid].health = d.servers[sid];
  }
  import('./fleet.js').then(function(m) { m.renderServerCards(state.fleetServers); });
  _rebuildTargetSelector();
}

function onFleetConsole(d) {
  if (state.currentFleetOp && d.server === state.currentFleetOp.server) {
    var el = document.getElementById('opModalConsole');
    var div = document.createElement('div');
    div.className = 'line';
    div.textContent = d.line;
    el.appendChild(div);
    el.scrollTop = el.scrollHeight;
    while (el.children.length > 1000) el.removeChild(el.firstChild);
  }
}

function onFleetOp(d) {
  if (state.currentFleetOp && d.server === state.currentFleetOp.server) {
    var statusEl = document.getElementById('opModalStatus');
    if (d.status === 'running') {
      statusEl.textContent = d.msg || 'running...';
      statusEl.style.color = 'var(--accent)';
    } else if (d.status === 'done') {
      statusEl.textContent = 'completed successfully';
      statusEl.style.color = 'var(--ok)';
      import('./fleet.js').then(function(m) { setTimeout(m.refreshFleetList, 2000); });
    } else if (d.status === 'error') {
      statusEl.textContent = 'failed: ' + (d.msg || 'rc=' + d.returncode);
      statusEl.style.color = 'var(--warn)';
    }
  }
}

