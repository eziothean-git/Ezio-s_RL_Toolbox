// training.js — 训练控制 + Container 管理
import state from './state.js';
import { apiUrl, showToast } from './api.js';

// appendConsole 通过动态 import 避免 training↔sse 循环依赖
function appendConsole(text, isErr) {
  import('./sse.js').then(function(m) { m.appendConsole(text, isErr); });
}

// ── Target Selector ──

export function rebuildTargetSelector() {
  ['cfgTarget'].forEach(function(selId) {
    var sel = document.getElementById(selId);
    if (!sel) return;
    var cur = sel.value;
    sel.innerHTML = '<option value="">Local</option>';
    var ids = Object.keys(state.fleetServers);
    ids.forEach(function(sid) {
      var s = state.fleetServers[sid];
      var health = (s.health && s.health.status) || 'unknown';
      var opt = document.createElement('option');
      opt.value = sid;
      opt.textContent = (s.name || sid) + (health === 'online' ? '' : ' [' + health + ']');
      sel.appendChild(opt);
    });
    if (cur && state.fleetServers[cur]) sel.value = cur;
    else sel.value = '';
  });
  onTargetChange();
}

export function onTargetChange() {
  var sel1 = document.getElementById('cfgTarget');
  state.activeTarget = sel1.value;
  var sid = state.activeTarget;

  var dotEl = document.getElementById('targetDot');
  var hint = document.getElementById('targetHint');
  if (!sid) {
    dotEl.style.background = 'var(--ok)';
    if (hint) hint.textContent = '';
  } else {
    var s = state.fleetServers[sid];
    var health = (s && s.health && s.health.status) || 'unknown';
    dotEl.style.background = health === 'online' ? 'var(--ok)' : health === 'manager_down' ? '#f0ad4e' : 'var(--warn)';
    if (hint) hint.textContent = health === 'online' ? 'will run on ' + (s.name || sid) : 'server ' + health;
  }

  // 延迟导入 SSE 模块避免循环
  import('./sse.js').then(function(m) {
    m.connectSSE();
    m.loadConsoleHistory();
    m.fetchStatus();
  });
  fetchContainerStatus();
}

// ── Start Training ──

export function startTraining() {
  if (!state.selected) { showToast('Please select an experiment or task in the Editor tab first', true); return; }
  var body = {
    num_envs: parseInt(document.getElementById('cfgEnvs').value) || 16,
    extra_args: [],
  };
  if (state.selected.type === 'experiment') {
    body.experiment = state.selected.name;
  } else {
    body.task = state.selected.name;
    // task 选中时也带上 experiment（树形层级下 task 归属于 experiment）
    if (state.selected.experiment) body.experiment = state.selected.experiment;
  }
  var maxIter = document.getElementById('cfgIter').value;
  if (maxIter) body.max_iterations = parseInt(maxIter);
  var device = document.getElementById('cfgDevice').value;
  if (device && device !== 'cuda:0') body.extra_args.push('--device', device);
  var sigPort = document.getElementById('cfgSignalPort').value;
  if (sigPort) body.signal_server_port = parseInt(sigPort) || 7002;
  body.headless = document.getElementById('cfgHeadless').checked;
  var extra = document.getElementById('cfgExtra').value.trim();
  if (extra) body.extra_args = body.extra_args.concat(extra.split(/\s+/));

  var btn = document.getElementById('btnStart');
  btn.disabled = true;
  fetch(apiUrl('/start'), {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (!d.ok) appendConsole('[Editor] Start failed: ' + (d.msg || d.error), true);
      btn.disabled = false;
    })
    .catch(function(e) {
      appendConsole('[Editor] Request failed: ' + e, true);
      btn.disabled = false;
    });
}

// ── Control buttons ──

export function postCmd(cmd) {
  fetch(apiUrl('/' + cmd), {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'})
    .then(function(r) { return r.json(); })
    .catch(function() {});
}

// ── Container ──

export function fetchContainerStatus() {
  fetch(apiUrl('/container')).then(function(r) { return r.json(); }).then(function(d) {
    state.containerRunning = d.running;
    if (d.direct) {
      // direct mode：没有容器管理，禁用按钮
      document.getElementById('containerDot').className = 'dot gray';
      document.getElementById('containerState').textContent = 'direct mode (no container)';
      document.getElementById('btnContainerStart').disabled = true;
      document.getElementById('btnContainerStop').disabled = true;
    } else {
      document.getElementById('containerDot').className = 'dot ' + (d.running ? 'on' : 'off');
      document.getElementById('containerState').textContent = (d.running ? 'running' : 'stopped') + ' (' + d.name + ')';
      document.getElementById('btnContainerStart').disabled = d.running;
      document.getElementById('btnContainerStop').disabled = !d.running;
    }
  }).catch(function() {});
}

export function containerCmd(action) {
  var btn = (action === 'start') ? document.getElementById('btnContainerStart') : document.getElementById('btnContainerStop');
  btn.disabled = true;
  btn.textContent = action === 'start' ? 'Starting...' : 'Stopping...';
  fetch(apiUrl('/container/' + action), {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'})
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (!d.ok) appendConsole('[Editor] Container ' + action + ' failed: ' + d.msg, true);
      setTimeout(fetchContainerStatus, 2000);
    })
    .catch(function(e) { appendConsole('[Editor] ' + e, true); })
    .finally(function() { btn.textContent = action === 'start' ? 'Start' : 'Stop'; });
}
