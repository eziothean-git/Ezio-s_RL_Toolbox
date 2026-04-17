// debug-tools.js — Debug Tools 页面模块
import state from './state.js';
import { apiUrl, showToast } from './api.js';

// ── 状态 ─────────────────────────────────────────────────────────
var debugState = {
  connected: false,
  paused: false,
  timeScale: 1.0,
  bodies: [],
  joints: [],
  numActions: 0,
  muxActiveEnvs: [],
  forceActive: false,
  anchorActive: false,
  vizContacts: false,
  vizTrajectories: false,
  selectedBody: 0,
  selectedEnv: 0,
  pollTimer: null,
};

// ── 初始化 ───────────────────────────────────────────────────────

export function initDebugTools() {
  // 每 2 秒轮询 debug state
  debugState.pollTimer = setInterval(function() {
    var page = document.getElementById('page-debug');
    if (page && page.classList.contains('active')) {
      fetchDebugState();
    }
  }, 2000);
}

// ── 数据获取 ─────────────────────────────────────────────────────

function signalUrl(path) {
  // signal server 端口从 Run 页的 cfgSignalPort 读取
  var port = 7002;
  var el = document.getElementById('cfgSignalPort');
  if (el) port = parseInt(el.value) || 7002;
  // 如果有 fleet target，走 fleet proxy
  if (state.activeTarget) return apiUrl(path);
  return 'http://' + location.hostname + ':' + port + path;
}

function fetchDebugState() {
  fetch(signalUrl('/debug/state')).then(function(r) { return r.json(); }).then(function(d) {
    debugState.connected = true;
    debugState.paused = d.paused;
    debugState.timeScale = d.time_scale;
    debugState.bodies = d.body_names || [];
    debugState.joints = d.joint_names || [];
    debugState.muxActiveEnvs = d.mux_active_envs || [];
    debugState.forceActive = d.force_active;
    debugState.anchorActive = d.anchor_active;
    debugState.vizContacts = d.viz_contact_forces;
    debugState.vizTrajectories = d.viz_trajectories;
    updateStatusUI(d);
  }).catch(function() {
    debugState.connected = false;
    var dot = document.getElementById('debugConnDot');
    if (dot) dot.className = 'dot off';
    var txt = document.getElementById('debugConnText');
    if (txt) txt.textContent = 'disconnected';
  });
}

function updateStatusUI(d) {
  var dot = document.getElementById('debugConnDot');
  if (dot) dot.className = 'dot on';
  var txt = document.getElementById('debugConnText');
  if (txt) txt.textContent = 'connected (' + (d.plugins || []).length + ' plugins)';

  var pauseBtn = document.getElementById('dbgPauseBtn');
  if (pauseBtn) pauseBtn.textContent = d.paused ? 'Resume' : 'Pause';

  var tsLabel = document.getElementById('dbgTimeLabel');
  if (tsLabel) tsLabel.textContent = (d.time_scale * 100).toFixed(0) + '%';

  // body selector
  var sel = document.getElementById('dbgBodySel');
  if (sel && sel.options.length !== d.body_names.length) {
    sel.innerHTML = '';
    (d.body_names || []).forEach(function(name, i) {
      var opt = document.createElement('option');
      opt.value = i;
      opt.textContent = i + ': ' + name;
      sel.appendChild(opt);
    });
  }

  // force indicator
  var fi = document.getElementById('dbgForceStatus');
  if (fi) fi.textContent = d.force_active ? 'ACTIVE' : 'off';
  if (fi) fi.style.color = d.force_active ? 'var(--warn)' : 'var(--dim)';

  // anchor indicator
  var ai = document.getElementById('dbgAnchorStatus');
  if (ai) {
    var count = Object.keys(d.anchored_bodies || {}).length;
    ai.textContent = count > 0 ? count + ' anchored' : 'none';
    ai.style.color = count > 0 ? 'var(--accent)' : 'var(--dim)';
  }

  // mux indicator
  var mi = document.getElementById('dbgMuxStatus');
  if (mi) {
    var active = d.mux_active_envs || [];
    mi.textContent = active.length > 0 ? 'envs: ' + active.join(',') : 'off';
    mi.style.color = active.length > 0 ? 'var(--accent)' : 'var(--dim)';
  }

  // viz checkboxes
  var vc = document.getElementById('dbgVizContacts');
  if (vc) vc.checked = d.viz_contact_forces;
  var vt = document.getElementById('dbgVizTrajectory');
  if (vt) vt.checked = d.viz_trajectories;
}

// ── POST 命令 ────────────────────────────────────────────────────

function debugPost(path, body) {
  return fetch(signalUrl(path), {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {})
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.ok) fetchDebugState();
    return d;
  }).catch(function(e) {
    showToast('Debug command failed: ' + e, true);
  });
}

// ── Time Scale ───────────────────────────────────────────────────

export function dbgSetTimeScale() {
  var slider = document.getElementById('dbgTimeSlider');
  var val = parseFloat(slider.value) / 100;
  debugPost('/debug/timescale', {scale: val});
}

export function dbgTogglePause() {
  debugPost('/debug/pause');
}

export function dbgSingleStep() {
  debugPost('/debug/step');
}

// ── Force ────────────────────────────────────────────────────────

export function dbgApplyForce() {
  var envId = parseInt(document.getElementById('dbgEnvId').value) || 0;
  var bodyId = parseInt(document.getElementById('dbgBodySel').value) || 0;
  var fx = parseFloat(document.getElementById('dbgFx').value) || 0;
  var fy = parseFloat(document.getElementById('dbgFy').value) || 0;
  var fz = parseFloat(document.getElementById('dbgFz').value) || 0;
  var mag = parseFloat(document.getElementById('dbgForceMag').value) || 100;

  // normalize direction * magnitude
  var len = Math.sqrt(fx*fx + fy*fy + fz*fz);
  if (len < 1e-6) { fz = 1; len = 1; }
  var s = mag / len;
  debugPost('/debug/force', {
    env_id: envId, body_id: bodyId,
    force: [fx*s, fy*s, fz*s]
  }).then(function() { showToast('Force applied'); });
}

export function dbgClearForce() {
  debugPost('/debug/force/clear').then(function() { showToast('Forces cleared'); });
}

// ── MUX ──────────────────────────────────────────────────────────

export function dbgMuxSet() {
  var envId = parseInt(document.getElementById('dbgMuxEnv').value) || 0;
  var jointIdx = parseInt(document.getElementById('dbgMuxJoint').value) || 0;
  var value = parseFloat(document.getElementById('dbgMuxValue').value) || 0;
  debugPost('/debug/mux/set', {env_id: envId, joint_idx: jointIdx, value: value});
}

export function dbgMuxClear() {
  var envId = parseInt(document.getElementById('dbgMuxEnv').value) || 0;
  debugPost('/debug/mux/clear', {env_id: envId}).then(function() { showToast('MUX cleared for env ' + envId); });
}

// ── Anchor ───────────────────────────────────────────────────────

export function dbgToggleAnchor() {
  var envId = parseInt(document.getElementById('dbgEnvId').value) || 0;
  var bodyId = parseInt(document.getElementById('dbgBodySel').value) || 0;
  debugPost('/debug/anchor/toggle', {env_id: envId, body_id: bodyId});
}

// ── Viz ──────────────────────────────────────────────────────────

export function dbgSetViz() {
  var contacts = document.getElementById('dbgVizContacts').checked;
  var traj = document.getElementById('dbgVizTrajectory').checked;
  debugPost('/debug/viz', {contact_forces: contacts, trajectories: traj});
}

// ── 页面进入时刷新 ───────────────────────────────────────────────

export function onDebugPageEnter() {
  fetchDebugState();
  // 拉取 joints 列表填充 MUX joint 选择器
  fetch(signalUrl('/debug/joints')).then(function(r) { return r.json(); }).then(function(d) {
    var sel = document.getElementById('dbgMuxJoint');
    if (!sel) return;
    sel.innerHTML = '';
    (d.joints || []).forEach(function(name, i) {
      var opt = document.createElement('option');
      opt.value = i;
      opt.textContent = i + ': ' + name;
      sel.appendChild(opt);
    });
  }).catch(function() {});
}
