// reward-timeline.js — 奖励占比时间轴（Canvas2D 堆积面积图）
//
// 订阅 SignalServer 的 reward/metrics/* channel，渲染各 term 的 magnitude
// fraction 沿时间演化。IndexedDB 跨会话持久化历史（最近 10 run × 100k 样本）。
// X 轴 dropdown 支持 step / iteration / wall-clock 三选一。
//
// 设计为 Part A 的主 widget；Part B 的 pre-training 分析叠加层会在 B5 阶段
// 增强到本文件（canvas 左侧参考区 + 顶部警告 + smoke 按钮）。

import state from './state.js';
import { apiUrl, showToast } from './api.js';

const METRIC_PREFIX = 'reward/metrics/mag_frac/';
const CHANNEL_STEP = 'reward/metrics/step';
const CHANNEL_WALL = 'reward/metrics/wall_clock';
const MAX_HISTORY = 4096;      // rolling window per term
const IDB_NAME = 'myrl_reward_timeline';
const IDB_STORE = 'metrics';
const IDB_RUN_CAP = 10;        // 保留最近 N 个 runId
const IDB_SAMPLE_CAP = 100000; // 单 runId/channel 最多样本数
const PRE_REF_WIDTH = 56;      // canvas 左侧预训练参考区宽度（像素）

// ── 模块状态 ──────────────────────────────────────────────────────────
const rt = {
  mounted: false,
  canvas: null,
  ctx: null,
  legend: null,
  sse: null,
  frozen: false,
  xAxis: 'step',              // 'step' | 'iteration' | 'wall_clock'
  numStepsPerEnv: 24,
  runId: null,
  history: {},                // history[term] = {xStep, xWall, y}
  stepBuf: [],
  wallBuf: [],
  terms: [],
  discoverTimer: null,
  db: null,
  keyframes: [],              // [{term, step, weight}]
  // Part B: 预训练分析参考
  signatures: null,           // 从 /reward-signatures/<name> 拿到的完整结果
  overallWarnings: [],        // 顶部警告条
  sigRefreshTimer: null,
};

// ── 小工具 ────────────────────────────────────────────────────────────
function signalUrl(path) {
  // 与 debug-tools.js 一致：同机直连 SignalServer 端口；fleet 时走代理
  let port = 7002;
  const el = document.getElementById('cfgSignalPort');
  if (el) port = parseInt(el.value) || 7002;
  if (state.activeTarget) {
    // 期望 editor server 做反向代理 /fleet/<id>/signal/... 形式
    return '/fleet/' + state.activeTarget + '/signal' + path;
  }
  return 'http://' + location.hostname + ':' + port + path;
}

function colorFor(termIdx, alpha) {
  const base = state.rewTermColors[termIdx % state.rewTermColors.length];
  if (alpha == null || alpha >= 1) return base;
  // base 格式 #rrggbb
  const r = parseInt(base.slice(1, 3), 16);
  const g = parseInt(base.slice(3, 5), 16);
  const b = parseInt(base.slice(5, 7), 16);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}

// ── IndexedDB 封装 ────────────────────────────────────────────────────
function openDB() {
  return new Promise(function (resolve, reject) {
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = function () {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE);   // key=`${runId}/${channel}`
      }
    };
    req.onsuccess = function () { resolve(req.result); };
    req.onerror = function () { reject(req.error); };
  });
}

async function idbGet(key) {
  if (!rt.db) return null;
  return new Promise(function (resolve) {
    const tx = rt.db.transaction([IDB_STORE], 'readonly');
    const req = tx.objectStore(IDB_STORE).get(key);
    req.onsuccess = function () { resolve(req.result); };
    req.onerror = function () { resolve(null); };
  });
}

async function idbPut(key, value) {
  if (!rt.db) return;
  return new Promise(function (resolve) {
    const tx = rt.db.transaction([IDB_STORE], 'readwrite');
    tx.objectStore(IDB_STORE).put(value, key);
    tx.oncomplete = function () { resolve(); };
    tx.onerror = function () { resolve(); };
  });
}

async function idbKeys() {
  if (!rt.db) return [];
  return new Promise(function (resolve) {
    const tx = rt.db.transaction([IDB_STORE], 'readonly');
    const req = tx.objectStore(IDB_STORE).getAllKeys();
    req.onsuccess = function () { resolve(req.result || []); };
    req.onerror = function () { resolve([]); };
  });
}

async function idbDeleteKeys(keys) {
  if (!rt.db || !keys.length) return;
  const tx = rt.db.transaction([IDB_STORE], 'readwrite');
  const st = tx.objectStore(IDB_STORE);
  keys.forEach(function (k) { st.delete(k); });
  return new Promise(function (resolve) { tx.oncomplete = resolve; });
}

async function cleanupOldRuns() {
  // 保留最近 IDB_RUN_CAP 个 runId（按字典序最后为最新；runId 含时间戳后缀）
  const all = await idbKeys();
  const runIds = Array.from(new Set(all.map(function (k) {
    const idx = k.indexOf('/');
    return idx > 0 ? k.slice(0, idx) : k;
  })));
  runIds.sort();
  if (runIds.length <= IDB_RUN_CAP) return;
  const drop = runIds.slice(0, runIds.length - IDB_RUN_CAP);
  const deleteKeys = all.filter(function (k) {
    const rid = k.slice(0, k.indexOf('/'));
    return drop.indexOf(rid) >= 0;
  });
  await idbDeleteKeys(deleteKeys);
}

async function persistCurrentRun() {
  if (!rt.db || !rt.runId) return;
  for (const term of rt.terms) {
    const h = rt.history[term];
    if (!h || !h.y.length) continue;
    const key = rt.runId + '/' + term;
    // 截断到 IDB_SAMPLE_CAP
    const N = Math.min(h.y.length, IDB_SAMPLE_CAP);
    const off = h.y.length - N;
    const payload = {
      xStep: Float32Array.from(h.xStep.slice(off)),
      xWall: Float32Array.from(h.xWall.slice(off)),
      y: Float32Array.from(h.y.slice(off)),
    };
    await idbPut(key, payload);
  }
}

async function restoreLatestRun() {
  // 找最新 runId，恢复所有 term 历史到 rt.history（只读显示）
  const all = await idbKeys();
  const byRun = {};
  all.forEach(function (k) {
    const idx = k.indexOf('/');
    if (idx < 0) return;
    const rid = k.slice(0, idx);
    const ch = k.slice(idx + 1);
    (byRun[rid] || (byRun[rid] = [])).push({ key: k, channel: ch });
  });
  const rids = Object.keys(byRun).sort();
  if (!rids.length) return;
  const latest = rids[rids.length - 1];
  for (const entry of byRun[latest]) {
    const data = await idbGet(entry.key);
    if (!data) continue;
    rt.history[entry.channel] = {
      xStep: Array.from(data.xStep || []),
      xWall: Array.from(data.xWall || []),
      y: Array.from(data.y || []),
    };
    if (rt.terms.indexOf(entry.channel) < 0) rt.terms.push(entry.channel);
  }
  rt.terms.sort();
}

// ── 数据采集 ──────────────────────────────────────────────────────────
async function fetchChannels() {
  try {
    const r = await fetch(signalUrl('/channels'));
    const list = await r.json();
    return list.filter(function (c) { return c.indexOf(METRIC_PREFIX) === 0; })
      .map(function (c) { return c.slice(METRIC_PREFIX.length); });
  } catch (e) { return []; }
}

async function fetchHistoricalData(termNames) {
  // 一次性拉全部 term + step + wall_clock 历史
  const chs = termNames.map(function (t) { return METRIC_PREFIX + t; });
  chs.push(CHANNEL_STEP, CHANNEL_WALL);
  try {
    const url = signalUrl('/data?channels=' + chs.map(encodeURIComponent).join(',') + '&frames=' + MAX_HISTORY);
    const r = await fetch(url);
    const data = await r.json();
    // 对齐：各 term 的 values 长度可能不同；用 min 长度对齐
    const stepVals = (data[CHANNEL_STEP] && data[CHANNEL_STEP].values) || [];
    const wallVals = (data[CHANNEL_WALL] && data[CHANNEL_WALL].values) || [];
    for (const term of termNames) {
      const d = data[METRIC_PREFIX + term];
      if (!d) continue;
      const n = Math.min(d.values.length, stepVals.length, wallVals.length);
      const off = d.values.length - n;
      const offS = stepVals.length - n;
      const offW = wallVals.length - n;
      rt.history[term] = {
        xStep: stepVals.slice(offS, offS + n).map(Number),
        xWall: wallVals.slice(offW, offW + n).map(Number),
        y: d.values.slice(off, off + n).map(Number),
      };
    }
    if (stepVals.length) rt.stepBuf = stepVals.map(Number);
    if (wallVals.length) rt.wallBuf = wallVals.map(Number);
  } catch (e) { /* ignore */ }
}

function connectSSE() {
  if (rt.sse) { try { rt.sse.close(); } catch (e) { /* */ } }
  try {
    rt.sse = new EventSource(signalUrl('/stream?hz=10'));
  } catch (e) { return; }
  rt.sse.onmessage = function (evt) {
    if (rt.frozen) return;
    let snap;
    try { snap = JSON.parse(evt.data); } catch (e) { return; }
    delete snap._meta;
    // 先读 step / wall 本次 tick 的值
    let stepV = null, wallV = null;
    if (snap[CHANNEL_STEP]) stepV = snap[CHANNEL_STEP].mean;
    if (snap[CHANNEL_WALL]) wallV = snap[CHANNEL_WALL].mean;
    // 遍历 term
    let newTerm = false;
    for (const ch in snap) {
      if (ch.indexOf(METRIC_PREFIX) !== 0) continue;
      const term = ch.slice(METRIC_PREFIX.length);
      if (rt.terms.indexOf(term) < 0) { rt.terms.push(term); newTerm = true; }
      const h = rt.history[term] || (rt.history[term] = { xStep: [], xWall: [], y: [] });
      const v = snap[ch].mean;
      h.y.push(v);
      h.xStep.push(stepV != null ? stepV : (h.xStep.length ? h.xStep[h.xStep.length - 1] : 0));
      h.xWall.push(wallV != null ? wallV : (h.xWall.length ? h.xWall[h.xWall.length - 1] : 0));
      if (h.y.length > MAX_HISTORY) {
        h.y.shift(); h.xStep.shift(); h.xWall.shift();
      }
    }
    if (newTerm) rt.terms.sort();
    renderCanvas();
    renderLegend();
  };
  rt.sse.onerror = function () { /* 自动 retry 由浏览器处理 */ };
}

// ── Canvas 渲染 ───────────────────────────────────────────────────────
function xValueAt(h, i) {
  if (rt.xAxis === 'step') return h.xStep[i] || 0;
  if (rt.xAxis === 'wall_clock') return h.xWall[i] || 0;
  // iteration = step / num_steps_per_env
  return (h.xStep[i] || 0) / Math.max(rt.numStepsPerEnv, 1);
}

function renderCanvas() {
  if (!rt.ctx) return;
  const c = rt.canvas;
  const w = c.width = c.clientWidth * (window.devicePixelRatio || 1);
  const h = c.height = c.clientHeight * (window.devicePixelRatio || 1);
  const dpr = window.devicePixelRatio || 1;
  const ctx = rt.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.scale(dpr, dpr);
  const W = c.clientWidth, H = c.clientHeight;

  // 背景
  ctx.fillStyle = '#0a0e14';
  ctx.fillRect(0, 0, W, H);

  // 先画预训练参考区（即使没有实时数据也显示）
  const hasSig = rt.signatures && rt.signatures.terms;
  const preRefW = hasSig ? PRE_REF_WIDTH : 0;

  if (!rt.terms.length) {
    if (hasSig) renderPreReference(preRefW);
    ctx.fillStyle = '#6a7585';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(
      hasSig
        ? 'Pre-training analysis ready. Waiting for training to stream data…'
        : 'Waiting for reward/metrics/* channels…',
      preRefW + (W - preRefW) / 2, H / 2
    );
    return;
  }

  // 计算所有 term 共同的帧数（取 min，对齐）
  let nFrames = Infinity;
  for (const term of rt.terms) {
    const hist = rt.history[term];
    if (hist && hist.y.length) nFrames = Math.min(nFrames, hist.y.length);
  }
  if (!isFinite(nFrames) || nFrames < 2) {
    ctx.fillStyle = '#6a7585';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Collecting samples…', W / 2, H / 2);
    return;
  }

  // X 值范围（用第一个 term 的 x 序列）
  const firstHist = rt.history[rt.terms[0]];
  const offset = firstHist.y.length - nFrames;
  let xMin = xValueAt(firstHist, offset);
  let xMax = xValueAt(firstHist, firstHist.y.length - 1);
  if (xMax - xMin < 1e-6) xMax = xMin + 1;

  // 左右留白（左侧为 Y 轴 label；如果有预训练参考则先画 preRef 再让实时图向右偏移）
  const padL = 44 + preRefW, padR = 8, padT = 8, padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // 预训练参考区
  if (hasSig) renderPreReference(preRefW);

  // Y 轴 gridlines (0, 25, 50, 75, 100%)
  ctx.strokeStyle = '#1a2332';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + plotH * (i / 4);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    ctx.fillStyle = '#4a5562';
    ctx.font = '9px system-ui';
    ctx.textAlign = 'right';
    ctx.fillText((100 - i * 25) + '%', padL - 4, y + 3);
  }

  // 堆积面积图：bottom-up 累加
  const xPix = function (i) {
    const x = xValueAt(firstHist, offset + i);
    return padL + plotW * ((x - xMin) / (xMax - xMin));
  };
  // 每个 frame 的底 Y 从 100% 开始向上累加
  const baselines = new Float32Array(nFrames);   // [0..1]
  for (let i = 0; i < nFrames; i++) baselines[i] = 1.0;

  for (let ti = 0; ti < rt.terms.length; ti++) {
    const term = rt.terms[ti];
    const hist = rt.history[term];
    if (!hist || hist.y.length < nFrames) continue;
    const yOff = hist.y.length - nFrames;
    ctx.fillStyle = colorFor(ti, 0.9);
    ctx.beginPath();
    // 底边（当前 baseline）→ 右→左 反向
    for (let i = 0; i < nFrames; i++) {
      const y = padT + plotH * (1 - baselines[i]);
      if (i === 0) ctx.moveTo(xPix(i), y);
      else ctx.lineTo(xPix(i), y);
    }
    // 顶边（baseline - y[i]）
    for (let i = nFrames - 1; i >= 0; i--) {
      const v = hist.y[yOff + i] || 0;
      baselines[i] -= v;
      if (baselines[i] < 0) baselines[i] = 0;
      const y = padT + plotH * (1 - baselines[i]);
      ctx.lineTo(xPix(i), y);
    }
    ctx.closePath();
    ctx.fill();
  }

  // WeightSchedule keyframes (在 step / iteration 轴上)
  if (rt.keyframes.length && (rt.xAxis === 'step' || rt.xAxis === 'iteration')) {
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.setLineDash([4, 4]);
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.font = '9px system-ui';
    ctx.textAlign = 'left';
    for (const kf of rt.keyframes) {
      let xVal = kf.step;
      if (rt.xAxis === 'iteration') xVal = kf.step / Math.max(rt.numStepsPerEnv, 1);
      if (xVal < xMin || xVal > xMax) continue;
      const px = padL + plotW * ((xVal - xMin) / (xMax - xMin));
      ctx.beginPath();
      ctx.moveTo(px, padT);
      ctx.lineTo(px, padT + plotH);
      ctx.stroke();
      ctx.fillText(kf.term + '=' + kf.weight.toFixed(2), px + 2, padT + 10);
    }
    ctx.setLineDash([]);
  }

  // X 轴 label
  ctx.fillStyle = '#6a7585';
  ctx.font = '9px system-ui';
  ctx.textAlign = 'center';
  const xMid = padL + plotW / 2;
  const lbl = rt.xAxis === 'wall_clock' ? 'wall-clock (s)'
    : rt.xAxis === 'iteration' ? 'iteration'
    : 'step';
  ctx.fillText(lbl + '  [' + formatX(xMin) + ' → ' + formatX(xMax) + ']', xMid, H - 6);
}

// ── Part B：预训练参考区渲染 ──────────────────────────────────────────
function renderPreReference(width) {
  if (!rt.signatures || !rt.ctx) return;
  const ctx = rt.ctx;
  const H = rt.canvas.clientHeight;
  const padT = 8, padB = 22;
  const plotH = H - padT - padB;
  // 参考区：x=[0..width]
  // 顶部留给标签
  const labelH = 14;
  const bandX = 6;
  const bandW = width - 10;
  const bandTop = padT + labelH;
  const bandBottom = H - padB;
  const bandH = bandBottom - bandTop;

  // 标题
  ctx.fillStyle = '#7a8694';
  ctx.font = '9px system-ui';
  ctx.textAlign = 'left';
  ctx.fillText('analytical', 4, padT + 10);

  // 各 term 占比（以 analytical mag_frac 推算：weighted_max / Σ(weighted_max ok)）
  const terms = rt.signatures.terms || {};
  const ok = [];
  let sum = 0;
  for (const name in terms) {
    const e = terms[name];
    if (e.status === 'ok' && typeof e.weighted_max === 'number' && e.weighted_max > 0) {
      ok.push({ name: name, mag: e.weighted_max });
      sum += e.weighted_max;
    }
  }
  if (sum <= 0) {
    ctx.fillStyle = '#4a5562';
    ctx.font = '9px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('no static', bandX + bandW / 2, bandTop + bandH / 2);
    ctx.fillText('estimate', bandX + bandW / 2, bandTop + bandH / 2 + 10);
    return;
  }
  // 排序以便按 rt.terms 顺序上色（实时图和参考图颜色对齐）
  const termOrder = rt.terms.length ? rt.terms : ok.map(o => o.name).sort();
  let y = bandTop;
  for (let i = 0; i < termOrder.length; i++) {
    const tn = termOrder[i];
    const e = terms[tn];
    if (!e || e.status !== 'ok' || !e.weighted_max) continue;
    const frac = e.weighted_max / sum;
    const segH = frac * bandH;
    ctx.fillStyle = colorFor(i, 0.6);  // 半透明与实时图区分
    ctx.fillRect(bandX, y, bandW, segH);
    y += segH;
  }
  // 边框
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.strokeRect(bandX, bandTop, bandW, bandH);
  // 分隔虚线（参考区 → 实时区）
  ctx.strokeStyle = 'rgba(255,255,255,0.3)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(width, padT);
  ctx.lineTo(width, H - padB);
  ctx.stroke();
  ctx.setLineDash([]);
  // "requires_runtime" 提示小标注
  const reqRt = Object.entries(terms).filter(
    ([_, e]) => e.status === 'requires_runtime'
  ).length;
  if (reqRt > 0) {
    ctx.fillStyle = '#f0ad4e';
    ctx.font = '8px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText('+' + reqRt + ' runtime', 4, H - padB - 2);
  }
}

function formatX(v) {
  if (v >= 10000) return (v / 1000).toFixed(1) + 'k';
  if (v >= 1000) return (v / 1000).toFixed(2) + 'k';
  return v.toFixed(0);
}

function renderLegend() {
  if (!rt.legend) return;
  const el = rt.legend;
  el.innerHTML = '';
  for (let ti = 0; ti < rt.terms.length; ti++) {
    const term = rt.terms[ti];
    const hist = rt.history[term];
    const lastV = hist && hist.y.length ? hist.y[hist.y.length - 1] : 0;
    const row = document.createElement('div');
    row.className = 'rt-legend-row';
    row.innerHTML =
      '<span class="rt-dot" style="background:' + colorFor(ti, 1) + '"></span>' +
      '<span class="rt-name">' + term + '</span>' +
      '<span class="rt-pct">' + (lastV * 100).toFixed(1) + '%</span>';
    el.appendChild(row);
  }
}

// ── 事件 / 初始化 ──────────────────────────────────────────────────────
function bindToolbar() {
  const xSel = document.getElementById('rtXAxis');
  if (xSel) xSel.onchange = function () { rt.xAxis = this.value; renderCanvas(); };
  const freeze = document.getElementById('rtFreeze');
  if (freeze) freeze.onclick = function () {
    rt.frozen = !rt.frozen;
    this.textContent = rt.frozen ? 'Resume' : 'Freeze';
    this.classList.toggle('active', rt.frozen);
  };
  const toggle = document.getElementById('rtToggle');
  if (toggle) toggle.onclick = function () {
    const body = document.getElementById('rtCanvasContainer');
    const vis = body.style.display !== 'none';
    body.style.display = vis ? 'none' : '';
    toggle.textContent = (vis ? '▶' : '▼') + ' Reward Timeline';
    if (!vis) {
      requestAnimationFrame(renderCanvas);
    }
  };
  const smoke = document.getElementById('rtSmokeBtn');
  if (smoke) smoke.onclick = function () { runSmokeProbe(); };
}

// ── Part B：预训练分析获取 + 警告渲染 ──────────────────────────────────
async function fetchSignatures() {
  const name = state.rewPipelineName;
  if (!name) return;
  const robot = state.robotName || '';
  try {
    const url = apiUrl('/reward-signatures/' + encodeURIComponent(name) +
      (robot ? '?robot=' + encodeURIComponent(robot) : ''));
    const r = await fetch(url);
    const data = await r.json();
    if (data && !data.error) {
      rt.signatures = data;
      rt.overallWarnings = (data.overall && data.overall.warnings) || [];
      renderWarningBar();
      // smoke button 在有 pipeline 时显示
      const sm = document.getElementById('rtSmokeBtn');
      if (sm) sm.style.display = '';
    }
  } catch (e) { /* silent */ }
}

function renderWarningBar() {
  const el = document.getElementById('rtWarnBadge');
  if (!el) return;
  if (!rt.overallWarnings.length) {
    el.style.display = 'none';
    el.textContent = '';
    el.classList.remove('severe');
    return;
  }
  // 取最严重的警告显示
  const severities = { severe: 3, warn: 2, info: 1 };
  let top = rt.overallWarnings[0];
  for (const w of rt.overallWarnings) {
    if ((severities[w.severity] || 0) > (severities[top.severity] || 0)) top = w;
  }
  el.textContent = '⚠ ' + top.message;
  el.title = rt.overallWarnings.map(w => `[${w.severity}] ${w.message}`).join('\n');
  el.style.display = '';
  el.classList.toggle('severe', top.severity === 'severe');
}

async function runSmokeProbe() {
  const name = state.rewPipelineName;
  if (!name) { showToast('No reward pipeline selected', true); return; }
  const btn = document.getElementById('rtSmokeBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Running…'; }
  try {
    const r = await fetch(apiUrl('/reward-smoke-probe'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pipeline: name, num_steps: 200 }),
    });
    const d = await r.json();
    if (d.ok) {
      showToast('Sim smoke running — metrics will appear on timeline');
    } else {
      showToast('Smoke probe failed: ' + (d.error || 'unknown'), true);
    }
  } catch (e) {
    showToast('Smoke probe error: ' + e, true);
  } finally {
    setTimeout(function () {
      if (btn) { btn.disabled = false; btn.textContent = 'Sim smoke ▶'; }
    }, 2000);
  }
}

function syncNumStepsPerEnv() {
  // 从 state.algoPipeline 读取（algo editor 会填充）
  try {
    const ap = state.algoPipeline;
    if (ap && ap.runner && ap.runner.num_steps_per_env) {
      rt.numStepsPerEnv = parseInt(ap.runner.num_steps_per_env) || 24;
    }
  } catch (e) { /* */ }
}

function extractKeyframes() {
  // 从 state.rewPipeline.transforms 找 weight_schedule 节
  rt.keyframes = [];
  try {
    const tfs = (state.rewPipeline && state.rewPipeline.transforms) || [];
    for (const tf of tfs) {
      if (tf.name !== 'weight_schedule') continue;
      const sch = tf.params && tf.params.schedules;
      if (!sch) continue;
      for (const term in sch) {
        for (const kf of sch[term]) {
          rt.keyframes.push({ term: term, step: kf[0], weight: kf[1] });
        }
      }
    }
  } catch (e) { /* */ }
}

function makeRunId() {
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const exp = (state.selected && state.selected.name) || 'default';
  return exp + '_' + ts;
}

export async function initRewardTimeline() {
  if (rt.mounted) return;
  rt.mounted = true;
  rt.canvas = document.getElementById('rtCanvas');
  rt.legend = document.getElementById('rtLegend');
  if (!rt.canvas) return;        // 容器未注入到 DOM 时 noop
  rt.ctx = rt.canvas.getContext('2d');

  // IndexedDB
  try {
    rt.db = await openDB();
    await cleanupOldRuns();
    await restoreLatestRun();
  } catch (e) { rt.db = null; }

  bindToolbar();
  rt.runId = makeRunId();
  syncNumStepsPerEnv();
  extractKeyframes();

  // 首次拉 channels + 历史
  const found = await fetchChannels();
  if (found.length) {
    rt.terms = Array.from(new Set(rt.terms.concat(found))).sort();
    await fetchHistoricalData(rt.terms);
  }

  // Part B：预训练分析（若 rewPipelineName 已知）
  await fetchSignatures();

  connectSSE();
  renderCanvas();
  renderLegend();

  // 每 5 秒周期性持久化 + 重试 discover
  rt.discoverTimer = setInterval(async function () {
    syncNumStepsPerEnv();
    extractKeyframes();
    const ch = await fetchChannels();
    let changed = false;
    for (const t of ch) {
      if (rt.terms.indexOf(t) < 0) { rt.terms.push(t); changed = true; }
    }
    if (changed) { rt.terms.sort(); renderLegend(); }
    persistCurrentRun().catch(function () { /* */ });
  }, 5000);

  // 每 15 秒刷新一次 signature（pipeline 名变化或权重调整后可看到更新）
  rt.sigRefreshTimer = setInterval(async function () {
    await fetchSignatures();
    renderCanvas();
  }, 15000);

  window.addEventListener('beforeunload', function () {
    persistCurrentRun().catch(function () { /* */ });
  });

  window.addEventListener('resize', function () {
    requestAnimationFrame(renderCanvas);
  });
}

export function refreshRewardTimelineKeyframes() {
  // 供外部（reward.js Save 后）主动通知刷新
  extractKeyframes();
  // Reward 配置变了 → 重新估计预训练分布
  fetchSignatures().then(function () {
    renderCanvas();
  }).catch(function () { renderCanvas(); });
}
