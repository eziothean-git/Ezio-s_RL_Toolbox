// robot-viewer.js — 机器人 3D viewer 主模块
// 编排：mesh 加载 → WebGPU 渲染 → link 选择 → sensor 可视化 → sidebar 交互
import state from './state.js';
import { apiUrl, esc, showToast } from './api.js';
import { parseBinarySTL } from './webgpu/stl-parser.js';
import { buildTransformMap, buildMeshMap, mat4Perspective, mat4LookAt, mat4Multiply } from './webgpu/urdf-transform.js';
import { GPUPipeline } from './webgpu/gpu-pipeline.js';
import { OrbitCamera } from './webgpu/orbit-camera.js';
import { LinkPicker } from './webgpu/link-picker.js';
import { makeSensorGizmo } from './webgpu/sensor-gizmos.js';

var gpu = null;
var camera = null;
var picker = null;
var linkMeshes = [];  // [{name, vbuf, vertexCount, linkBuf, linkBG, linkId}]
var linkNameToIdx = {};
var robotData = null;
var sensorManifest = null;
var animFrameId = null;
var initialized = false;

// ── 颜色方案 ──
var LINK_COLORS = {
  _default:  [0.45, 0.50, 0.55, 1.0],
  pelvis:    [0.50, 0.55, 0.60, 1.0],
  torso:     [0.50, 0.55, 0.60, 1.0],
  hip:       [0.40, 0.45, 0.55, 1.0],
  knee:      [0.35, 0.42, 0.52, 1.0],
  ankle:     [0.32, 0.40, 0.50, 1.0],
  shoulder:  [0.42, 0.48, 0.56, 1.0],
  elbow:     [0.38, 0.44, 0.52, 1.0],
  wrist:     [0.35, 0.42, 0.50, 1.0],
  hand:      [0.30, 0.38, 0.48, 1.0],
};

function getLinkColor(name) {
  for (var key in LINK_COLORS) {
    if (key !== '_default' && name.indexOf(key) >= 0) return LINK_COLORS[key];
  }
  return LINK_COLORS._default;
}

// ── Public API ──

export async function renderRobotViewer(cfg) {
  var panel = document.getElementById('robotSensorPanel');
  if (!panel) return;

  var robotRef = (cfg.assets && cfg.assets.robot_model) || {};
  var robotName = robotRef.name || '';
  if (!robotName) {
    panel.innerHTML = '<div class="todo-hint">No robot_model in experiment</div>';
    return;
  }

  state.robotName = robotName;
  panel.innerHTML = '<div class="todo-hint">Loading robot...</div>';

  try {
    // 并行请求 link 树和 sensor manifest
    var [linksResp, sensorsResp] = await Promise.all([
      fetch(apiUrl('/robot/' + robotName + '/links')).then(function(r) { return r.json(); }),
      fetch(apiUrl('/robot/' + robotName + '/sensors')).then(function(r) { return r.json(); }),
    ]);

    if (linksResp.error) {
      panel.innerHTML = '<div class="todo-hint">' + esc(linksResp.error) + '</div>';
      return;
    }

    robotData = linksResp;
    sensorManifest = sensorsResp;
    state.robotSensors = sensorsResp.sensors || [];
    state.robotLinks = linksResp;
    state.sensorDirty = false;

    // 渲染 UI（sidebar 不依赖 WebGPU）
    renderSensorPanel();
    updateSensorCount();

    // 初始化 WebGPU（失败不影响 sidebar 功能）
    try {
      if (!navigator.gpu) throw new Error('WebGPU API not available');
      await initWebGPU();
      await loadMeshes();
      startRenderLoop();
    } catch (gpuErr) {
      console.warn('[robot-viewer] WebGPU init failed:', gpuErr);
      var wrap = document.getElementById('robotCanvasWrap');
      if (wrap) wrap.innerHTML =
        '<div class="todo-hint" style="padding:40px;text-align:center">' +
        'WebGPU: ' + esc(String(gpuErr.message || gpuErr)) +
        '<br><br><span style="color:var(--dim);font-size:10px">Try: chrome://flags → #enable-unsafe-webgpu → Enabled<br>' +
        'Sensor editing works without 3D preview.</span></div>';
    }

  } catch (e) {
    console.error('robot-viewer:', e);
    // 保留 panel 不清空（sidebar 已渲染）
    showToast('Robot viewer error: ' + e.message, true);
  }
}

// ── WebGPU Init ──

async function initWebGPU() {
  if (initialized) return;
  var canvas = document.getElementById('robotCanvas');
  if (!canvas) return;

  gpu = new GPUPipeline();
  await gpu.init(canvas);

  camera = new OrbitCamera();
  camera.attach(canvas);
  picker = new LinkPicker(gpu);

  // 点击选择 link
  canvas.addEventListener('click', async function(e) {
    var rect = canvas.getBoundingClientRect();
    var x = e.clientX - rect.left, y = e.clientY - rect.top;
    var id = await picker.pick(x, y, function(pass) {
      drawLinks(pass);
    });
    selectLink(id);
  });

  // ResizeObserver
  if (window.ResizeObserver) {
    new ResizeObserver(function() {
      var c = canvas.parentElement;
      canvas.width = c.clientWidth * (window.devicePixelRatio || 1);
      canvas.height = c.clientHeight * (window.devicePixelRatio || 1);
      canvas.style.width = c.clientWidth + 'px';
      canvas.style.height = c.clientHeight + 'px';
      gpu.resize(c.clientWidth, c.clientHeight);
      camera.dirty = true;
    }).observe(canvas.parentElement);
  }

  initialized = true;
}

// ── Mesh Loading ──

async function loadMeshes() {
  if (!robotData || !gpu) return;
  linkMeshes = [];
  linkNameToIdx = {};

  var transforms = buildTransformMap(robotData);
  var meshMap = buildMeshMap(robotData);
  var loadOrder = robotData.links
    .filter(function(l) { return l.has_mesh; })
    .map(function(l) { return l.name; });

  // 计算 bounding box
  var bmin = [Infinity, Infinity, Infinity], bmax = [-Infinity, -Infinity, -Infinity];

  var linkId = 1; // 0 = background
  for (var i = 0; i < loadOrder.length; i++) {
    var name = loadOrder[i];
    var meshFile = meshMap[name];
    if (!meshFile) continue;

    try {
      var resp = await fetch(apiUrl('/robot/' + state.robotName + '/meshes/' + meshFile));
      if (!resp.ok) continue;
      var buf = await resp.arrayBuffer();
      var parsed = parseBinarySTL(buf);

      var vbuf = gpu.createVertexBuffer(parsed.vertices);

      // visual origin 变换
      var tfKey = name + '/__visual__';
      var tf = transforms[tfKey] || transforms[name];
      if (!tf) tf = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);

      var color = getLinkColor(name);
      var linkBuf = gpu.createLinkBuffer(tf, color, linkId, false);
      var linkBG = gpu.createLinkBindGroup(linkBuf);

      // bounding box (approximate from transform position)
      var px = tf[12], py = tf[13], pz = tf[14];
      bmin[0] = Math.min(bmin[0], px - 0.3);
      bmin[1] = Math.min(bmin[1], py - 0.3);
      bmin[2] = Math.min(bmin[2], pz - 0.3);
      bmax[0] = Math.max(bmax[0], px + 0.3);
      bmax[1] = Math.max(bmax[1], py + 0.3);
      bmax[2] = Math.max(bmax[2], pz + 0.3);

      linkNameToIdx[name] = linkMeshes.length;
      linkMeshes.push({
        name: name, vbuf: vbuf, vertexCount: parsed.vertexCount,
        linkBuf: linkBuf, linkBG: linkBG, linkId: linkId,
        transform: tf, color: color,
      });
      linkId++;
    } catch (e) {
      console.warn('mesh load failed:', name, e);
    }
  }

  if (bmin[0] < Infinity) {
    camera.fitBounds(bmin, bmax);
  }
}

// ── Render Loop ──

function startRenderLoop() {
  if (animFrameId) return;
  function loop() {
    animFrameId = requestAnimationFrame(loop);
    if (!camera.dirty) return;
    camera.dirty = false;
    render();
  }
  loop();
}

function render() {
  if (!gpu || !gpu.device || !linkMeshes.length) return;

  var canvas = document.getElementById('robotCanvas');
  if (!canvas) return;
  var w = canvas.clientWidth, h = canvas.clientHeight;
  if (w === 0 || h === 0) return;

  // 更新 frame uniforms
  var proj = mat4Perspective(camera.fovY, w / h, camera.near, camera.far);
  var view = mat4LookAt(camera.eye, camera.target, [0, 0, 1]);
  var vp = mat4Multiply(proj, view);
  gpu.updateFrameUniforms(vp, camera.eye);

  // Main render pass
  var encoder = gpu.device.createCommandEncoder();
  var pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: gpu.context.getCurrentTexture().createView(),
      clearValue: { r: 0.08, g: 0.09, b: 0.12, a: 1 },
      loadOp: 'clear', storeOp: 'store',
    }],
    depthStencilAttachment: {
      view: gpu.depthTexture.createView(),
      depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
    },
  });
  pass.setPipeline(gpu.mainPipeline);
  pass.setBindGroup(0, gpu.frameBindGroup);
  drawLinks(pass);
  pass.end();
  gpu.device.queue.submit([encoder.finish()]);
}

function drawLinks(pass) {
  for (var i = 0; i < linkMeshes.length; i++) {
    var m = linkMeshes[i];
    pass.setBindGroup(1, m.linkBG);
    pass.setVertexBuffer(0, m.vbuf);
    pass.draw(m.vertexCount);
  }
}

// ── Link Selection ──

function selectLink(linkId) {
  var found = null;
  for (var i = 0; i < linkMeshes.length; i++) {
    var m = linkMeshes[i];
    var isSelected = (linkId !== null && m.linkId === linkId);
    if (isSelected) found = m.name;
    // 更新 selected 状态
    var color = isSelected ? [0.2, 0.7, 1.0, 1.0] : m.color;
    m.linkBuf = gpu.createLinkBuffer(m.transform, color, m.linkId, isSelected);
    m.linkBG = gpu.createLinkBindGroup(m.linkBuf);
  }
  state.selectedLink = found;
  camera.dirty = true;

  // 更新 sidebar 高亮
  document.querySelectorAll('.link-tree-item').forEach(function(el) {
    el.classList.toggle('active', el.dataset.link === found);
  });
  renderSensorDetail();
}

export function selectLinkByName(name) {
  var idx = linkNameToIdx[name];
  if (idx != null) {
    selectLink(linkMeshes[idx].linkId);
  }
}

// ── Sensor Sidebar ──

function renderSensorPanel() {
  var container = document.getElementById('robotSensorPanel');
  if (!container) return;
  container.innerHTML = '';

  // 3D viewer + sidebar 布局
  container.innerHTML =
    '<div class="robot-viewer-layout">' +
      '<div class="robot-canvas-wrap" id="robotCanvasWrap">' +
        '<canvas id="robotCanvas"></canvas>' +
      '</div>' +
      '<div class="robot-sensor-sidebar">' +
        '<div class="sensor-link-tree" id="sensorLinkTree"></div>' +
        '<div id="sensorDetail"></div>' +
        '<button class="btn sm" onclick="showAddSensorMenu(event)">+ Add Sensor</button>' +
      '</div>' +
    '</div>';

  renderLinkTree();
}

function renderLinkTree() {
  var el = document.getElementById('sensorLinkTree');
  if (!el || !robotData) return;
  el.innerHTML = '';

  var tree = robotData.tree || {};
  var sensors = state.robotSensors || [];
  var sensorMap = {};
  sensors.forEach(function(s) {
    if (s.mount_link) {
      sensorMap[s.mount_link] = sensorMap[s.mount_link] || [];
      sensorMap[s.mount_link].push(s);
    }
  });

  var ICONS = {imu: '\u{1F535}', depth_camera: '\u{1F4F7}', height_scanner: '\u{1F4CF}',
               force_sensor: '\u{1F4A5}', contact: '\u{1F4A5}'};

  function buildNode(linkName, depth) {
    var item = document.createElement('div');
    item.className = 'link-tree-item';
    item.dataset.link = linkName;
    item.style.paddingLeft = (8 + depth * 12) + 'px';

    var label = linkName.replace(/_link$/, '').replace(/_/g, ' ');
    var html = '<span class="link-name">' + esc(label) + '</span>';

    // 传感器图标
    (sensorMap[linkName] || []).forEach(function(s) {
      html += '<span class="sensor-badge" title="' + esc(s.name) + '">' +
        (ICONS[s.type] || '\u2699') + '</span>';
    });

    item.innerHTML = html;
    item.onclick = function(e) {
      e.stopPropagation();
      selectLinkByName(linkName);
    };
    el.appendChild(item);

    // 递归子节点
    (tree[linkName] || []).forEach(function(child) {
      buildNode(child.child, depth + 1);
    });
  }

  var root = robotData.root || 'pelvis';
  buildNode(root, 0);
}

function renderSensorDetail() {
  var el = document.getElementById('sensorDetail');
  if (!el) return;
  var link = state.selectedLink;
  if (!link) { el.innerHTML = ''; return; }

  var sensors = (state.robotSensors || []).filter(function(s) {
    return s.mount_link === link;
  });

  if (!sensors.length) {
    el.innerHTML = '<div class="sensor-detail-empty">No sensors on ' + esc(link) + '</div>';
    return;
  }

  var html = '';
  sensors.forEach(function(s, idx) {
    html += '<div class="sensor-detail-card">';
    html += '<div class="sensor-detail-header">' +
      '<strong>' + esc(s.name) + '</strong>' +
      '<span class="sensor-type-badge">' + esc(s.type) + '</span>' +
      '<button class="btn sm danger" onclick="removeSensor(\'' + esc(s.name) + '\')">&times;</button>' +
    '</div>';
    // config 字段
    var cfg = s.config || {};
    Object.keys(cfg).forEach(function(k) {
      html += '<div class="sensor-config-row">' +
        '<label>' + esc(k) + '</label>' +
        '<input value="' + esc(String(cfg[k])) + '" data-sensor="' + esc(s.name) + '" data-key="' + esc(k) + '" onchange="onSensorConfigChange(this)">' +
      '</div>';
    });
    html += '</div>';
  });
  el.innerHTML = html;
}

function updateSensorCount() {
  var el = document.getElementById('sensorCount');
  if (el) el.textContent = (state.robotSensors || []).length;
  var dirty = document.getElementById('sensorDirtyBadge');
  var saveBtn = document.getElementById('sensorSaveBtn');
  if (dirty) dirty.style.display = state.sensorDirty ? '' : 'none';
  if (saveBtn) saveBtn.style.display = state.sensorDirty ? '' : 'none';
}

// ── Sensor CRUD ──

export function showAddSensorMenu(e) {
  var existing = document.querySelector('.sensor-add-menu');
  if (existing) { existing.remove(); return; }

  var link = state.selectedLink;
  if (!link) { showToast('Select a link first', true); return; }

  var types = [
    {type: 'imu', label: 'IMU'},
    {type: 'depth_camera', label: 'Depth Camera'},
    {type: 'height_scanner', label: 'Height Scanner'},
    {type: 'force_sensor', label: 'Force Sensor'},
    {type: 'contact', label: 'Contact Sensor'},
  ];

  var menu = document.createElement('div');
  menu.className = 'sensor-add-menu';
  types.forEach(function(t) {
    var item = document.createElement('div');
    item.className = 'menu-item';
    item.textContent = t.label;
    item.onclick = function() {
      addSensor(t.type, link);
      menu.remove();
    };
    menu.appendChild(item);
  });

  var btn = e.target;
  var rect = btn.getBoundingClientRect();
  var sidebar = btn.parentElement;
  var sidebarRect = sidebar.getBoundingClientRect();
  menu.style.position = 'absolute';
  menu.style.left = (rect.left - sidebarRect.left) + 'px';
  menu.style.bottom = (sidebarRect.bottom - rect.top + 4) + 'px';
  sidebar.style.position = 'relative';
  sidebar.appendChild(menu);

  setTimeout(function() {
    document.addEventListener('click', function close(ev) {
      if (!menu.contains(ev.target)) { menu.remove(); document.removeEventListener('click', close); }
    });
  }, 0);
}

function addSensor(type, link) {
  if (!state.robotSensors) state.robotSensors = [];
  var name = type + '_' + link.replace(/_link$/, '');
  var n = 0;
  while (state.robotSensors.some(function(s) { return s.name === name; })) {
    n++; name = type + '_' + link.replace(/_link$/, '') + '_' + n;
  }
  var sensor = { name: name, type: type, mount_link: link, config: {} };
  // 默认 config
  if (type === 'depth_camera') sensor.config = {width: 64, height: 36, fov_deg: 87, max_range: 5};
  if (type === 'height_scanner') sensor.config = {size: [0.3, 0.2], resolution: 0.05, max_range: 1.5};
  if (type === 'imu') sensor.config = {update_rate: 200};
  if (type === 'contact') sensor.config = {history_length: 3, track_air_time: true};

  state.robotSensors.push(sensor);
  state.sensorDirty = true;
  updateSensorCount();
  renderLinkTree();
  renderSensorDetail();
  showToast('Added ' + type + ' on ' + link);
}

export function removeSensor(name) {
  state.robotSensors = (state.robotSensors || []).filter(function(s) { return s.name !== name; });
  state.sensorDirty = true;
  updateSensorCount();
  renderLinkTree();
  renderSensorDetail();
}

export function onSensorConfigChange(input) {
  var sensorName = input.dataset.sensor;
  var key = input.dataset.key;
  var val = input.value;
  var sensor = (state.robotSensors || []).find(function(s) { return s.name === sensorName; });
  if (!sensor) return;
  // 尝试 JSON 解析
  try { val = JSON.parse(val); } catch(_) {
    var num = Number(val);
    if (!isNaN(num) && String(num) === val) val = num;
  }
  if (!sensor.config) sensor.config = {};
  sensor.config[key] = val;
  state.sensorDirty = true;
  updateSensorCount();
}

export async function saveSensorManifest() {
  if (!state.robotName) return;
  var payload = {
    schema: 'sensor_manifest_v1',
    name: state.robotName + '_sensors',
    robot_model: state.robotName,
    sensors: state.robotSensors || [],
  };
  try {
    var resp = await fetch(apiUrl('/robot/' + state.robotName + '/sensors'), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    var d = await resp.json();
    if (d.ok) {
      state.sensorDirty = false;
      updateSensorCount();
      showToast('Sensor manifest saved');
    } else {
      showToast('Save failed: ' + (d.error || '?'), true);
    }
  } catch (e) {
    showToast('Save error: ' + e, true);
  }
}

export function robotResetView() {
  if (camera && linkMeshes.length) {
    camera.fitBounds([-0.5, -0.5, -0.1], [0.5, 0.5, 1.2]);
  }
}
