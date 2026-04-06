// obs-graph.js — OBS Pipeline 图编辑器（增强版）
// 新增：缩放平移、拖拽连线、浮动添加菜单、维度标注、改进布局、ResizeObserver
import state from './state.js';
import { apiUrl, esc, showToast } from './api.js';

var OBS_COLORS = {obs:'#3498db', modifier:'#f39c12', encoder:'#9b59b6', group:'#2ecc71', transform:'#e74c3c'};
var OBS_NODE_W = 150, OBS_NODE_H = 54;
var PORT_R = 5;     // 端口圆半径
var PORT_HIT = 10;  // 端口点击判定半径

// ── 对外接口 ──

export function renderObsPipeline(cfg) {
  var obsRef = (cfg.assets && cfg.assets.obs_pipeline) || {};
  state.obsPipelineName = obsRef.name || '';
  if (!state.obsPipelineName) {
    document.getElementById('obsGraphContainer').style.display = 'none';
    document.getElementById('obsFallback').style.display = '';
    document.getElementById('obsFallback').innerHTML = '<div class="todo-hint">No obs pipeline in this experiment</div>';
    return;
  }
  document.getElementById('obsFallback').style.display = 'none';
  document.getElementById('obsGraphContainer').style.display = '';

  fetch(apiUrl('/pipeline/obs/' + state.obsPipelineName)).then(function(r) { return r.json(); }).then(function(data) {
    if (data.error) { document.getElementById('obsFallback').style.display=''; document.getElementById('obsFallback').innerHTML='<div class="todo-hint">'+esc(data.error)+'</div>'; return; }
    if (data.blocks) {
      state.obsBlocks = {};
      data.blocks.forEach(function(b, i) {
        var bid = typeof b === 'object' ? (b.id || 'b'+i) : String(b);
        var block = typeof b === 'object' ? b : {id:bid};
        block.id = bid;
        block._x = 0; block._y = 0;
        state.obsBlocks[bid] = block;
      });
    } else {
      // v1 转换
      state.obsBlocks = {};
      var meta = ['name','version','description'];
      var gy = 0;
      Object.keys(data).forEach(function(groupName) {
        if (meta.indexOf(groupName) >= 0 || typeof data[groupName] !== 'object') return;
        var gid = groupName + '_group';
        state.obsBlocks[gid] = {id:gid, type:'group', kind:groupName, _x:500, _y:gy*70};
        var ti = 0;
        Object.keys(data[groupName]).forEach(function(termName) {
          state.obsBlocks[termName] = {id:termName, type:'obs', kind:'mdp', func:data[groupName][termName].func, outputs:[gid], _x:50, _y:gy*70+ti*60};
          ti++;
        });
        gy += ti + 1;
      });
    }
    obsAutoLayout();
    state.obsDirty = false;
    state.obsPan = {x: 20, y: 20};
    state.obsZoom = 1.0;
    obsUpdateUI();
    obsDrawGraph();
    obsInitCanvas();
  }).catch(function(e) { document.getElementById('obsFallback').innerHTML='<div class="todo-hint">Error: '+e+'</div>'; document.getElementById('obsFallback').style.display=''; });
}

// ── 布局 ──

function obsAutoLayout() {
  if (!state.obsBlocks) return;
  var blocks = Object.values(state.obsBlocks);
  // 拓扑排序确定层级
  var layers = assignLayers(blocks);
  var colGap = OBS_NODE_W + 40;
  var rowGap = OBS_NODE_H + 16;

  layers.forEach(function(layer, ci) {
    layer.forEach(function(b, ri) {
      b._x = ci * colGap;
      b._y = ri * rowGap;
    });
  });
}

function assignLayers(blocks) {
  // 按类型分层：obs(0) → modifier(1) → encoder(2) → group(3)
  var typeOrder = {obs: 0, modifier: 1, encoder: 2, group: 3, transform: 1};
  var layers = [[], [], [], []];
  blocks.forEach(function(b) {
    var layer = typeOrder[b.type] != null ? typeOrder[b.type] : 0;
    layers[layer].push(b);
  });
  // 层内按连接目标排序减少交叉
  layers.forEach(function(layer) {
    layer.sort(function(a, b) {
      var aOut = (a.outputs || []).join(',');
      var bOut = (b.outputs || []).join(',');
      return aOut < bOut ? -1 : aOut > bOut ? 1 : 0;
    });
  });
  return layers.filter(function(l) { return l.length > 0; });
}

// ── UI 更新 ──

function obsUpdateUI() {
  if (!state.obsBlocks) return;
  document.getElementById('obsBlockCount').textContent = Object.keys(state.obsBlocks).length;
  document.getElementById('obsDirtyBadge').style.display = state.obsDirty ? '' : 'none';
  document.getElementById('obsSaveBtn').style.display = state.obsDirty ? '' : 'none';
}

// ── Canvas 初始化 ──

function obsInitCanvas() {
  var canvas = document.getElementById('obsCanvas');
  if (canvas._obsInit) return;
  canvas._obsInit = true;
  canvas.addEventListener('mousedown', obsOnMouseDown);
  canvas.addEventListener('mousemove', obsOnMouseMove);
  canvas.addEventListener('mouseup', obsOnMouseUp);
  canvas.addEventListener('dblclick', obsOnDblClick);
  canvas.addEventListener('wheel', obsOnWheel, {passive: false});
  canvas.addEventListener('contextmenu', function(e) { e.preventDefault(); });

  // ResizeObserver
  var container = canvas.parentElement;
  if (window.ResizeObserver) {
    new ResizeObserver(function() { obsDrawGraph(); }).observe(container);
  }
}

// ── 坐标转换 ──

function obsCanvasCoords(e) {
  var canvas = document.getElementById('obsCanvas');
  var rect = canvas.getBoundingClientRect();
  return {x: e.clientX - rect.left, y: e.clientY - rect.top};
}

function screenToWorld(sx, sy) {
  return {
    x: (sx - state.obsPan.x) / state.obsZoom,
    y: (sy - state.obsPan.y) / state.obsZoom
  };
}

function worldToScreen(wx, wy) {
  return {
    x: wx * state.obsZoom + state.obsPan.x,
    y: wy * state.obsZoom + state.obsPan.y
  };
}

// ── Hit testing ──

function obsHitTest(wx, wy) {
  if (!state.obsBlocks) return null;
  var hits = Object.values(state.obsBlocks).filter(function(b) {
    return wx >= b._x && wx <= b._x + OBS_NODE_W && wy >= b._y && wy <= b._y + OBS_NODE_H;
  });
  return hits.length ? hits[hits.length-1] : null;
}

function hitOutputPort(wx, wy) {
  if (!state.obsBlocks) return null;
  var r2 = PORT_HIT * PORT_HIT;
  var found = null;
  Object.values(state.obsBlocks).forEach(function(b) {
    var px = b._x + OBS_NODE_W;
    var py = b._y + OBS_NODE_H / 2;
    var dx = wx - px, dy = wy - py;
    if (dx*dx + dy*dy < r2) found = b;
  });
  return found;
}

function hitInputPort(wx, wy) {
  if (!state.obsBlocks) return null;
  var r2 = PORT_HIT * PORT_HIT;
  var found = null;
  Object.values(state.obsBlocks).forEach(function(b) {
    if (b.type === 'obs') return; // obs 节点没有输入端口
    var px = b._x;
    var py = b._y + OBS_NODE_H / 2;
    var dx = wx - px, dy = wy - py;
    if (dx*dx + dy*dy < r2) found = b;
  });
  return found;
}

// 判定点击是否命中连线（距离贝塞尔曲线 < threshold）
function hitConnection(wx, wy) {
  if (!state.obsBlocks) return null;
  var threshold = 6;
  var best = null;
  Object.values(state.obsBlocks).forEach(function(b) {
    if (!b.outputs) return;
    b.outputs.forEach(function(outId) {
      var target = state.obsBlocks[outId];
      if (!target) return;
      var x1 = b._x + OBS_NODE_W, y1 = b._y + OBS_NODE_H/2;
      var x2 = target._x, y2 = target._y + OBS_NODE_H/2;
      var dist = distToBezier(wx, wy, x1, y1, x2, y2);
      if (dist < threshold) {
        best = {from: b, toId: outId, dist: dist};
      }
    });
  });
  return best;
}

function distToBezier(px, py, x1, y1, x2, y2) {
  var cx = (x1 + x2) / 2;
  var minD = Infinity;
  for (var t = 0; t <= 1; t += 0.05) {
    var it = 1 - t;
    var bx = it*it*it*x1 + 3*it*it*t*cx + 3*it*t*t*cx + t*t*t*x2;
    var by = it*it*it*y1 + 3*it*it*t*y1 + 3*it*t*t*y2 + t*t*t*y2;
    var dx = px - bx, dy = py - by;
    var d = Math.sqrt(dx*dx + dy*dy);
    if (d < minD) minD = d;
  }
  return minD;
}

// ── Mouse handlers ──

function obsOnMouseDown(e) {
  var p = obsCanvasCoords(e);
  var w = screenToWorld(p.x, p.y);

  // 中键/右键 → 开始平移
  if (e.button === 1 || e.button === 2) {
    state.obsPanning = true;
    state.obsDragOff = {x: p.x - state.obsPan.x, y: p.y - state.obsPan.y};
    e.preventDefault();
    return;
  }

  // 左键：检查输出端口 → 开始连线
  var portHit = hitOutputPort(w.x, w.y);
  if (portHit) {
    state.obsConnecting = {fromId: portHit.id, mouseX: p.x, mouseY: p.y};
    return;
  }

  // 左键：检查节点 → 选中/拖拽
  var hit = obsHitTest(w.x, w.y);
  if (hit) {
    state.obsSelectedId = hit.id;
    state.obsDragId = hit.id;
    state.obsDragOff = {x: w.x - hit._x, y: w.y - hit._y};
    obsShowInspector(hit);
  } else {
    state.obsSelectedId = null;
    state.obsDragId = null;
    document.getElementById('obsInspector').style.display = 'none';
    // 左键空白 → 也可以平移
    state.obsPanning = true;
    state.obsDragOff = {x: p.x - state.obsPan.x, y: p.y - state.obsPan.y};
  }
  obsDrawGraph();
}

function obsOnMouseMove(e) {
  var p = obsCanvasCoords(e);

  // 正在平移
  if (state.obsPanning) {
    state.obsPan.x = p.x - state.obsDragOff.x;
    state.obsPan.y = p.y - state.obsDragOff.y;
    obsDrawGraph();
    return;
  }

  // 正在连线
  if (state.obsConnecting) {
    state.obsConnecting.mouseX = p.x;
    state.obsConnecting.mouseY = p.y;
    obsDrawGraph();
    return;
  }

  // 正在拖拽节点
  if (state.obsDragId) {
    var w = screenToWorld(p.x, p.y);
    var b = state.obsBlocks[state.obsDragId];
    if (b) {
      b._x = w.x - state.obsDragOff.x;
      b._y = w.y - state.obsDragOff.y;
      obsDrawGraph();
    }
  }
}

function obsOnMouseUp(e) {
  var p = obsCanvasCoords(e);

  // 完成连线
  if (state.obsConnecting) {
    var w = screenToWorld(p.x, p.y);
    var target = hitInputPort(w.x, w.y);
    if (target && target.id !== state.obsConnecting.fromId) {
      var fromBlock = state.obsBlocks[state.obsConnecting.fromId];
      if (fromBlock) {
        if (!fromBlock.outputs) fromBlock.outputs = [];
        if (fromBlock.outputs.indexOf(target.id) < 0) {
          fromBlock.outputs.push(target.id);
          state.obsDirty = true;
          obsUpdateUI();
        }
      }
    }
    state.obsConnecting = null;
    obsDrawGraph();
    return;
  }

  state.obsDragId = null;
  state.obsPanning = false;
}

function obsOnDblClick(e) {
  var p = obsCanvasCoords(e);
  var w = screenToWorld(p.x, p.y);
  var hit = obsHitTest(w.x, w.y);
  if (hit) obsShowInspector(hit);
}

function obsOnWheel(e) {
  e.preventDefault();
  var p = obsCanvasCoords(e);
  var oldZoom = state.obsZoom;
  var delta = e.deltaY > 0 ? 0.9 : 1.1;
  state.obsZoom = Math.min(3.0, Math.max(0.3, state.obsZoom * delta));
  // 以鼠标位置为中心缩放
  state.obsPan.x = p.x - (p.x - state.obsPan.x) * (state.obsZoom / oldZoom);
  state.obsPan.y = p.y - (p.y - state.obsPan.y) * (state.obsZoom / oldZoom);
  obsDrawGraph();
}

// ── 右键删除连线 ──

export function obsContextDelete(e) {
  var p = obsCanvasCoords(e);
  var w = screenToWorld(p.x, p.y);
  var conn = hitConnection(w.x, w.y);
  if (conn) {
    var idx = conn.from.outputs.indexOf(conn.toId);
    if (idx >= 0) {
      conn.from.outputs.splice(idx, 1);
      state.obsDirty = true;
      obsUpdateUI();
      obsDrawGraph();
    }
  }
}

// ── 绘制 ──

function obsDrawGraph() {
  var canvas = document.getElementById('obsCanvas');
  if (!canvas) return;
  var container = canvas.parentElement;
  var dpr = window.devicePixelRatio || 1;
  var cw = container.clientWidth, ch = container.clientHeight;
  canvas.width = cw * dpr; canvas.height = ch * dpr;
  canvas.style.width = cw + 'px'; canvas.style.height = ch + 'px';
  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cw, ch);

  // 网格背景
  drawGrid(ctx, cw, ch);

  if (!state.obsBlocks) return;

  ctx.save();
  ctx.translate(state.obsPan.x, state.obsPan.y);
  ctx.scale(state.obsZoom, state.obsZoom);

  // 画连线
  drawConnections(ctx);

  // 画正在拖拽的临时连线
  if (state.obsConnecting) {
    var fromBlock = state.obsBlocks[state.obsConnecting.fromId];
    if (fromBlock) {
      var x1 = fromBlock._x + OBS_NODE_W;
      var y1 = fromBlock._y + OBS_NODE_H / 2;
      var mp = screenToWorld(state.obsConnecting.mouseX, state.obsConnecting.mouseY);
      ctx.strokeStyle = 'rgba(0,212,255,0.5)';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      var cx = (x1 + mp.x) / 2;
      ctx.moveTo(x1, y1);
      ctx.bezierCurveTo(cx, y1, cx, mp.y, mp.x, mp.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // 画节点
  drawNodes(ctx);

  ctx.restore();

  // 缩放指示器
  updateZoomIndicator(cw, ch);
}

function drawGrid(ctx, cw, ch) {
  var gridSize = 30 * state.obsZoom;
  if (gridSize < 8) return;
  ctx.strokeStyle = 'rgba(255,255,255,0.03)';
  ctx.lineWidth = 1;
  var ox = state.obsPan.x % gridSize;
  var oy = state.obsPan.y % gridSize;
  for (var x = ox; x < cw; x += gridSize) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke();
  }
  for (var y = oy; y < ch; y += gridSize) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke();
  }
}

function drawConnections(ctx) {
  ctx.lineWidth = 1.5;
  Object.values(state.obsBlocks).forEach(function(b) {
    if (!b.outputs) return;
    b.outputs.forEach(function(outId) {
      var target = state.obsBlocks[outId];
      if (!target) return;
      var x1 = b._x + OBS_NODE_W, y1 = b._y + OBS_NODE_H/2;
      var x2 = target._x, y2 = target._y + OBS_NODE_H/2;
      var color = OBS_COLORS[b.type] || '#888';

      ctx.strokeStyle = color + '44';
      ctx.beginPath();
      var cx = (x1+x2)/2;
      ctx.moveTo(x1, y1);
      ctx.bezierCurveTo(cx, y1, cx, y2, x2, y2);
      ctx.stroke();

      // 箭头
      var angle = Math.atan2(y2 - (y2+y1)/2, x2 - cx);
      ctx.fillStyle = color + '66';
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - 6*Math.cos(angle-0.4), y2 - 6*Math.sin(angle-0.4));
      ctx.lineTo(x2 - 6*Math.cos(angle+0.4), y2 - 6*Math.sin(angle+0.4));
      ctx.fill();

      // 维度标注
      var dim = getDimLabel(b);
      if (dim) {
        var mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        ctx.fillStyle = '#666';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(dim, mx, my - 4);
        ctx.textAlign = 'left';
      }
    });
  });
}

function getDimLabel(block) {
  if (block.output_size) return '[' + block.output_size + ']';
  if (block.shape) {
    var s = Array.isArray(block.shape) ? block.shape : [block.shape];
    return '[' + s.join('x') + ']';
  }
  if (block.channels) return '[ch:' + (Array.isArray(block.channels) ? block.channels[block.channels.length-1] : block.channels) + ']';
  return '';
}

function drawNodes(ctx) {
  Object.values(state.obsBlocks).forEach(function(b) {
    var color = OBS_COLORS[b.type] || '#888';
    var isSelected = b.id === state.obsSelectedId;

    // 选中发光
    if (isSelected) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
    }

    // 背景
    ctx.fillStyle = isSelected ? color + '33' : '#16213e';
    ctx.strokeStyle = isSelected ? color : 'rgba(255,255,255,0.1)';
    ctx.lineWidth = isSelected ? 2 : 1;
    ctx.beginPath();
    ctx.roundRect(b._x, b._y, OBS_NODE_W, OBS_NODE_H, 4);
    ctx.fill(); ctx.stroke();
    ctx.shadowBlur = 0;

    // 左侧色条
    ctx.fillStyle = color;
    ctx.fillRect(b._x, b._y, 4, OBS_NODE_H);

    // 文字
    ctx.fillStyle = '#e0e0e0';
    ctx.font = '11px monospace';
    ctx.textBaseline = 'top';
    ctx.fillText(b.id, b._x + 10, b._y + 6, OBS_NODE_W - 14);
    ctx.fillStyle = '#888';
    ctx.font = '9px monospace';
    ctx.fillText(b.type + '/' + (b.kind || ''), b._x + 10, b._y + 22, OBS_NODE_W - 14);
    // 关键参数
    var detail = b.func || b.class_name || (b.kind === 'scale' ? 'x'+b.factor : '') || '';
    if (detail) {
      ctx.fillStyle = '#666';
      ctx.fillText(detail, b._x + 10, b._y + 36, OBS_NODE_W - 14);
    }

    // 输出端口（右侧）
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(b._x + OBS_NODE_W, b._y + OBS_NODE_H/2, PORT_R, 0, Math.PI*2);
    ctx.fill();
    if (b.outputs && b.outputs.length) {
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(b._x + OBS_NODE_W, b._y + OBS_NODE_H/2, 2, 0, Math.PI*2);
      ctx.fill();
    }

    // 输入端口（左侧，非 obs 节点）
    if (b.type !== 'obs') {
      ctx.fillStyle = '#888';
      ctx.beginPath();
      ctx.arc(b._x, b._y + OBS_NODE_H/2, PORT_R, 0, Math.PI*2);
      ctx.fill();
    }
  });
}

function updateZoomIndicator() {
  var el = document.getElementById('obsZoomIndicator');
  if (!el) {
    el = document.createElement('div');
    el.id = 'obsZoomIndicator';
    el.className = 'obs-zoom-indicator';
    document.getElementById('obsGraphContainer').appendChild(el);
  }
  el.textContent = Math.round(state.obsZoom * 100) + '%';
}

// ── Inspector ──

// 传感器类型 → 可用 output 属性
var SENSOR_OUTPUTS = {
  depth_camera:   ['depth_flat', 'depth', 'history'],
  height_scanner: ['heights_rel', 'heights_w'],
  force_sensor:   ['forces', 'magnitude', 'torques'],
  imu:            ['data', 'lin_acc_b', 'ang_vel_b'],
  contact:        ['data'],
};

function obsShowInspector(block) {
  var el = document.getElementById('obsInspector');
  el.style.display = '';
  var color = OBS_COLORS[block.type] || '#888';
  var html = '<h4 style="color:'+color+'">' + esc(block.id) + '</h4>';
  html += '<div class="field"><label>type</label><span>' + esc(block.type + '/' + (block.kind||'')) + '</span></div>';

  // ID 编辑
  html += '<div class="field"><label>id</label><input value="' + esc(block.id) + '" data-key="_id"></div>';

  // sensor block: 使用下拉选择器
  if (block.type === 'obs' && block.kind === 'sensor') {
    html += _buildSensorInspector(block);
  } else {
    // 通用 config 字段
    var skip = ['id','type','kind','outputs','_x','_y'];
    Object.keys(block).forEach(function(k) {
      if (skip.indexOf(k) >= 0) return;
      var v = block[k];
      if (typeof v === 'object' && v !== null) v = JSON.stringify(v);
      html += '<div class="field"><label>' + esc(k) + '</label><input value="' + esc(String(v||'')) + '" data-key="' + k + '"></div>';
    });
  }

  // outputs
  html += '<div class="field"><label>outputs</label><input value="' + esc((block.outputs||[]).join(', ')) + '" data-key="_outputs"></div>';

  // 操作按钮
  html += '<div style="margin-top:8px;display:flex;gap:4px">';
  html += '<button class="btn sm" id="obsDupBtn">Duplicate</button>';
  html += '<button class="btn sm danger" id="obsDelBtn">Delete</button>';
  html += '</div>';

  el.innerHTML = html;

  // 绑定通用事件
  el.querySelectorAll('input[data-key]').forEach(function(inp) {
    inp.onchange = function() {
      var key = this.dataset.key;
      var val = this.value;
      if (key === '_id') {
        var oldId = block.id;
        var newId = val.trim();
        if (!newId || newId === oldId) return;
        if (state.obsBlocks[newId]) { showToast('ID already exists', true); this.value = oldId; return; }
        Object.values(state.obsBlocks).forEach(function(b) {
          if (b.outputs) {
            b.outputs = b.outputs.map(function(o) { return o === oldId ? newId : o; });
          }
        });
        block.id = newId;
        delete state.obsBlocks[oldId];
        state.obsBlocks[newId] = block;
        state.obsSelectedId = newId;
      } else if (key === '_outputs') {
        block.outputs = val.split(',').map(function(s){return s.trim();}).filter(Boolean);
      } else {
        try { var parsed = JSON.parse(val); block[key] = parsed; }
        catch(_) {
          var num = Number(val);
          block[key] = (val !== '' && !isNaN(num) && String(num) === val) ? num : val;
        }
      }
      state.obsDirty = true; obsUpdateUI(); obsDrawGraph();
    };
  });

  // sensor 下拉事件
  _bindSensorInspectorEvents(el, block);

  document.getElementById('obsDelBtn').onclick = function() { obsDeleteBlock(block.id); };
  document.getElementById('obsDupBtn').onclick = function() { obsDuplicateBlock(block.id); };
}

function _buildSensorInspector(block) {
  var sensors = state.robotSensors || [];
  var html = '';

  // sensor_name 下拉
  html += '<div class="field"><label>sensor</label><select data-key="_sensor_name">';
  html += '<option value="">-- select sensor --</option>';
  sensors.forEach(function(s) {
    var sel = s.name === block.sensor_name ? ' selected' : '';
    html += '<option value="' + esc(s.name) + '"' + sel + '>' + esc(s.name) + ' (' + esc(s.type) + ')</option>';
  });
  html += '</select></div>';

  // output 下拉（根据选中 sensor 的 type）
  var selectedSensor = sensors.find(function(s) { return s.name === block.sensor_name; });
  var sensorType = selectedSensor ? selectedSensor.type : '';
  var outputs = SENSOR_OUTPUTS[sensorType] || [];

  html += '<div class="field"><label>output</label><select data-key="_sensor_output">';
  if (!outputs.length) {
    html += '<option value="">-- select sensor first --</option>';
  } else {
    outputs.forEach(function(o) {
      var sel = o === block.output ? ' selected' : '';
      html += '<option value="' + esc(o) + '"' + sel + '>' + esc(o) + '</option>';
    });
  }
  html += '</select></div>';

  // mount link（只读展示）
  if (selectedSensor && selectedSensor.mount_link) {
    html += '<div class="field"><label>mount</label><span style="color:var(--dim)">' + esc(selectedSensor.mount_link) + '</span></div>';
  }

  return html;
}

function _bindSensorInspectorEvents(el, block) {
  var sensorSelect = el.querySelector('[data-key="_sensor_name"]');
  var outputSelect = el.querySelector('[data-key="_sensor_output"]');
  if (!sensorSelect) return;

  sensorSelect.onchange = function() {
    block.sensor_name = this.value;
    // 自动设置默认 output
    var sensors = state.robotSensors || [];
    var s = sensors.find(function(s) { return s.name === block.sensor_name; });
    if (s) {
      var defaults = {depth_camera:'depth_flat', height_scanner:'heights_rel', force_sensor:'forces', imu:'data', contact:'data'};
      block.output = defaults[s.type] || '';
    }
    state.obsDirty = true; obsUpdateUI(); obsDrawGraph();
    obsShowInspector(block); // 重绘 inspector 刷新 output 下拉
  };

  if (outputSelect) {
    outputSelect.onchange = function() {
      block.output = this.value;
      state.obsDirty = true; obsUpdateUI(); obsDrawGraph();
    };
  }
}

export function obsDeleteBlock(id) {
  if (!state.obsBlocks || !state.obsBlocks[id]) return;
  Object.values(state.obsBlocks).forEach(function(b) {
    if (b.outputs) b.outputs = b.outputs.filter(function(o) { return o !== id; });
  });
  delete state.obsBlocks[id];
  state.obsSelectedId = null;
  document.getElementById('obsInspector').style.display = 'none';
  state.obsDirty = true; obsUpdateUI(); obsDrawGraph();
}

function obsDuplicateBlock(id) {
  var src = state.obsBlocks[id];
  if (!src) return;
  var newId = id + '_copy';
  var n = 1;
  while (state.obsBlocks[newId]) { newId = id + '_copy' + n; n++; }
  var clone = JSON.parse(JSON.stringify(src));
  clone.id = newId;
  clone._x = src._x + 20;
  clone._y = src._y + 20;
  clone.outputs = [];
  state.obsBlocks[newId] = clone;
  state.obsSelectedId = newId;
  state.obsDirty = true;
  obsUpdateUI();
  obsDrawGraph();
  obsShowInspector(clone);
}

// ── 浮动添加菜单 ──

var ADD_BLOCK_TYPES = [
  {group: 'Obs', items: [
    {type:'obs', kind:'mdp', label:'MDP term'},
    {type:'obs', kind:'sensor', label:'Sensor'},
  ]},
  {group: 'Modifier', items: [
    {type:'modifier', kind:'scale', label:'Scale'},
    {type:'modifier', kind:'noise', label:'Noise'},
    {type:'modifier', kind:'clip', label:'Clip'},
    {type:'modifier', kind:'normalize', label:'Normalize'},
    {type:'modifier', kind:'history', label:'History'},
    {type:'modifier', kind:'remap', label:'Remap'},
  ]},
  {group: 'Encoder', items: [
    {type:'encoder', kind:'conv2d', label:'Conv2D'},
    {type:'encoder', kind:'mlp', label:'MLP'},
    {type:'encoder', kind:'transformer', label:'Transformer'},
  ]},
  {group: 'Group', items: [
    {type:'group', kind:'policy', label:'Policy'},
    {type:'group', kind:'critic', label:'Critic'},
    {type:'group', kind:'amp_policy', label:'AMP Policy'},
    {type:'group', kind:'amp_reference', label:'AMP Reference'},
    {type:'group', kind:'estimator', label:'Estimator'},
    {type:'group', kind:'custom', label:'Custom'},
  ]},
];

export function obsShowAddMenu(e) {
  // 移除已有菜单
  var existing = document.querySelector('.obs-add-menu');
  if (existing) { existing.remove(); return; }

  var menu = document.createElement('div');
  menu.className = 'obs-add-menu';

  ADD_BLOCK_TYPES.forEach(function(group) {
    var gEl = document.createElement('div');
    gEl.className = 'menu-group';
    gEl.textContent = group.group;
    menu.appendChild(gEl);

    group.items.forEach(function(item) {
      var mEl = document.createElement('div');
      mEl.className = 'menu-item';
      var color = OBS_COLORS[item.type] || '#888';
      mEl.innerHTML = '<span class="menu-dot" style="background:' + color + '"></span>' + esc(item.label);
      mEl.onclick = function() {
        addBlock(item.type, item.kind);
        menu.remove();
      };
      menu.appendChild(mEl);
    });
  });

  // 定位在按钮下方
  var btn = e.target;
  var rect = btn.getBoundingClientRect();
  var container = document.getElementById('obsGraphContainer');
  var containerRect = container.getBoundingClientRect();
  menu.style.position = 'absolute';
  menu.style.left = (rect.left - containerRect.left) + 'px';
  menu.style.top = (rect.bottom - containerRect.top + 4) + 'px';
  container.appendChild(menu);

  // 点击外部关闭
  setTimeout(function() {
    document.addEventListener('click', function closeMenu(ev) {
      if (!menu.contains(ev.target) && ev.target !== btn) {
        menu.remove();
        document.removeEventListener('click', closeMenu);
      }
    });
  }, 0);
}

function addBlock(type, kind) {
  if (!state.obsBlocks) state.obsBlocks = {};
  var newId = kind + '_' + Object.keys(state.obsBlocks).length;
  var n = 0;
  while (state.obsBlocks[newId]) { n++; newId = kind + '_' + n; }

  // 在画布中央位置创建
  var canvas = document.getElementById('obsCanvas');
  var container = canvas.parentElement;
  var cx = container.clientWidth / 2, cy = container.clientHeight / 2;
  var w = screenToWorld(cx, cy);

  var block = {id: newId, type: type, kind: kind, outputs: [], _x: w.x - OBS_NODE_W/2, _y: w.y - OBS_NODE_H/2};
  if (type === 'obs' && kind === 'sensor') {
    block.sensor_name = '';
    block.output = '';
  } else if (type === 'obs') {
    block.func = '';
  }
  if (kind === 'scale') block.factor = 1.0;
  if (kind === 'noise') { block.noise_type = 'gaussian'; block.std = 0.01; }
  if (kind === 'history') { block.length = 8; block.flatten = true; }
  if (kind === 'clip') { block.min = -5.0; block.max = 5.0; }
  if (kind === 'normalize') { block.type_name = 'empirical'; }
  if (kind === 'conv2d' || kind === 'mlp' || kind === 'transformer') block.output_size = 128;
  if (kind === 'conv2d') { block.channels = [4]; block.kernel = [3]; block.stride = [1]; }
  if (kind === 'mlp') { block.hidden = [256, 128]; }

  state.obsBlocks[newId] = block;
  state.obsSelectedId = newId;
  state.obsDirty = true;
  obsUpdateUI();
  obsDrawGraph();
  obsShowInspector(block);
}

// ── Fit to view ──

export function obsFitView() {
  if (!state.obsBlocks || !Object.keys(state.obsBlocks).length) return;
  var canvas = document.getElementById('obsCanvas');
  var container = canvas.parentElement;
  var cw = container.clientWidth, ch = container.clientHeight;

  var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  Object.values(state.obsBlocks).forEach(function(b) {
    if (b._x < minX) minX = b._x;
    if (b._y < minY) minY = b._y;
    if (b._x + OBS_NODE_W > maxX) maxX = b._x + OBS_NODE_W;
    if (b._y + OBS_NODE_H > maxY) maxY = b._y + OBS_NODE_H;
  });

  var pad = 40;
  var bw = maxX - minX + pad * 2;
  var bh = maxY - minY + pad * 2;
  var zx = cw / bw, zy = ch / bh;
  state.obsZoom = Math.min(zx, zy, 2.0);
  state.obsPan.x = (cw - bw * state.obsZoom) / 2 - minX * state.obsZoom + pad * state.obsZoom;
  state.obsPan.y = (ch - bh * state.obsZoom) / 2 - minY * state.obsZoom + pad * state.obsZoom;
  obsDrawGraph();
}

export function obsResetLayout() {
  obsAutoLayout();
  state.obsDirty = true;
  obsUpdateUI();
  obsFitView();
}

// ── 保存 ──

export function saveObsPipeline() {
  if (!state.obsBlocks || !state.obsPipelineName) return;
  var blocks = Object.values(state.obsBlocks).map(function(b) {
    var item = {};
    var skip = ['_x','_y'];
    Object.keys(b).forEach(function(k) { if (skip.indexOf(k) < 0) item[k] = b[k]; });
    return item;
  });
  var payload = {schema: 'obs_pipeline_v2', name: state.obsPipelineName, version: '2.0.0', blocks: blocks};
  fetch(apiUrl('/pipeline/obs/' + state.obsPipelineName), {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
  }).then(function(r){return r.json();}).then(function(d) {
    if (d.ok) { state.obsDirty = false; obsUpdateUI(); showToast('Obs pipeline saved'); }
    else showToast('Save failed: ' + (d.error||'?'), true);
  }).catch(function(e) { showToast('Save error: '+e, true); });
}
