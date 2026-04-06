// reward.js — 奖励管线编辑器
import state from './state.js';
import { apiUrl, esc, showToast } from './api.js';

export function renderRewardPipeline(cfg) {
  var el = document.getElementById('rewardList');
  var rp = (cfg.assets && cfg.assets.reward_pipeline) || {};
  state.rewPipelineName = rp.name || '';
  if (!state.rewPipelineName) {
    el.innerHTML = '<div class="todo-hint">No reward pipeline configured in this experiment</div>';
    document.getElementById('rewGlobalCtrl').style.display = 'none';
    return;
  }
  el.innerHTML = '<div class="todo-hint">Loading reward pipeline...</div>';
  Promise.all([
    state.rewSchema ? Promise.resolve(state.rewSchema) : fetch(apiUrl('/reward-schema')).then(function(r) { return r.json(); }),
    fetch(apiUrl('/pipeline/reward/' + state.rewPipelineName)).then(function(r) { return r.json(); })
  ]).then(function(results) {
    state.rewSchema = results[0];
    state.rewPipeline = results[1];
    if (state.rewPipeline.error) { el.innerHTML = '<div class="todo-hint">' + esc(state.rewPipeline.error) + '</div>'; return; }
    state.rewRatios = state.rewPipeline.terms.map(function(t) { return Math.abs(t.weight) || 0.01; });
    var rebalance = (state.rewPipeline.transforms || []).find(function(t) { return t.name === 'relative_rebalance'; });
    var normalize = (state.rewPipeline.transforms || []).find(function(t) { return t.name === 'running_normalize'; });
    document.getElementById('rewTotalScale').value = (rebalance && rebalance.params && rebalance.params.total_scale) || 1.0;
    document.getElementById('rewNormalize').checked = !!normalize;
    document.getElementById('rewGlobalCtrl').style.display = '';
    document.getElementById('rewTotalScale').onchange = function() { markRewDirty(); };
    document.getElementById('rewNormalize').onchange = function() { markRewDirty(); };
    state.rewDirty = false;
    populateAddTermSelect();
    renderRewardTerms();
  }).catch(function(e) { el.innerHTML = '<div class="todo-hint">Failed: ' + esc(String(e)) + '</div>'; });
}

function populateAddTermSelect() {
  var sel = document.getElementById('rewAddSelect');
  sel.innerHTML = '<option value="">+ Add reward term...</option>';
  if (!state.rewSchema || !state.rewSchema.terms) return;
  var names = Object.keys(state.rewSchema.terms).sort();
  names.forEach(function(n) {
    var t = state.rewSchema.terms[n];
    var opt = document.createElement('option');
    opt.value = n;
    opt.textContent = n + (t.tags ? ' [' + t.tags.join(', ') + ']' : '');
    sel.appendChild(opt);
  });
  sel.onchange = function() {
    if (!this.value) return;
    addRewardTerm(this.value);
    this.value = '';
  };
}

function renderRewardTerms() {
  var el = document.getElementById('rewardList');
  el.innerHTML = '';
  if (!state.rewPipeline || !state.rewPipeline.terms) return;
  document.getElementById('rewTermCount').textContent = state.rewPipeline.terms.length;
  state.rewPipeline.terms.forEach(function(term, idx) {
    el.appendChild(buildTermCard(term, idx));
  });
  renderRatioBar();
  updateDirtyUI();
}

function renderRatioBar() {
  var bar = document.getElementById('rewRatioBar');
  bar.innerHTML = '';
  if (!state.rewPipeline || !state.rewPipeline.terms || !state.rewRatios.length) return;
  var total = state.rewRatios.reduce(function(a,b) { return a + b; }, 0) || 1;
  state.rewPipeline.terms.forEach(function(term, idx) {
    var pct = (state.rewRatios[idx] / total * 100);
    var seg = document.createElement('div');
    seg.className = 'seg';
    seg.style.width = pct + '%';
    seg.style.background = state.rewTermColors[idx % state.rewTermColors.length];
    seg.title = term.name + ': ' + pct.toFixed(1) + '%';
    if (pct > 8) seg.textContent = pct.toFixed(0) + '%';
    bar.appendChild(seg);
  });
}

function buildTermCard(term, idx) {
  var card = document.createElement('div');
  card.className = 'reward-card';
  var schema = (state.rewSchema && state.rewSchema.terms && state.rewSchema.terms[term.name]) || {};
  var desc = schema.description || '';
  var tags = (schema.tags || []).join(', ');

  var color = state.rewTermColors[idx % state.rewTermColors.length];
  var sign = term.weight < 0 ? 'penalty' : 'reward';
  var totalR = state.rewRatios.reduce(function(a,b) { return a + b; }, 0) || 1;
  var pct = (state.rewRatios[idx] / totalR * 100).toFixed(1);

  var hdr = document.createElement('div');
  hdr.className = 'reward-card-header';
  hdr.innerHTML =
    '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + color + '"></span>' +
    '<span class="rw-name">' + esc(term.name) + '</span>' +
    '<span class="rw-tags">' + esc(sign) + (tags ? ' · ' + esc(tags) : '') + '</span>' +
    '<div class="rw-controls">' +
      '<input type="range" class="rw-ratio-slider" min="0" max="100" value="' + Math.round(state.rewRatios[idx] * 10) + '" data-idx="' + idx + '">' +
      '<span class="rw-ratio-val" data-idx="' + idx + '">' + pct + '%</span>' +
      '<select class="rw-sign-select" data-idx="' + idx + '" style="background:var(--bg);color:var(--text);border:1px solid var(--border);font-size:10px;padding:1px 2px;border-radius:2px">' +
        '<option value="+"' + (term.weight >= 0 ? ' selected' : '') + '>+</option>' +
        '<option value="-"' + (term.weight < 0 ? ' selected' : '') + '>&minus;</option>' +
      '</select>' +
      '<button class="btn sm" title="Move up" data-action="move-up" data-idx="' + idx + '">&uarr;</button>' +
      '<button class="btn sm" title="Move down" data-action="move-down" data-idx="' + idx + '">&darr;</button>' +
      '<button class="btn sm danger" title="Remove" data-action="remove" data-idx="' + idx + '">&times;</button>' +
    '</div>';
  card.appendChild(hdr);

  // Button handlers via delegation
  hdr.querySelectorAll('[data-action]').forEach(function(btn) {
    btn.onclick = function() {
      var i = parseInt(this.dataset.idx);
      if (this.dataset.action === 'move-up') moveRewardTerm(i, -1);
      else if (this.dataset.action === 'move-down') moveRewardTerm(i, 1);
      else if (this.dataset.action === 'remove') removeRewardTerm(i);
    };
  });

  var slider = hdr.querySelector('.rw-ratio-slider');
  var valSpan = hdr.querySelector('.rw-ratio-val');
  slider.oninput = function() {
    state.rewRatios[idx] = parseInt(this.value) / 10;
    var tot = state.rewRatios.reduce(function(a,b){return a+b;},0)||1;
    valSpan.textContent = (state.rewRatios[idx]/tot*100).toFixed(1) + '%';
    renderRatioBar();
    markRewDirty();
  };
  hdr.querySelector('.rw-sign-select').onchange = function() {
    var absW = Math.abs(state.rewPipeline.terms[idx].weight) || state.rewRatios[idx];
    state.rewPipeline.terms[idx].weight = this.value === '-' ? -absW : absW;
    markRewDirty();
  };

  if (desc) {
    var descEl = document.createElement('div');
    descEl.className = 'rw-desc';
    descEl.textContent = desc;
    card.appendChild(descEl);
  }

  var params = term.params || {};
  var ps = schema.params_schema;
  if (ps && ps.properties && Object.keys(ps.properties).length > 0) {
    var paramsDiv = document.createElement('div');
    paramsDiv.className = 'rw-params';
    Object.keys(ps.properties).forEach(function(pname) {
      var pschema = ps.properties[pname];
      var val = params[pname];
      var row = document.createElement('div');
      row.className = 'rw-param-row';

      if (val && typeof val === 'object' && (val.__query_sensor__ || val.__query_pattern__)) {
        row.innerHTML = '<label>' + esc(pname) + '</label><span class="rw-deferred">' + esc(JSON.stringify(val)) + '</span>';
        paramsDiv.appendChild(row);
        return;
      }

      var inputHtml = '';
      if (pschema.type === 'number' || pschema.type === 'integer') {
        var min = pschema.minimum != null ? pschema.minimum : '';
        var max = pschema.maximum != null ? pschema.maximum : '';
        var step = pschema.type === 'integer' ? '1' : 'any';
        inputHtml = '<input type="number" step="' + step + '"' +
          (min !== '' ? ' min="' + min + '"' : '') +
          (max !== '' ? ' max="' + max + '"' : '') +
          ' value="' + (val != null ? val : (pschema.default || '')) + '"' +
          ' data-idx="' + idx + '" data-param="' + pname + '" class="rw-param-input">';
      } else if (pschema.type === 'string') {
        inputHtml = '<input type="text" value="' + esc(val || pschema.default || '') + '"' +
          ' data-idx="' + idx + '" data-param="' + pname + '" class="rw-param-input">';
      } else {
        inputHtml = '<span class="rw-readonly">' + esc(JSON.stringify(val != null ? val : pschema.default)) + '</span>';
      }

      var unit = pschema.unit ? ' <span class="rw-unit">' + esc(pschema.unit) + '</span>' : '';
      row.innerHTML = '<label title="' + esc(pschema.description || '') + '">' + esc(pname) + '</label>' + inputHtml + unit;
      paramsDiv.appendChild(row);
    });
    card.appendChild(paramsDiv);

    paramsDiv.querySelectorAll('.rw-param-input').forEach(function(inp) {
      inp.onchange = function() {
        var i = parseInt(this.dataset.idx);
        var p = this.dataset.param;
        var v = this.type === 'number' ? parseFloat(this.value) : this.value;
        if (!state.rewPipeline.terms[i].params) state.rewPipeline.terms[i].params = {};
        state.rewPipeline.terms[i].params[p] = v;
        markRewDirty();
      };
    });
  }

  return card;
}

function addRewardTerm(name) {
  if (!state.rewPipeline) return;
  var schema = (state.rewSchema && state.rewSchema.terms && state.rewSchema.terms[name]) || {};
  var defaultParams = {};
  if (schema.params_schema && schema.params_schema.properties) {
    Object.keys(schema.params_schema.properties).forEach(function(p) {
      var ps = schema.params_schema.properties[p];
      if (ps.default != null) defaultParams[p] = ps.default;
    });
  }
  var avgRatio = state.rewRatios.length ? state.rewRatios.reduce(function(a,b){return a+b;},0) / state.rewRatios.length : 1;
  state.rewPipeline.terms.push({name: name, weight: avgRatio, params: defaultParams});
  state.rewRatios.push(avgRatio);
  markRewDirty();
  renderRewardTerms();
}

export function removeRewardTerm(idx) {
  if (!state.rewPipeline) return;
  state.rewPipeline.terms.splice(idx, 1);
  state.rewRatios.splice(idx, 1);
  markRewDirty();
  renderRewardTerms();
}

export function moveRewardTerm(idx, dir) {
  if (!state.rewPipeline) return;
  var newIdx = idx + dir;
  if (newIdx < 0 || newIdx >= state.rewPipeline.terms.length) return;
  var t = state.rewPipeline.terms.splice(idx, 1)[0];
  state.rewPipeline.terms.splice(newIdx, 0, t);
  var r = state.rewRatios.splice(idx, 1)[0];
  state.rewRatios.splice(newIdx, 0, r);
  markRewDirty();
  renderRewardTerms();
}

function markRewDirty() {
  state.rewDirty = true;
  updateDirtyUI();
}

function updateDirtyUI() {
  document.getElementById('rewDirtyBadge').style.display = state.rewDirty ? '' : 'none';
  document.getElementById('rewSaveBtn').style.display = state.rewDirty ? '' : 'none';
}

export function saveRewardPipeline() {
  if (!state.rewPipeline || !state.rewPipelineName) return;
  var totalScale = parseFloat(document.getElementById('rewTotalScale').value) || 1.0;
  var useNormalize = document.getElementById('rewNormalize').checked;

  var totalR = state.rewRatios.reduce(function(a,b){return a+b;},0) || 1;
  state.rewPipeline.terms.forEach(function(term, idx) {
    var ratio = state.rewRatios[idx] / totalR;
    var sign = term.weight < 0 ? -1 : 1;
    term.weight = parseFloat((sign * ratio * totalScale).toFixed(6));
  });

  var transforms = [];
  var targetRatios = {};
  state.rewPipeline.terms.forEach(function(term, idx) {
    targetRatios[term.name] = parseFloat((state.rewRatios[idx] / totalR).toFixed(4));
  });
  transforms.push({
    name: 'relative_rebalance',
    params: {target_ratios: targetRatios, total_scale: totalScale, window: 500, lr: 0.01}
  });
  if (useNormalize) {
    transforms.push({name: 'running_normalize', params: {window: 1000, min_std: 0.001}});
  }
  state.rewPipeline.transforms = transforms;

  var payload = {name: state.rewPipeline.name, version: state.rewPipeline.version || '1.0.0',
    description: state.rewPipeline.description || '', terms: state.rewPipeline.terms, transforms: state.rewPipeline.transforms};
  fetch(apiUrl('/pipeline/reward/' + state.rewPipelineName), {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.ok) {
      state.rewDirty = false;
      updateDirtyUI();
      showToast('Saved: ' + state.rewPipeline.terms.length + ' terms + rebalance transform (scale=' + totalScale + ')');
    } else {
      showToast('Save failed: ' + (d.error || 'unknown'), true);
    }
  }).catch(function(e) { showToast('Save error: ' + e, true); });
}
