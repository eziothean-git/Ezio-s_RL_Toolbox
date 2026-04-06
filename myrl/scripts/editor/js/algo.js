// algo.js — 算法配置编辑器
import state from './state.js';
import { apiUrl, esc, showToast } from './api.js';

var ALGO_FIELDS = [
  // Runner
  {key:'num_steps_per_env', label:'Steps per env', type:'int', group:'Runner', hint:'rollout 步数/env', min:1, max:256},
  {key:'max_iterations', label:'Max iterations', type:'int', group:'Runner', hint:'PPO 总迭代数', min:1, max:1000000},
  {key:'save_interval', label:'Save interval', type:'int', group:'Runner', hint:'每 N iter 保存 checkpoint', min:1, max:100000},
  {key:'device', label:'Device', type:'select', group:'Runner', options:['cuda:0','cuda:1','cpu']},
  // Policy
  {key:'policy.class_name', label:'Policy class', type:'select', group:'Policy', options:['ActorCritic','ActorCriticRecurrent','MoEActorCritic','EncoderActorCritic','EncoderMoEActorCritic','VaeActorCritic']},
  {key:'policy.actor_hidden_dims', label:'Actor dims', type:'text', group:'Policy', hint:'逗号分隔，如 256,256,128'},
  {key:'policy.critic_hidden_dims', label:'Critic dims', type:'text', group:'Policy', hint:'逗号分隔'},
  {key:'policy.init_noise_std', label:'Init noise std', type:'float', group:'Policy', min:0.01, max:5.0, step:0.1},
  // PPO Algorithm
  {key:'algorithm.class_name', label:'Algorithm', type:'select', group:'PPO', options:['PPO','TPPO','EstimatorPPO','WasabiPPO','WasabiEstimatorPPO','VAEDistillPPO']},
  {key:'algorithm.learning_rate', label:'Learning rate', type:'float', group:'PPO', min:0.00001, max:0.1, step:0.0001},
  {key:'algorithm.num_learning_epochs', label:'Learning epochs', type:'int', group:'PPO', min:1, max:32},
  {key:'algorithm.num_mini_batches', label:'Mini batches', type:'int', group:'PPO', min:1, max:64},
  {key:'algorithm.gamma', label:'Gamma (γ)', type:'float', group:'PPO', hint:'折扣因子', min:0.9, max:0.9999, step:0.001},
  {key:'algorithm.lam', label:'Lambda (λ)', type:'float', group:'PPO', hint:'GAE λ', min:0.8, max:1.0, step:0.01},
  {key:'algorithm.desired_kl', label:'Desired KL', type:'float', group:'PPO', hint:'adaptive schedule 目标 KL', min:0.001, max:0.1, step:0.001},
  {key:'algorithm.max_grad_norm', label:'Max grad norm', type:'float', group:'PPO', min:0.1, max:10.0, step:0.1},
  {key:'algorithm.entropy_coef', label:'Entropy coef', type:'float', group:'PPO', hint:'熵正则化系数', min:0, max:0.1, step:0.001},
  {key:'algorithm.use_clipped_value_loss', label:'Clipped value loss', type:'bool', group:'PPO'},
  {key:'algorithm.schedule', label:'LR schedule', type:'select', group:'PPO', options:['fixed','adaptive']},
];

export function renderAlgoInfo(cfg) {
  var el = document.getElementById('algoInfo');
  var algoRef = (cfg.assets && cfg.assets.algo_cfg) || {};
  state.algoPipelineName = algoRef.name || '';
  if (!state.algoPipelineName) {
    el.innerHTML = '<div class="todo-hint">No algo config in this experiment</div>';
    return;
  }
  el.innerHTML = '<div class="todo-hint">Loading algo config...</div>';
  fetch(apiUrl('/pipeline/algo/' + state.algoPipelineName)).then(function(r) { return r.json(); }).then(function(data) {
    if (data.error) { el.innerHTML = '<div class="todo-hint">' + esc(data.error) + '</div>'; return; }
    state.algoPipeline = data;
    state.algoDirty = false;
    renderAlgoFields();
  }).catch(function(e) { el.innerHTML = '<div class="todo-hint">Failed: ' + e + '</div>'; });
}

function getNestedVal(obj, path) {
  var parts = path.split('.');
  var cur = obj;
  for (var i = 0; i < parts.length; i++) {
    if (cur == null) return undefined;
    cur = cur[parts[i]];
  }
  return cur;
}

function setNestedVal(obj, path, val) {
  var parts = path.split('.');
  var cur = obj;
  for (var i = 0; i < parts.length - 1; i++) {
    if (!cur[parts[i]]) cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = val;
}

function renderAlgoFields() {
  var el = document.getElementById('algoInfo');
  el.innerHTML = '';
  if (!state.algoPipeline) return;
  var lastGroup = '';
  ALGO_FIELDS.forEach(function(f) {
    if (f.group !== lastGroup) {
      lastGroup = f.group;
      var g = document.createElement('div');
      g.className = 'algo-group';
      g.innerHTML = '<div class="algo-group-title">' + esc(f.group) + '</div>';
      el.appendChild(g);
    }
    var row = document.createElement('div');
    row.className = 'algo-row';
    var val = getNestedVal(state.algoPipeline, f.key);
    var inputHtml = '';

    if (f.type === 'int') {
      inputHtml = '<input type="number" step="1"' +
        (f.min != null ? ' min="' + f.min + '"' : '') +
        (f.max != null ? ' max="' + f.max + '"' : '') +
        ' value="' + (val != null ? val : '') + '" data-key="' + f.key + '" data-type="int">';
    } else if (f.type === 'float') {
      inputHtml = '<input type="number" step="' + (f.step || 'any') + '"' +
        (f.min != null ? ' min="' + f.min + '"' : '') +
        (f.max != null ? ' max="' + f.max + '"' : '') +
        ' value="' + (val != null ? val : '') + '" data-key="' + f.key + '" data-type="float">';
    } else if (f.type === 'bool') {
      inputHtml = '<input type="checkbox"' + (val ? ' checked' : '') + ' data-key="' + f.key + '" data-type="bool">';
    } else if (f.type === 'select') {
      inputHtml = '<select data-key="' + f.key + '" data-type="select">';
      (f.options || []).forEach(function(o) {
        inputHtml += '<option' + (o === String(val) || o === val ? ' selected' : '') + '>' + esc(o) + '</option>';
      });
      inputHtml += '</select>';
    } else if (f.type === 'text') {
      var displayVal = Array.isArray(val) ? val.join(', ') : (val || '');
      inputHtml = '<input type="text" value="' + esc(displayVal) + '" data-key="' + f.key + '" data-type="intarray">';
    }

    var hint = f.hint ? '<span class="algo-hint">' + esc(f.hint) + '</span>' : '';
    row.innerHTML = '<label>' + esc(f.label) + '</label>' + inputHtml + hint;
    el.lastElementChild.appendChild(row);
  });

  el.querySelectorAll('[data-key]').forEach(function(inp) {
    var handler = function() {
      var key = this.dataset.key;
      var t = this.dataset.type;
      var v;
      if (t === 'int') v = parseInt(this.value);
      else if (t === 'float') v = parseFloat(this.value);
      else if (t === 'bool') v = this.checked;
      else if (t === 'intarray') v = this.value.split(',').map(function(s) { return parseInt(s.trim()); }).filter(function(n) { return !isNaN(n); });
      else v = this.value;
      setNestedVal(state.algoPipeline, key, v);
      state.algoDirty = true;
      document.getElementById('algoDirtyBadge').style.display = '';
      document.getElementById('algoSaveBtn').style.display = '';
    };
    inp.onchange = handler;
  });
  document.getElementById('algoDirtyBadge').style.display = 'none';
  document.getElementById('algoSaveBtn').style.display = 'none';
}

export function saveAlgoPipeline() {
  if (!state.algoPipeline || !state.algoPipelineName) return;
  fetch(apiUrl('/pipeline/algo/' + state.algoPipelineName), {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(state.algoPipeline)
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.ok) {
      state.algoDirty = false;
      document.getElementById('algoDirtyBadge').style.display = 'none';
      document.getElementById('algoSaveBtn').style.display = 'none';
      showToast('Algorithm config saved');
    } else {
      showToast('Save failed: ' + (d.error || 'unknown'), true);
    }
  }).catch(function(e) { showToast('Save error: ' + e, true); });
}
