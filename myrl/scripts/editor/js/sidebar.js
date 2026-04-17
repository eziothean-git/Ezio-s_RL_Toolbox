// sidebar.js — 侧栏（Project > Task 树形）+ 实验详情加载 + 资产列表
import state from './state.js';
import { apiUrl, esc } from './api.js';
import { renderRewardPipeline } from './reward.js';
import { renderObsPipeline } from './obs-graph.js';
import { renderAlgoInfo } from './algo.js';
import { renderRobotViewer } from './robot-viewer.js';

export function loadSidebar() {
  fetch(apiUrl('/experiments')).then(function(r) { return r.json(); }).then(function(data) {
    var el = document.getElementById('expList');
    el.innerHTML = '';
    if (!data.length) {
      el.innerHTML = '<div class="item" style="color:var(--dim);cursor:default">no experiments</div>';
      return;
    }
    data.forEach(function(exp) {
      // Project 节点（experiment）
      var node = document.createElement('div');
      node.className = 'tree-node';

      var header = document.createElement('div');
      header.className = 'item tree-parent';
      header.dataset.type = 'experiment';
      header.dataset.name = exp.name;
      header.innerHTML = '<span class="tree-arrow">&#9660;</span> ' + esc(exp.name) +
        '<span class="sub">' + esc(exp.description || '') + '</span>';
      header.onclick = function(e) {
        // 点击箭头折叠/展开
        var children = node.querySelector('.tree-children');
        var arrow = header.querySelector('.tree-arrow');
        if (children) {
          var collapsed = children.style.display === 'none';
          children.style.display = collapsed ? '' : 'none';
          arrow.textContent = collapsed ? '\u25BE' : '\u25B8';
        }
        selectExperiment(exp);
      };
      node.appendChild(header);

      // Task 子节点
      var tasks = exp.tasks || [];
      if (tasks.length) {
        var childrenEl = document.createElement('div');
        childrenEl.className = 'tree-children';
        tasks.forEach(function(task) {
          var taskEl = document.createElement('div');
          taskEl.className = 'item tree-child';
          taskEl.dataset.type = 'task';
          taskEl.dataset.name = task.id;
          taskEl.dataset.experiment = exp.name;
          var label = task.label || task.id.replace('myrl/', '');
          var icon = task.type === 'play' ? '\u25B6' : '\u2699';
          taskEl.innerHTML = '<span style="font-size:10px;margin-right:4px">' + icon + '</span>' + esc(label);
          taskEl.onclick = function(e) {
            e.stopPropagation();
            selectTask(exp, task);
          };
          childrenEl.appendChild(taskEl);
        });
        node.appendChild(childrenEl);
      }

      el.appendChild(node);
    });
  }).catch(function() {});
}

function selectExperiment(exp) {
  clearActiveItems();
  var items = document.querySelectorAll('.sidebar .item[data-name="' + exp.name + '"]');
  items.forEach(function(el) { el.classList.add('active'); });

  state.selected = { type: 'experiment', name: exp.name, data: exp };

  document.getElementById('emptyPanel').style.display = 'none';
  document.getElementById('expDetailPanel').style.display = '';
  document.getElementById('taskConfigPanel').style.display = 'none';
  document.getElementById('expTitle').textContent = exp.name;
  document.getElementById('expVersion').textContent = exp.version ? 'v' + exp.version : '';
  document.getElementById('expDesc').textContent = exp.description || '';
  loadExperimentDetail(exp.name);
}

function selectTask(exp, task) {
  clearActiveItems();
  var items = document.querySelectorAll('.sidebar .item[data-name="' + task.id + '"]');
  items.forEach(function(el) { el.classList.add('active'); });

  // 选中 task 时同时记录 experiment（训练时需要）
  state.selected = { type: 'task', name: task.id, data: task, experiment: exp.name };

  document.getElementById('emptyPanel').style.display = 'none';
  document.getElementById('expDetailPanel').style.display = '';
  document.getElementById('taskConfigPanel').style.display = 'none';
  document.getElementById('expTitle').textContent = exp.name + ' / ' + (task.label || task.id);
  document.getElementById('expVersion').textContent = exp.version ? 'v' + exp.version : '';
  document.getElementById('expDesc').textContent = task.type + ' task';
  loadExperimentDetail(exp.name);
}

function clearActiveItems() {
  document.querySelectorAll('.sidebar .item').forEach(function(el) {
    el.classList.remove('active');
  });
}

// 保留旧接口供 app.js 兼容
export function selectItem(type, name, data) {
  if (type === 'experiment') {
    selectExperiment(data);
  } else {
    selectTask(data._exp || {name: ''}, data);
  }
}

export function loadExperimentDetail(name) {
  fetch(apiUrl('/experiment/' + name)).then(function(r) { return r.json(); }).then(function(cfg) {
    renderAssets(cfg);
    renderRobotViewer(cfg);
    renderRewardPipeline(cfg);
    renderObsPipeline(cfg);
    renderAlgoInfo(cfg);
  }).catch(function() {
    document.getElementById('assetList').innerHTML = '<div class="todo-hint">Failed to load experiment config</div>';
  });
}

export function renderAssets(cfg) {
  var el = document.getElementById('assetList');
  el.innerHTML = '';
  var assets = cfg.assets || {};
  var types = ['robot_model', 'actuator_cfg', 'sensor_cfg', 'terrain', 'reward_fns', 'reward_pipeline', 'obs_pipeline', 'algo_cfg', 'env_script'];
  var count = 0;
  types.forEach(function(t) {
    var a = assets[t];
    if (!a) return;
    var items = Array.isArray(a) ? a : [a];
    items.forEach(function(item) {
      count++;
      var row = document.createElement('div');
      row.className = 'asset-row';
      row.innerHTML =
        '<span class="asset-type">' + t.replace(/_/g, ' ') + '</span>' +
        '<span class="asset-name">' + esc(item.name || '—') + '</span>' +
        '<span class="asset-ver">' + esc(item.version || '') + '</span>';
      el.appendChild(row);
    });
  });
  document.getElementById('assetCount').textContent = count;
  if (!count) el.innerHTML = '<div class="todo-hint">No assets defined in this experiment</div>';
}

export function toggleSidebarSection(id) {
  var el = document.getElementById(id);
  el.style.display = el.style.display === 'none' ? '' : 'none';
}

export function switchPage(name) {
  document.querySelectorAll('.page').forEach(function(p) { p.classList.remove('active'); });
  document.querySelectorAll('header .tab').forEach(function(t) { t.classList.remove('active'); });
  document.getElementById('page-' + name).classList.add('active');
  var tabs = document.querySelectorAll('header .tab');
  for (var i = 0; i < tabs.length; i++) {
    if (tabs[i].textContent.toLowerCase() === name) tabs[i].classList.add('active');
  }
  if (name === 'servers') {
    import('./fleet.js').then(function(m) { m.refreshFleetList(); });
  }
  if (name === 'debug') {
    import('./debug-tools.js').then(function(m) { m.onDebugPageEnter(); });
  }
  if (name === 'run') {
    populateRunTaskSelector();
  }
}

// ── Run 页 Task 下拉选择器 ──

function populateRunTaskSelector() {
  var sel = document.getElementById('runTaskSelect');
  if (!sel) return;

  // 从 /experiments 获取所有 task
  fetch(apiUrl('/experiments')).then(function(r) { return r.json(); }).then(function(exps) {
    sel.innerHTML = '';
    var currentTask = (state.selected && state.selected.type === 'task') ? state.selected.name : '';

    exps.forEach(function(exp) {
      // 实验名作为 optgroup
      var group = document.createElement('optgroup');
      group.label = exp.name;
      var tasks = exp.tasks || [];
      tasks.forEach(function(task) {
        var opt = document.createElement('option');
        opt.value = task.id;
        var icon = task.type === 'play' ? '\u25B6 ' : '\u2699 ';
        opt.textContent = icon + (task.label || task.id);
        if (task.id === currentTask) opt.selected = true;
        group.appendChild(opt);
      });
      sel.appendChild(group);
    });

    // 没有从 Editor 选中的话，默认选第一个 train 类型
    if (!currentTask && sel.options.length > 0) {
      for (var i = 0; i < sel.options.length; i++) {
        if (sel.options[i].textContent.indexOf('\u2699') >= 0) {
          sel.selectedIndex = i;
          break;
        }
      }
    }
  }).catch(function() {});
}
