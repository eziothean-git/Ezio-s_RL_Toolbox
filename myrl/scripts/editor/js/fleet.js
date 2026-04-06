// fleet.js — Fleet/Servers 页面
import state from './state.js';
import { esc } from './api.js';
import { rebuildTargetSelector } from './training.js';

export function refreshFleetList() {
  fetch('/fleet').then(function(r) { return r.json(); }).then(function(data) {
    state.fleetServers = data;
    renderServerCards(data);
    rebuildTargetSelector();
  }).catch(function() {});
}

export function renderServerCards(servers) {
  var grid = document.getElementById('serverGrid');
  var ids = Object.keys(servers);
  var noServers = document.getElementById('noServers');

  if (!ids.length) {
    grid.innerHTML = '';
    grid.appendChild(noServers);
    noServers.style.display = '';
    return;
  }
  grid.innerHTML = '';
  noServers.style.display = 'none';

  ids.forEach(function(sid) {
    var s = servers[sid];
    var health = (s.health && s.health.status) || 'unknown';
    var card = document.createElement('div');
    card.className = 'server-card';
    card.innerHTML =
      '<div class="card-header">' +
        '<span class="card-name">' + esc(s.name || sid) + '</span>' +
        '<span class="card-status ' + health + '">' + health + '</span>' +
      '</div>' +
      '<div class="card-info">' +
        esc(s.ssh_host) + '<br>' +
        'Dir: ' + esc(s.remote_dir || '') +
        (s.connect_mode === 'direct' ? ' &middot; direct' : '') +
      '</div>' +
      '<div class="card-actions"></div>';

    var actions = card.querySelector('.card-actions');
    var btns = [
      {label:'Sync Code', op:'sync'},
      {label:'Setup', op:'setup'},
      {label:'Start Mgr', op:'start-manager'},
      {label:'Stop Mgr', op:'stop-manager'},
      {label:'Deploy', op:'deploy'},
      {label:'Remove', op:'remove', cls:'danger'},
    ];
    btns.forEach(function(b) {
      var btn = document.createElement('button');
      btn.className = 'btn sm' + (b.cls ? ' ' + b.cls : '');
      btn.textContent = b.label;
      btn.onclick = function() {
        if (b.op === 'deploy') showDeployModal(sid);
        else if (b.op === 'remove') removeServer(sid);
        else fleetOp(sid, b.op);
      };
      actions.appendChild(btn);
    });

    grid.appendChild(card);
  });
}

export function toggleAddForm() {
  var f = document.getElementById('addForm');
  f.style.display = f.style.display === 'none' ? '' : 'none';
}

export function addServer() {
  var body = {
    name: document.getElementById('srvName').value.trim(),
    ssh_host: document.getElementById('srvHost').value.trim(),
    ssh_port: parseInt(document.getElementById('srvPort').value) || 22,
    ssh_key: document.getElementById('srvKey').value.trim(),
    remote_dir: document.getElementById('srvDir').value.trim() || '~/Ezios_RL_Toolbox',
    connect_mode: document.getElementById('srvMode').value,
  };
  if (!body.name || !body.ssh_host) { alert('Name and SSH Host are required'); return; }
  fetch('/fleet/add', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.ok) {
        document.getElementById('addForm').style.display = 'none';
        document.getElementById('srvName').value = '';
        document.getElementById('srvHost').value = '';
        refreshFleetList();
      } else {
        alert('Failed: ' + d.msg);
      }
    })
    .catch(function(e) { alert('Error: ' + e); });
}

export function removeServer(sid) {
  var s = state.fleetServers[sid];
  if (!confirm('Remove "' + (s && s.name || sid) + '"?')) return;
  fetch('/fleet/remove', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({id: sid})})
    .then(function(r) { return r.json(); })
    .then(function(d) { if (d.ok) refreshFleetList(); })
    .catch(function() {});
}

export function fleetOp(sid, op) {
  state.currentFleetOp = {server: sid, op: op};
  var s = state.fleetServers[sid];
  document.getElementById('opModalTitle').textContent = op + ' — ' + (s && s.name || sid);
  document.getElementById('opModalStatus').textContent = 'starting...';
  document.getElementById('opModalStatus').style.color = 'var(--dim)';
  document.getElementById('opModalConsole').innerHTML = '';
  document.getElementById('opModal').style.display = '';

  fetch('/fleet/' + sid + '/' + op, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'})
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (!d.ok) {
        document.getElementById('opModalStatus').textContent = 'failed: ' + (d.msg || d.error);
        document.getElementById('opModalStatus').style.color = 'var(--warn)';
      }
    })
    .catch(function(e) { document.getElementById('opModalStatus').textContent = 'error: ' + e; });
}

export function showDeployModal(sid) {
  state.deployTargetId = sid;
  var s = state.fleetServers[sid];
  document.getElementById('deployTarget').textContent = s && s.name || sid;
  document.getElementById('deployPkgPath').value = '';
  document.getElementById('deployModal').style.display = '';
}

export function deployPackage() {
  var pkg = document.getElementById('deployPkgPath').value.trim();
  if (!pkg) { alert('Package path is required'); return; }
  document.getElementById('deployModal').style.display = 'none';
  state.currentFleetOp = {server: state.deployTargetId, op: 'deploy'};
  var s = state.fleetServers[state.deployTargetId];
  document.getElementById('opModalTitle').textContent = 'deploy — ' + (s && s.name || state.deployTargetId);
  document.getElementById('opModalStatus').textContent = 'deploying...';
  document.getElementById('opModalStatus').style.color = 'var(--dim)';
  document.getElementById('opModalConsole').innerHTML = '';
  document.getElementById('opModal').style.display = '';
  fetch('/fleet/' + state.deployTargetId + '/deploy', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({package: pkg})})
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (!d.ok) {
        document.getElementById('opModalStatus').textContent = 'failed: ' + d.msg;
        document.getElementById('opModalStatus').style.color = 'var(--warn)';
      }
    })
    .catch(function(e) { document.getElementById('opModalStatus').textContent = 'error: ' + e; });
}
