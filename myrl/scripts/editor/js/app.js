// app.js — 入口模块：初始化 + window 全局导出
import state from './state.js';
import { switchPage, toggleSidebarSection, loadSidebar } from './sidebar.js';
import { saveRewardPipeline, moveRewardTerm, removeRewardTerm } from './reward.js';
import { obsShowAddMenu, obsDeleteBlock, saveObsPipeline, obsFitView, obsResetLayout } from './obs-graph.js';
import { saveAlgoPipeline } from './algo.js';
import { startTraining, postCmd, onTargetChange, containerCmd, fetchContainerStatus } from './training.js';
import { connectSSE, connectFleetSSE, fetchStatus, loadConsoleHistory, clearConsole } from './sse.js';
import { refreshFleetList, toggleAddForm, addServer, deployPackage } from './fleet.js';
import { saveSensorManifest, robotResetView, showAddSensorMenu, removeSensor, onSensorConfigChange, selectLinkByName } from './robot-viewer.js';

// ── 将 inline onclick 需要的函数挂载到 window ──
window.switchPage = switchPage;
window.toggleSidebarSection = toggleSidebarSection;
window.saveRewardPipeline = saveRewardPipeline;
window.moveRewardTerm = moveRewardTerm;
window.removeRewardTerm = removeRewardTerm;
window.obsShowAddMenu = obsShowAddMenu;
window.obsDeleteBlock = obsDeleteBlock;
window.saveObsPipeline = saveObsPipeline;
window.obsFitView = obsFitView;
window.obsResetLayout = obsResetLayout;
window.saveAlgoPipeline = saveAlgoPipeline;
window.startTraining = startTraining;
window.postCmd = postCmd;
window.onTargetChange = onTargetChange;
window.containerCmd = containerCmd;
window.clearConsole = clearConsole;
window.toggleAddForm = toggleAddForm;
window.addServer = addServer;
window.deployPackage = deployPackage;
window.refreshFleetList = refreshFleetList;
window.saveSensorManifest = saveSensorManifest;
window.robotResetView = robotResetView;
window.showAddSensorMenu = showAddSensorMenu;
window.removeSensor = removeSensor;
window.onSensorConfigChange = onSensorConfigChange;
window.selectLinkByName = selectLinkByName;

// ── 初始化 ──
(function init() {
  loadSidebar();
  loadConsoleHistory();
  connectSSE();
  connectFleetSSE();
  fetchStatus();
  fetchContainerStatus();
  refreshFleetList();
  setInterval(function() {
    if (state.currentState === 'running' || state.currentState === 'halted') fetchStatus();
  }, 5000);
  setInterval(fetchContainerStatus, 5000);
})();
