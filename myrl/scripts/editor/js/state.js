// state.js — 中央状态存储
// 所有模块通过 import state from './state.js' 共享

const state = {
  // 全局
  selected: null,            // {type, name, data}
  currentState: 'stopped',
  consoleLines: 0,
  sse: null,
  fleetSSE: null,
  containerRunning: false,
  activeTarget: '',          // '' = local, 'server-id' = remote
  fleetServers: {},          // {server_id: {name, health, ssh_host, ...}}
  currentFleetOp: null,      // {server, op}
  deployTargetId: '',

  // Reward editor
  rewSchema: null,
  rewPipeline: null,
  rewPipelineName: '',
  rewDirty: false,
  rewRatios: [],
  rewTermColors: ['#00d4ff','#4ecdc4','#ff6b6b','#f0ad4e','#9b59b6','#3498db','#e74c3c','#2ecc71','#e67e22','#1abc9c'],

  // Obs graph editor
  obsBlocks: null,
  obsPipelineName: '',
  obsDirty: false,
  obsSelectedId: null,
  obsDragId: null,
  obsDragOff: { x: 0, y: 0 },
  obsPan: { x: 0, y: 0 },
  obsZoom: 1.0,
  obsConnecting: null,       // {fromId, fromPort} when dragging a connection
  obsPanning: false,         // middle/right button pan in progress

  // Algo editor
  algoPipeline: null,
  algoPipelineName: '',
  algoDirty: false,

  // Robot viewer
  robotName: '',
  robotLinks: null,
  robotSensors: [],
  selectedLink: null,
  sensorDirty: false,
};

export default state;
