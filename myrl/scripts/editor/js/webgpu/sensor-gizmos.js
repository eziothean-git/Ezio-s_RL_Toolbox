// sensor-gizmos.js — 传感器可视化几何体生成器
// 生成线段顶点用于 line-list 渲染

// 深度相机视锥（4条边线 + 4条底边）
export function makeDepthCameraFrustum(fovDeg, aspect, range) {
  var halfH = Math.tan((fovDeg || 87) * Math.PI / 360) * (range || 2);
  var halfW = halfH * (aspect || 64/36);
  var d = range || 2;
  // 4 个远平面角点（相机局部坐标，Z 朝前）
  var corners = [
    [-halfW, -halfH, d], [ halfW, -halfH, d],
    [ halfW,  halfH, d], [-halfW,  halfH, d],
  ];
  var lines = [];
  // apex → corners
  for (var i = 0; i < 4; i++) {
    lines.push(0, 0, 0); lines.push(corners[i][0], corners[i][1], corners[i][2]);
  }
  // bottom edges
  for (var i = 0; i < 4; i++) {
    var j = (i + 1) % 4;
    lines.push(corners[i][0], corners[i][1], corners[i][2]);
    lines.push(corners[j][0], corners[j][1], corners[j][2]);
  }
  return new Float32Array(lines);
}

// 高度扫描射线扇（向下的射线阵列）
export function makeHeightScanRays(sizeX, sizeY, resolution, maxRange) {
  sizeX = sizeX || 0.3; sizeY = sizeY || 0.2;
  resolution = resolution || 0.05; maxRange = maxRange || 1.0;
  var lines = [];
  var nx = Math.max(1, Math.round(sizeX / resolution));
  var ny = Math.max(1, Math.round(sizeY / resolution));
  for (var ix = 0; ix <= nx; ix++) {
    for (var iy = 0; iy <= ny; iy++) {
      var x = -sizeX/2 + ix * resolution;
      var y = -sizeY/2 + iy * resolution;
      lines.push(x, y, 0);
      lines.push(x, y, -maxRange);
    }
  }
  return new Float32Array(lines);
}

// 力传感器坐标轴箭头（RGB = XYZ）
export function makeForceAxes(length) {
  length = length || 0.05;
  return {
    vertices: new Float32Array([
      0,0,0, length,0,0,  // X red
      0,0,0, 0,length,0,  // Y green
      0,0,0, 0,0,length,  // Z blue
    ]),
    colors: [[1,0,0], [0,1,0], [0,0,1]],
  };
}

// IMU 小坐标系（三轴短箭头）
export function makeIMUGizmo() {
  return makeForceAxes(0.03);
}

// 根据传感器类型返回对应几何体
export function makeSensorGizmo(sensor) {
  var type = sensor.type;
  var cfg = sensor.config || {};
  if (type === 'depth_camera') {
    return {
      type: 'lines',
      data: makeDepthCameraFrustum(cfg.fov_deg, (cfg.width||64)/(cfg.height||36), Math.min(cfg.max_range||2, 2)),
      color: [0.2, 0.6, 1.0],
    };
  }
  if (type === 'height_scanner') {
    var sz = cfg.size || [0.3, 0.2];
    return {
      type: 'lines',
      data: makeHeightScanRays(sz[0], sz[1], cfg.resolution, Math.min(cfg.max_range||1, 1.5)),
      color: [0.4, 1.0, 0.4],
    };
  }
  if (type === 'force_sensor' || type === 'contact') {
    return { type: 'axes', data: makeForceAxes(0.05), color: [1.0, 0.4, 0.2] };
  }
  if (type === 'imu') {
    return { type: 'axes', data: makeIMUGizmo(), color: [0.8, 0.8, 0.2] };
  }
  return null;
}
