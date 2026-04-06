// orbit-camera.js — 轨道相机控制器
// 左键旋转、滚轮缩放、中键/右键平移

export class OrbitCamera {
  constructor() {
    this.distance = 2.0;
    this.azimuth = 0.8;     // 水平角 (rad)
    this.elevation = 0.4;   // 仰角 (rad)
    this.target = [0, 0, 0.4]; // 看向机器人中心偏上
    this.fovY = Math.PI / 4;
    this.near = 0.01;
    this.far = 50.0;
    this._dragging = false;
    this._panning = false;
    this._lastX = 0;
    this._lastY = 0;
    this.dirty = true;
  }

  get eye() {
    var d = this.distance;
    var ce = Math.cos(this.elevation), se = Math.sin(this.elevation);
    var ca = Math.cos(this.azimuth), sa = Math.sin(this.azimuth);
    return [
      this.target[0] + d * ce * ca,
      this.target[1] + d * ce * sa,
      this.target[2] + d * se,
    ];
  }

  attach(canvas) {
    var self = this;
    canvas.addEventListener('mousedown', function(e) {
      if (e.button === 0) { self._dragging = true; }
      if (e.button === 1 || e.button === 2) { self._panning = true; }
      self._lastX = e.clientX;
      self._lastY = e.clientY;
      e.preventDefault();
    });
    canvas.addEventListener('mousemove', function(e) {
      var dx = e.clientX - self._lastX;
      var dy = e.clientY - self._lastY;
      self._lastX = e.clientX;
      self._lastY = e.clientY;
      if (self._dragging) {
        self.azimuth -= dx * 0.005;
        self.elevation += dy * 0.005;
        self.elevation = Math.max(-Math.PI/2 + 0.01, Math.min(Math.PI/2 - 0.01, self.elevation));
        self.dirty = true;
      }
      if (self._panning) {
        var panSpeed = self.distance * 0.002;
        var ca = Math.cos(self.azimuth), sa = Math.sin(self.azimuth);
        // 屏幕 X → 世界 right 方向
        self.target[0] += sa * dx * panSpeed;
        self.target[1] -= ca * dx * panSpeed;
        // 屏幕 Y → 世界 up 方向
        self.target[2] += dy * panSpeed;
        self.dirty = true;
      }
    });
    canvas.addEventListener('mouseup', function() {
      self._dragging = false;
      self._panning = false;
    });
    canvas.addEventListener('mouseleave', function() {
      self._dragging = false;
      self._panning = false;
    });
    canvas.addEventListener('wheel', function(e) {
      e.preventDefault();
      self.distance *= e.deltaY > 0 ? 1.1 : 0.9;
      self.distance = Math.max(0.1, Math.min(20, self.distance));
      self.dirty = true;
    }, {passive: false});
    canvas.addEventListener('contextmenu', function(e) { e.preventDefault(); });
  }

  fitBounds(min, max) {
    this.target = [
      (min[0] + max[0]) / 2,
      (min[1] + max[1]) / 2,
      (min[2] + max[2]) / 2,
    ];
    var dx = max[0]-min[0], dy = max[1]-min[1], dz = max[2]-min[2];
    this.distance = Math.max(dx, dy, dz) * 1.5;
    this.dirty = true;
  }
}
