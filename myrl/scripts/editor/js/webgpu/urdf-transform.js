// urdf-transform.js — URDF joint tree → 4x4 world transforms (FK)
// 输入 API 返回的 links/joints/tree/transforms，输出每个 link 的 Float32Array(16)

export function buildTransformMap(robotData) {
  // 后端已经计算了 rest-pose transforms（column-major float[16]）
  var tfs = robotData.transforms || {};
  var result = {};
  for (var linkName in tfs) {
    result[linkName] = new Float32Array(tfs[linkName]);
  }
  return result;
}

// 从 link 数据构建 link→mesh 映射
export function buildMeshMap(robotData) {
  var map = {};
  (robotData.links || []).forEach(function(l) {
    if (l.mesh) {
      map[l.name] = l.mesh;
    }
  });
  return map;
}

// 4x4 矩阵工具（column-major）
export function mat4Identity() {
  return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
}

export function mat4Multiply(a, b) {
  var r = new Float32Array(16);
  for (var col = 0; col < 4; col++) {
    for (var row = 0; row < 4; row++) {
      var s = 0;
      for (var k = 0; k < 4; k++) {
        s += a[k * 4 + row] * b[col * 4 + k];
      }
      r[col * 4 + row] = s;
    }
  }
  return r;
}

export function mat4Translate(x, y, z) {
  var m = mat4Identity();
  m[12] = x; m[13] = y; m[14] = z;
  return m;
}

export function mat4Scale(sx, sy, sz) {
  var m = mat4Identity();
  m[0] = sx; m[5] = sy; m[10] = sz;
  return m;
}

export function mat4Perspective(fovY, aspect, near, far) {
  var f = 1.0 / Math.tan(fovY / 2);
  var nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

export function mat4LookAt(eye, center, up) {
  var zx = eye[0]-center[0], zy = eye[1]-center[1], zz = eye[2]-center[2];
  var zl = Math.sqrt(zx*zx+zy*zy+zz*zz);
  zx/=zl; zy/=zl; zz/=zl;
  var xx = up[1]*zz-up[2]*zy, xy = up[2]*zx-up[0]*zz, xz = up[0]*zy-up[1]*zx;
  var xl = Math.sqrt(xx*xx+xy*xy+xz*xz);
  xx/=xl; xy/=xl; xz/=xl;
  var yx = zy*xz-zz*xy, yy = zz*xx-zx*xz, yz = zx*xy-zy*xx;
  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx*eye[0]+xy*eye[1]+xz*eye[2]),
    -(yx*eye[0]+yy*eye[1]+yz*eye[2]),
    -(zx*eye[0]+zy*eye[1]+zz*eye[2]),
    1,
  ]);
}
