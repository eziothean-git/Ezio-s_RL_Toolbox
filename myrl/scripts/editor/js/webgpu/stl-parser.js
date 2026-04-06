// stl-parser.js — 二进制 STL 解析器
// Binary STL: 80 byte header + uint32 num_triangles + N * 50 byte records
// 输出 interleaved float32: [px,py,pz, nx,ny,nz] * 3 vertices per triangle

export function parseBinarySTL(buffer) {
  var view = new DataView(buffer);
  var numTri = view.getUint32(80, true);
  // 6 floats per vertex (pos3 + normal3), 3 vertices per triangle
  var data = new Float32Array(numTri * 18);
  var offset = 84;
  for (var i = 0; i < numTri; i++) {
    var nx = view.getFloat32(offset, true);
    var ny = view.getFloat32(offset + 4, true);
    var nz = view.getFloat32(offset + 8, true);
    offset += 12;
    for (var v = 0; v < 3; v++) {
      var idx = (i * 3 + v) * 6;
      data[idx]     = view.getFloat32(offset, true);
      data[idx + 1] = view.getFloat32(offset + 4, true);
      data[idx + 2] = view.getFloat32(offset + 8, true);
      data[idx + 3] = nx;
      data[idx + 4] = ny;
      data[idx + 5] = nz;
      offset += 12;
    }
    offset += 2; // attribute byte count
  }
  return { vertices: data, triangleCount: numTri, vertexCount: numTri * 3 };
}
