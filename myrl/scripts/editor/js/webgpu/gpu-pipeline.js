// gpu-pipeline.js — WebGPU 初始化 + render pipeline + 内联 WGSL shader

var SHADER_CODE = /* wgsl */`
struct FrameUniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec3<f32>,
  _pad: f32,
};

struct LinkUniforms {
  model: mat4x4<f32>,
  color: vec4<f32>,
  linkId: u32,
  selected: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> link: LinkUniforms;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
};

@vertex fn vsMain(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> VSOut {
  var out: VSOut;
  let worldPos4 = link.model * vec4<f32>(position, 1.0);
  out.pos = frame.viewProj * worldPos4;
  out.worldPos = worldPos4.xyz;
  // 法线变换（忽略 scale，只取旋转部分的近似）
  let n = (link.model * vec4<f32>(normal, 0.0)).xyz;
  out.worldNormal = normalize(n);
  return out;
}

@fragment fn fsMain(inp: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(inp.worldNormal);
  // 顶部右前方方向光
  let L = normalize(vec3<f32>(0.3, 0.2, 0.8));
  let V = normalize(frame.cameraPos - inp.worldPos);
  let H = normalize(L + V);

  let ambient = 0.18;
  let diffuse = max(dot(N, L), 0.0) * 0.65;
  let spec = pow(max(dot(N, H), 0.0), 64.0) * 0.25;

  var base = link.color.rgb;
  if (link.selected == 1u) {
    base = base * 1.4 + vec3<f32>(0.1, 0.15, 0.2);
  }
  let col = base * (ambient + diffuse) + vec3<f32>(spec);
  // gamma
  let gamma = pow(col, vec3<f32>(1.0 / 2.2));
  return vec4<f32>(gamma, link.color.a);
}

// ── Picking pass shader ──
@fragment fn fsPickMain(inp: VSOut) -> @location(0) u32 {
  return link.linkId;
}
`;

export class GPUPipeline {
  constructor() {
    this.device = null;
    this.context = null;
    this.mainPipeline = null;
    this.pickPipeline = null;
    this.frameBindGroupLayout = null;
    this.linkBindGroupLayout = null;
    this.frameUniformBuf = null;
    this.depthTexture = null;
    this.pickTexture = null;
    this.pickReadBuf = null;
    this.format = 'bgra8unorm';
    this.width = 0;
    this.height = 0;
  }

  async init(canvas) {
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    // 优先高性能（独显），失败则 fallback 低功耗（集显），再失败不指定
    var adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
    if (!adapter) adapter = await navigator.gpu.requestAdapter({powerPreference: 'low-power'});
    if (!adapter) adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter — try chrome://flags → #enable-unsafe-webgpu');
    console.log('[WebGPU] adapter:', adapter.info || 'unknown');
    this.device = await adapter.requestDevice();
    this.context = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'premultiplied',
    });

    // Bind group layouts
    this.frameBindGroupLayout = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                  buffer: { type: 'uniform' } }],
    });
    this.linkBindGroupLayout = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                  buffer: { type: 'uniform' } }],
    });

    var pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.frameBindGroupLayout, this.linkBindGroupLayout],
    });

    var shaderModule = this.device.createShaderModule({ code: SHADER_CODE });
    var vertexLayout = {
      arrayStride: 24, // 6 floats
      attributes: [
        { format: 'float32x3', offset: 0, shaderLocation: 0 },  // position
        { format: 'float32x3', offset: 12, shaderLocation: 1 }, // normal
      ],
    };

    // Main render pipeline
    this.mainPipeline = this.device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: 'vsMain', buffers: [vertexLayout] },
      fragment: {
        module: shaderModule, entryPoint: 'fsMain',
        targets: [{ format: this.format }],
      },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
    });

    // Pick pipeline (writes u32 link ID)
    this.pickPipeline = this.device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: 'vsMain', buffers: [vertexLayout] },
      fragment: {
        module: shaderModule, entryPoint: 'fsPickMain',
        targets: [{ format: 'r32uint' }],
      },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
    });

    // Frame uniform buffer (viewProj mat4 + cameraPos vec3 + pad = 80 bytes)
    this.frameUniformBuf = this.device.createBuffer({
      size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.frameBindGroup = this.device.createBindGroup({
      layout: this.frameBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.frameUniformBuf } }],
    });

    // Pick readback buffer
    this.pickReadBuf = this.device.createBuffer({
      size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.resize(canvas.clientWidth, canvas.clientHeight);
  }

  resize(w, h) {
    if (w === this.width && h === this.height) return;
    this.width = w; this.height = h;
    var dpr = window.devicePixelRatio || 1;
    var pw = Math.floor(w * dpr), ph = Math.floor(h * dpr);

    if (this.depthTexture) this.depthTexture.destroy();
    this.depthTexture = this.device.createTexture({
      size: [pw, ph], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    if (this.pickTexture) this.pickTexture.destroy();
    this.pickTexture = this.device.createTexture({
      size: [pw, ph], format: 'r32uint',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
  }

  updateFrameUniforms(viewProj, cameraPos) {
    var data = new Float32Array(20); // 16 + 3 + 1 pad
    data.set(viewProj, 0);
    data[16] = cameraPos[0];
    data[17] = cameraPos[1];
    data[18] = cameraPos[2];
    this.device.queue.writeBuffer(this.frameUniformBuf, 0, data);
  }

  createLinkBuffer(modelMatrix, color, linkId, selected) {
    // 16 floats mat4 + 4 floats color + 4 u32 (linkId, selected, pad, pad) = 96 bytes
    var buf = this.device.createBuffer({
      size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    var data = new ArrayBuffer(96);
    var f32 = new Float32Array(data, 0, 20);
    var u32 = new Uint32Array(data, 80, 4);
    f32.set(modelMatrix, 0);
    f32[16] = color[0]; f32[17] = color[1]; f32[18] = color[2]; f32[19] = color[3];
    u32[0] = linkId; u32[1] = selected ? 1 : 0;
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  createLinkBindGroup(linkBuffer) {
    return this.device.createBindGroup({
      layout: this.linkBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: linkBuffer } }],
    });
  }

  createVertexBuffer(vertexData) {
    var buf = this.device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, vertexData);
    return buf;
  }
}
