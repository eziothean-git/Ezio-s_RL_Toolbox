// link-picker.js — GPU color-ID picking
// 在 offscreen r32uint texture 上渲染每个 link 的 ID，点击时读取像素

export class LinkPicker {
  constructor(gpu) {
    this._gpu = gpu;
    this._pending = false;
  }

  async pick(x, y, drawCallback) {
    if (this._pending) return null;
    this._pending = true;

    var gpu = this._gpu;
    var dpr = window.devicePixelRatio || 1;
    var px = Math.floor(x * dpr), py = Math.floor(y * dpr);

    // 渲染 pick pass
    var encoder = gpu.device.createCommandEncoder();
    var pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: gpu.pickTexture.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear', storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: gpu.depthTexture.createView(),
        depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
      },
    });
    pass.setPipeline(gpu.pickPipeline);
    pass.setBindGroup(0, gpu.frameBindGroup);
    drawCallback(pass);
    pass.end();

    // 拷贝目标像素到 readback buffer
    encoder.copyTextureToBuffer(
      { texture: gpu.pickTexture, origin: [px, py, 0] },
      { buffer: gpu.pickReadBuf, bytesPerRow: 256 },
      [1, 1, 1],
    );
    gpu.device.queue.submit([encoder.finish()]);

    // 读取
    await gpu.pickReadBuf.mapAsync(GPUMapMode.READ);
    var data = new Uint32Array(gpu.pickReadBuf.getMappedRange().slice(0, 4));
    var id = data[0];
    gpu.pickReadBuf.unmap();
    this._pending = false;

    return id > 0 ? id : null; // 0 = background
  }
}
