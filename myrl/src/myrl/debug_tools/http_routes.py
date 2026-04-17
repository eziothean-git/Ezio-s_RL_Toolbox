"""调试工具 HTTP 路由 — 扩展 SignalServer 支持双向控制。

POST 端点供 curl / Editor WebUI / omni.ui 面板调用：
    /debug/force, /debug/mux/*, /debug/anchor/*, /debug/timescale, /debug/pause, /debug/step
GET 端点供状态查询：
    /debug/state, /debug/bodies, /debug/joints
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myrl.debug_tools.context import DebugContext


def register_debug_routes(signal_server, ctx: DebugContext) -> None:
    """向已有的 SignalServer 注册调试 HTTP 路由。

    通过 monkey-patch Handler 的 do_GET/do_POST 实现路由扩展。
    """
    handler_cls = signal_server._handler_cls if hasattr(signal_server, "_handler_cls") else None

    # 如果 SignalServer 没有暴露 handler class，退化为独立服务
    if handler_cls is None:
        print("[debug_tools] SignalServer handler class not accessible, "
              "HTTP debug routes not registered")
        return

    # 保存原始 handler 方法
    _orig_do_get = handler_cls.do_GET if hasattr(handler_cls, "do_GET") else None
    _orig_do_post = handler_cls.do_POST if hasattr(handler_cls, "do_POST") else None

    def _json_response(handler, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(body)

    def _read_json(handler) -> dict:
        length = int(handler.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = handler.rfile.read(length)
        return json.loads(raw)

    # ── GET 路由 ──────────────────────────────────────────────────────

    def do_GET(handler):
        path = handler.path.split("?")[0]

        if path == "/debug/state":
            _json_response(handler, ctx.state_snapshot())
            return

        if path == "/debug/bodies":
            data = {"bodies": ctx._body_names}
            if ctx._articulation is not None:
                try:
                    pos = ctx._articulation.data.body_pos_w[0].cpu().tolist()
                    data["positions"] = pos
                except Exception:
                    pass
            _json_response(handler, data)
            return

        if path == "/debug/joints":
            data = {
                "joints": ctx._joint_names,
                "num_actions": ctx._num_actions,
            }
            if ctx._articulation is not None:
                try:
                    data["positions"] = ctx._articulation.data.joint_pos[0].cpu().tolist()
                except Exception:
                    pass
            _json_response(handler, data)
            return

        # 非调试路由，交给原始 handler
        if _orig_do_get is not None:
            _orig_do_get(handler)
        else:
            handler.send_error(404)

    # ── POST 路由 ─────────────────────────────────────────────────────

    def do_POST(handler):
        path = handler.path.split("?")[0]

        # ── 时间控制 ──────────────────────────────────────────────
        if path == "/debug/timescale":
            body = _read_json(handler)
            plugin = ctx.get_plugin("time_scale")
            if plugin:
                plugin.set_scale(ctx, float(body.get("scale", 1.0)))
            _json_response(handler, {"ok": True, "time_scale": ctx.time_scale})
            return

        if path == "/debug/pause":
            plugin = ctx.get_plugin("time_scale")
            if plugin:
                plugin.toggle_pause(ctx)
            _json_response(handler, {"ok": True, "paused": ctx.paused})
            return

        if path == "/debug/step":
            plugin = ctx.get_plugin("time_scale")
            if plugin:
                plugin.single_step(ctx)
            _json_response(handler, {"ok": True})
            return

        # ── 力施加 ────────────────────────────────────────────────
        if path == "/debug/force":
            body = _read_json(handler)
            plugin = ctx.get_plugin("force_applicator")
            if plugin:
                plugin.set_force(
                    ctx,
                    env_id=int(body.get("env_id", 0)),
                    body_id=int(body.get("body_id", 0)),
                    force_w=body.get("force", [0, 0, 0]),
                    torque_w=body.get("torque", [0, 0, 0]),
                )
            _json_response(handler, {"ok": True})
            return

        if path == "/debug/force/clear":
            plugin = ctx.get_plugin("force_applicator")
            if plugin:
                plugin.clear_forces(ctx)
            _json_response(handler, {"ok": True})
            return

        # ── 动作 MUX ──────────────────────────────────────────────
        if path == "/debug/mux/set":
            body = _read_json(handler)
            plugin = ctx.get_plugin("action_mux")
            if plugin:
                if "values" in body:
                    plugin.set_all_overrides(
                        ctx, int(body.get("env_id", 0)), body["values"]
                    )
                else:
                    plugin.set_override(
                        ctx,
                        env_id=int(body.get("env_id", 0)),
                        joint_idx=int(body.get("joint_idx", 0)),
                        value=float(body.get("value", 0)),
                    )
            _json_response(handler, {"ok": True})
            return

        if path == "/debug/mux/clear":
            body = _read_json(handler)
            plugin = ctx.get_plugin("action_mux")
            if plugin:
                env_id = int(body.get("env_id", 0))
                joint_idx = body.get("joint_idx")
                if joint_idx is not None:
                    joint_idx = int(joint_idx)
                plugin.clear_override(ctx, env_id, joint_idx)
            _json_response(handler, {"ok": True})
            return

        # ── 锚点 ─────────────────────────────────────────────────
        if path == "/debug/anchor/toggle":
            body = _read_json(handler)
            plugin = ctx.get_plugin("body_anchor")
            if plugin:
                plugin.toggle_anchor(
                    ctx,
                    env_id=int(body.get("env_id", 0)),
                    body_id=int(body.get("body_id", 0)),
                )
            _json_response(handler, {"ok": True, "anchor_active": ctx.anchor_active})
            return

        # ── 可视化 ────────────────────────────────────────────────
        if path == "/debug/viz":
            body = _read_json(handler)
            if "contact_forces" in body:
                ctx.viz_contact_forces = bool(body["contact_forces"])
            if "trajectories" in body:
                ctx.viz_trajectories = bool(body["trajectories"])
            _json_response(handler, {
                "ok": True,
                "contact_forces": ctx.viz_contact_forces,
                "trajectories": ctx.viz_trajectories,
            })
            return

        # 非调试路由
        if _orig_do_post is not None:
            _orig_do_post(handler)
        else:
            handler.send_error(404)

    # ── CORS preflight ────────────────────────────────────────────────
    _orig_do_options = getattr(handler_cls, "do_OPTIONS", None)

    def do_OPTIONS(handler):
        handler.send_response(204)
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        handler.send_header("Access-Control-Allow-Headers", "Content-Type")
        handler.end_headers()

    # 替换 handler 方法
    handler_cls.do_GET = do_GET
    handler_cls.do_POST = do_POST
    handler_cls.do_OPTIONS = do_OPTIONS
