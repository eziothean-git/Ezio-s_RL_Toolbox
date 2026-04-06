"""obs_schema — Obs Pipeline v2 block-graph schema, loader, validator, compiler.

v2 将 obs pipeline 建模为 DAG：
  obs(数据源) → modifier(变换链) → encoder(编码器) → group(输出组)

每个 block 自包含所有配置，outputs 列表定义数据流向。
modifier 是独立节点（Blender modifier stack 风格），中间信号可通过 DataBus 检查。

用法:
    pipeline = load_obs_pipeline_v2("g1_parkour_amp_v2.yaml")
    errors = validate_pipeline(pipeline)
    obs_cfg, encoder_cfgs, history_cfg = PipelineCompiler(pipeline).compile()
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]


# ── Block Registry ────────────────────────────────────────────────────────────

_BLOCK_REGISTRY: dict[tuple[str, str], type[BlockHandler]] = {}


def register_block_type(block_type: str, kind: str):
    """装饰器：注册 (type, kind) → BlockHandler 子类。"""
    def decorator(cls):
        _BLOCK_REGISTRY[(block_type, kind)] = cls
        return cls
    return decorator


def get_handler(block_type: str, kind: str) -> type[BlockHandler] | None:
    return _BLOCK_REGISTRY.get((block_type, kind))


# ── Core Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class BlockDef:
    """一个 block 节点的完整定义。"""
    id: str
    type: str           # obs | modifier | encoder | group | transform
    kind: str           # subtype（mdp, scale, conv2d, policy, ...）
    config: dict        # type/kind 特有的参数
    outputs: list[str] = field(default_factory=list)

    def get(self, key: str, default=None):
        return self.config.get(key, default)


@dataclass
class ObsPipelineV2:
    """v2 obs pipeline 的完整表示。"""
    schema: str = "obs_pipeline_v2"
    name: str = ""
    version: str = "2.0.0"
    description: str = ""
    blocks: dict[str, BlockDef] = field(default_factory=dict)

    def get_blocks_by_type(self, block_type: str) -> list[BlockDef]:
        return [b for b in self.blocks.values() if b.type == block_type]

    def get_upstream(self, block_id: str) -> list[BlockDef]:
        """返回所有 outputs 包含 block_id 的上游节点。"""
        return [b for b in self.blocks.values() if block_id in b.outputs]

    def get_upstream_recursive(self, block_id: str) -> list[BlockDef]:
        """递归收集所有上游节点（BFS）。"""
        visited = set()
        queue = [block_id]
        result = []
        while queue:
            cur = queue.pop(0)
            for b in self.blocks.values():
                if cur in b.outputs and b.id not in visited:
                    visited.add(b.id)
                    result.append(b)
                    queue.append(b.id)
        return result

    def topo_sort(self) -> list[BlockDef]:
        """拓扑排序（Kahn's algorithm）。若有环返回部分排序。"""
        in_degree: dict[str, int] = {bid: 0 for bid in self.blocks}
        for b in self.blocks.values():
            for out in b.outputs:
                if out in in_degree:
                    in_degree[out] += 1
        queue = [bid for bid, d in in_degree.items() if d == 0]
        order = []
        while queue:
            bid = queue.pop(0)
            order.append(self.blocks[bid])
            for out in self.blocks[bid].outputs:
                if out in in_degree:
                    in_degree[out] -= 1
                    if in_degree[out] == 0:
                        queue.append(out)
        return order


# ── BlockHandler Base ─────────────────────────────────────────────────────────

class BlockHandler:
    """每个 (type, kind) 对应一个 handler，负责验证和编译。"""

    @staticmethod
    def validate(block: BlockDef) -> list[str]:
        """返回错误列表（空=通过）。"""
        return []

    @staticmethod
    def config_keys() -> list[str]:
        """此 handler 识别的 config key 列表（用于 UI schema 生成）。"""
        return []


# ── Obs Handlers ──────────────────────────────────────────────────────────────

@register_block_type("obs", "mdp")
class ObsMdpHandler(BlockHandler):
    @staticmethod
    def validate(block: BlockDef) -> list[str]:
        errors = []
        if not block.get("func"):
            errors.append(f"{block.id}: obs/mdp requires 'func'")
        return errors

    @staticmethod
    def config_keys():
        return ["func", "params", "shape"]


@register_block_type("obs", "sensor")
class ObsSensorHandler(BlockHandler):
    @staticmethod
    def validate(block: BlockDef) -> list[str]:
        errors = []
        if not block.get("func"):
            errors.append(f"{block.id}: obs/sensor requires 'func'")
        return errors

    @staticmethod
    def config_keys():
        return ["func", "params", "shape"]


@register_block_type("obs", "custom")
class ObsCustomHandler(BlockHandler):
    @staticmethod
    def validate(block: BlockDef) -> list[str]:
        if not block.get("func"):
            return [f"{block.id}: obs/custom requires 'func'"]
        return []

    @staticmethod
    def config_keys():
        return ["func", "params", "shape"]


# ── Modifier Handlers ─────────────────────────────────────────────────────────

@register_block_type("modifier", "scale")
class ModScaleHandler(BlockHandler):
    @staticmethod
    def validate(block: BlockDef) -> list[str]:
        if block.get("factor") is None:
            return [f"{block.id}: modifier/scale requires 'factor'"]
        return []

    @staticmethod
    def config_keys():
        return ["factor"]


@register_block_type("modifier", "noise")
class ModNoiseHandler(BlockHandler):
    @staticmethod
    def validate(block: BlockDef) -> list[str]:
        nt = block.get("noise_type")
        if nt not in ("gaussian", "uniform"):
            return [f"{block.id}: modifier/noise requires noise_type in (gaussian, uniform)"]
        return []

    @staticmethod
    def config_keys():
        return ["noise_type", "std", "min", "max"]


@register_block_type("modifier", "clip")
class ModClipHandler(BlockHandler):
    @staticmethod
    def validate(block: BlockDef) -> list[str]:
        if block.get("min") is None or block.get("max") is None:
            return [f"{block.id}: modifier/clip requires 'min' and 'max'"]
        return []

    @staticmethod
    def config_keys():
        return ["min", "max"]


@register_block_type("modifier", "normalize")
class ModNormalizeHandler(BlockHandler):
    @staticmethod
    def validate(block: BlockDef) -> list[str]:
        return []

    @staticmethod
    def config_keys():
        return ["method", "window", "min_std"]


@register_block_type("modifier", "history")
class ModHistoryHandler(BlockHandler):
    @staticmethod
    def validate(block: BlockDef) -> list[str]:
        length = block.get("length")
        if length is None or not isinstance(length, int) or length < 1:
            return [f"{block.id}: modifier/history requires 'length' (int >= 1)"]
        return []

    @staticmethod
    def config_keys():
        return ["length", "flatten"]


@register_block_type("modifier", "remap")
class ModRemapHandler(BlockHandler):
    @staticmethod
    def config_keys():
        return ["mapping"]


# ── Encoder Handlers ──────────────────────────────────────────────────────────

@register_block_type("encoder", "conv2d")
class EncConv2dHandler(BlockHandler):
    @staticmethod
    def validate(block: BlockDef) -> list[str]:
        if not block.get("output_size"):
            return [f"{block.id}: encoder/conv2d requires 'output_size'"]
        return []

    @staticmethod
    def config_keys():
        return ["output_size", "channels", "kernel_sizes", "strides",
                "paddings", "hidden_sizes", "nonlinearity", "use_maxpool",
                "takeout_input"]


@register_block_type("encoder", "mlp")
class EncMlpHandler(BlockHandler):
    @staticmethod
    def validate(block: BlockDef) -> list[str]:
        if not block.get("output_size"):
            return [f"{block.id}: encoder/mlp requires 'output_size'"]
        return []

    @staticmethod
    def config_keys():
        return ["output_size", "hidden_sizes", "nonlinearity"]


@register_block_type("encoder", "transformer")
class EncTransformerHandler(BlockHandler):
    @staticmethod
    def validate(block: BlockDef) -> list[str]:
        if not block.get("output_size"):
            return [f"{block.id}: encoder/transformer requires 'output_size'"]
        return []

    @staticmethod
    def config_keys():
        return ["output_size", "num_heads", "d_model", "num_layers", "hidden_sizes"]


# ── Group Handlers ────────────────────────────────────────────────────────────

_KNOWN_GROUPS = {"policy", "critic", "amp_policy", "amp_reference", "estimator", "custom"}


@register_block_type("group", "policy")
class GroupPolicyHandler(BlockHandler):
    @staticmethod
    def config_keys():
        return ["enable_corruption", "concatenate"]


@register_block_type("group", "critic")
class GroupCriticHandler(BlockHandler):
    @staticmethod
    def config_keys():
        return ["enable_corruption", "concatenate"]


# 注册其余 group kind
for _gk in ("amp_policy", "amp_reference", "estimator", "custom"):
    @register_block_type("group", _gk)
    class _GroupHandler(BlockHandler):
        @staticmethod
        def config_keys():
            return ["enable_corruption", "concatenate"]


# ── Load / Save ──────────────────────────────────────────────────────────────

def load_obs_pipeline_v2(path: str | Path) -> ObsPipelineV2:
    """从 YAML 文件加载 v2 obs pipeline。"""
    if _yaml is None:
        raise RuntimeError("PyYAML is required for obs_schema")
    path = Path(path)
    raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _parse_raw(raw)


def parse_obs_pipeline_v2(raw: dict) -> ObsPipelineV2:
    """从已解析的 dict 构造 ObsPipelineV2（供 API 端点使用）。"""
    return _parse_raw(raw)


def _parse_raw(raw: dict) -> ObsPipelineV2:
    pipeline = ObsPipelineV2(
        schema=raw.get("schema", "obs_pipeline_v2"),
        name=raw.get("name", ""),
        version=raw.get("version", "2.0.0"),
        description=raw.get("description", ""),
    )
    for item in raw.get("blocks", []):
        if isinstance(item, dict):
            # 列表格式: [{id: ..., type: ...}, ...]
            bid = item.get("id", "")
        else:
            continue
        btype = item.get("type", "")
        bkind = item.get("kind", "")
        outputs = item.get("outputs", [])
        if isinstance(outputs, str):
            outputs = [outputs]
        # 收集 config：除了 id/type/kind/outputs 之外的所有 key
        config = {k: v for k, v in item.items() if k not in ("id", "type", "kind", "outputs")}
        pipeline.blocks[bid] = BlockDef(
            id=bid, type=btype, kind=bkind, config=config, outputs=outputs,
        )
    return pipeline


def save_obs_pipeline_v2(pipeline: ObsPipelineV2, path: str | Path) -> None:
    """保存 v2 obs pipeline 为 YAML。"""
    if _yaml is None:
        raise RuntimeError("PyYAML is required for obs_schema")
    path = Path(path)
    raw = {
        "schema": pipeline.schema,
        "name": pipeline.name,
        "version": pipeline.version,
    }
    if pipeline.description:
        raw["description"] = pipeline.description
    blocks_list = []
    for b in pipeline.blocks.values():
        item: dict[str, Any] = {"id": b.id, "type": b.type, "kind": b.kind}
        item.update(b.config)
        if b.outputs:
            item["outputs"] = b.outputs
        blocks_list.append(item)
    raw["blocks"] = blocks_list
    text = _yaml.safe_dump(raw, allow_unicode=True, default_flow_style=False, sort_keys=False)
    path.write_text(text, encoding="utf-8")


# ── v1 → v2 Conversion ──────────────────────────────────────────────────────

def convert_v1_to_v2(raw: dict) -> ObsPipelineV2:
    """将 v1 扁平 obs YAML 转为 v2 block graph。

    v1 格式:
        policy:
          joint_pos:
            func: mdp.joint_pos_rel
            scale: 1.0
            history_length: 1
    """
    pipeline = ObsPipelineV2(
        name=raw.get("name", ""),
        version=raw.get("version", "1.0.0"),
        description=raw.get("description", ""),
    )
    meta_keys = {"name", "version", "description"}
    for group_name, terms in raw.items():
        if group_name in meta_keys or not isinstance(terms, dict):
            continue
        # 创建 group 节点
        group_id = f"{group_name}_group"
        gkind = group_name if group_name in _KNOWN_GROUPS else "custom"
        pipeline.blocks[group_id] = BlockDef(
            id=group_id, type="group", kind=gkind, config={}, outputs=[],
        )
        # 创建每个 term → modifier chain → group
        for term_name, term_cfg in terms.items():
            if not isinstance(term_cfg, dict):
                continue
            # obs 节点
            obs_id = term_name
            obs_config = {"func": term_cfg.get("func", "")}
            if "params" in term_cfg:
                obs_config["params"] = term_cfg["params"]

            # 构建 modifier 链
            chain_tip = obs_id  # 当前链尾
            modifier_blocks: list[BlockDef] = []

            scale = term_cfg.get("scale")
            if scale is not None and scale != 1.0:
                mid = f"{term_name}_scale"
                modifier_blocks.append(BlockDef(
                    id=mid, type="modifier", kind="scale",
                    config={"factor": scale}, outputs=[],
                ))
                chain_tip = mid

            noise = term_cfg.get("noise")
            if noise:
                mid = f"{term_name}_noise"
                modifier_blocks.append(BlockDef(
                    id=mid, type="modifier", kind="noise",
                    config=noise if isinstance(noise, dict) else {"noise_type": "gaussian", "std": 0.01},
                    outputs=[],
                ))
                chain_tip = mid

            history = term_cfg.get("history_length", term_cfg.get("history", {}).get("length", 0) if isinstance(term_cfg.get("history"), dict) else 0)
            if isinstance(history, int) and history > 1:
                mid = f"{term_name}_history"
                flatten = term_cfg.get("flatten_history_dim", term_cfg.get("history", {}).get("flatten", True) if isinstance(term_cfg.get("history"), dict) else True)
                modifier_blocks.append(BlockDef(
                    id=mid, type="modifier", kind="history",
                    config={"length": history, "flatten": flatten}, outputs=[],
                ))
                chain_tip = mid

            # 连接链: obs → mod1 → mod2 → ... → group
            prev = obs_id
            for mb in modifier_blocks:
                pipeline.blocks[prev].outputs.append(mb.id) if prev in pipeline.blocks else None
                pipeline.blocks[mb.id] = mb
                prev = mb.id
            # 链尾 → group
            if chain_tip == obs_id:
                obs_outputs = [group_id]
            else:
                pipeline.blocks[chain_tip].outputs.append(group_id)
                obs_outputs = [modifier_blocks[0].id] if modifier_blocks else [group_id]

            pipeline.blocks[obs_id] = BlockDef(
                id=obs_id, type="obs", kind="mdp", config=obs_config, outputs=obs_outputs,
            )

    return pipeline


def is_v2(raw: dict) -> bool:
    """检测 YAML dict 是否为 v2 格式。"""
    return raw.get("schema") == "obs_pipeline_v2"


# ── Validation ───────────────────────────────────────────────────────────────

# 合法的边规则: from_type → allowed to_types
_EDGE_RULES: dict[str, set[str]] = {
    "obs": {"modifier", "encoder", "group"},
    "modifier": {"modifier", "encoder", "group"},
    "encoder": {"modifier", "encoder", "group"},
    "transform": {"modifier", "group"},
    "group": set(),  # terminal
}


def validate_pipeline(pipeline: ObsPipelineV2) -> list[str]:
    """验证 pipeline DAG。返回错误列表（空=通过）。"""
    errors = []

    # 1. block id 唯一性（dataclass dict key 已保证）

    # 2. outputs 引用存在性
    for b in pipeline.blocks.values():
        for out in b.outputs:
            if out not in pipeline.blocks:
                errors.append(f"{b.id}: output '{out}' not found in blocks")

    # 3. 边类型规则
    for b in pipeline.blocks.values():
        allowed = _EDGE_RULES.get(b.type, set())
        for out in b.outputs:
            target = pipeline.blocks.get(out)
            if target and target.type not in allowed:
                errors.append(f"{b.id}({b.type}) → {out}({target.type}): edge not allowed")

    # 4. group 节点不应有 outputs
    for b in pipeline.blocks.values():
        if b.type == "group" and b.outputs:
            errors.append(f"{b.id}: group blocks should not have outputs")

    # 5. 至少一个 group
    groups = pipeline.get_blocks_by_type("group")
    if not groups:
        errors.append("pipeline has no group blocks")

    # 6. 环检测（topo sort 应包含所有节点）
    sorted_blocks = pipeline.topo_sort()
    if len(sorted_blocks) < len(pipeline.blocks):
        errors.append("pipeline contains cycles")

    # 7. 孤立节点（无 outputs 且非 group）
    for b in pipeline.blocks.values():
        if b.type != "group" and not b.outputs:
            errors.append(f"{b.id}({b.type}/{b.kind}): orphan block (no outputs)")

    # 8. 每个 handler 的自定义验证
    for b in pipeline.blocks.values():
        handler = get_handler(b.type, b.kind)
        if handler:
            errors.extend(handler.validate(b))
        elif b.type not in ("group",):
            errors.append(f"{b.id}: unknown block type ({b.type}/{b.kind})")

    return errors


# ── Compiler ─────────────────────────────────────────────────────────────────

class PipelineCompiler:
    """将 validated ObsPipelineV2 编译为 instinctlab/instinct_rl 可用的配置 dict。

    输出:
        obs_cfg:       {group_name: {term_name: {func, params, scale, noise, clip}}}
        encoder_cfgs:  {encoder_id: {class_name, component_names, output_size, ...}}
        history_cfg:   {group_name: {term_name: history_length}}
    """

    def __init__(self, pipeline: ObsPipelineV2):
        self._p = pipeline

    def compile(self) -> tuple[dict, dict, dict]:
        obs_cfg: dict[str, OrderedDict] = {}
        encoder_cfgs: dict[str, dict] = {}
        history_cfg: dict[str, dict] = {}

        for group in self._p.get_blocks_by_type("group"):
            group_name = group.kind if group.kind != "custom" else group.id.replace("_group", "")
            obs_cfg[group_name] = OrderedDict()

            # 收集所有到达此 group 的上游 obs 节点及其 modifier 链
            upstream_all = self._p.get_upstream_recursive(group.id)

            # 找出所有 obs 节点
            obs_blocks = [b for b in upstream_all if b.type == "obs"]
            for ob in obs_blocks:
                term_name = ob.id
                term_cfg = {"func": ob.get("func", "")}
                if ob.get("params"):
                    term_cfg["params"] = ob.get("params")

                # 追踪从这个 obs 到此 group 的 modifier 链
                modifiers = self._trace_modifiers(ob.id, group.id)
                for mod in modifiers:
                    if mod.kind == "scale":
                        term_cfg["scale"] = mod.get("factor", 1.0)
                    elif mod.kind == "noise":
                        term_cfg["noise"] = {
                            "type": mod.get("noise_type"),
                            "std": mod.get("std"),
                            "min": mod.get("min"),
                            "max": mod.get("max"),
                        }
                    elif mod.kind == "clip":
                        term_cfg["clip"] = (mod.get("min"), mod.get("max"))
                    elif mod.kind == "history":
                        length = mod.get("length", 1)
                        flatten = mod.get("flatten", True)
                        term_cfg["history_length"] = length
                        term_cfg["flatten_history_dim"] = flatten
                        history_cfg.setdefault(group_name, {})[term_name] = length

                obs_cfg[group_name][term_name] = term_cfg

        # 编译 encoder 配置
        for enc in self._p.get_blocks_by_type("encoder"):
            # component_names = 上游 obs 节点的 id（穿透 modifier）
            upstream = self._p.get_upstream(enc.id)
            component_names = []
            for up in upstream:
                if up.type == "obs":
                    component_names.append(up.id)
                elif up.type == "modifier":
                    # 回溯到 obs 源
                    src = self._trace_to_obs_source(up.id)
                    if src:
                        component_names.append(src.id)

            enc_cfg = {
                "class_name": self._map_encoder_class(enc.kind),
                "component_names": component_names,
                "output_size": enc.get("output_size", 128),
            }
            # 添加 kind 特有参数
            for key in ("hidden_sizes", "channels", "kernel_sizes", "strides",
                        "paddings", "nonlinearity", "use_maxpool", "takeout_input",
                        "num_heads", "d_model", "num_layers"):
                val = enc.get(key)
                if val is not None:
                    enc_cfg[key] = val

            encoder_cfgs[enc.id] = enc_cfg

        return obs_cfg, encoder_cfgs, history_cfg

    def _trace_modifiers(self, obs_id: str, group_id: str) -> list[BlockDef]:
        """追踪从 obs_id 到 group_id 路径上的所有 modifier 节点（有序）。"""
        path: list[BlockDef] = []
        visited = set()

        def dfs(cur_id: str) -> bool:
            if cur_id == group_id:
                return True
            if cur_id in visited:
                return False
            visited.add(cur_id)
            block = self._p.blocks.get(cur_id)
            if block is None:
                return False
            for out in block.outputs:
                if dfs(out):
                    target = self._p.blocks.get(out)
                    if target and target.type == "modifier":
                        path.append(target)
                    return True
            return False

        dfs(obs_id)
        # path 是从 group 往回收集的，反转为 obs→group 顺序
        path.reverse()
        return path

    def _trace_to_obs_source(self, block_id: str) -> BlockDef | None:
        """从 modifier 回溯到 obs 数据源。"""
        visited = set()
        cur = block_id
        while cur and cur not in visited:
            visited.add(cur)
            upstream = self._p.get_upstream(cur)
            for up in upstream:
                if up.type == "obs":
                    return up
                if up.type == "modifier":
                    cur = up.id
                    break
            else:
                return None
        return None

    @staticmethod
    def _map_encoder_class(kind: str) -> str:
        return {
            "conv2d": "Conv2dHeadModel",
            "mlp": "MlpModel",
            "transformer": "TransformerHeadModel",
        }.get(kind, kind)
