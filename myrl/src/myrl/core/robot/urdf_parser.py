"""轻量 URDF 解析器——提取 link 树 + mesh 引用，用于 Editor 3D viewer。

仅使用 stdlib xml.etree，不依赖任何第三方库。
提取：link 名、joint 父子/类型/origin、visual mesh 文件名+origin。
跳过：惯性、碰撞、joint limits、dynamics、材质。
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class URDFLink:
    name: str
    mesh_file: str | None = None           # 相对于 meshes/ 目录
    visual_origin_xyz: tuple = (0, 0, 0)
    visual_origin_rpy: tuple = (0, 0, 0)


@dataclass
class URDFJoint:
    name: str
    type: str                               # revolute / fixed / prismatic / continuous / floating
    parent_link: str
    child_link: str
    origin_xyz: tuple = (0, 0, 0)
    origin_rpy: tuple = (0, 0, 0)
    axis: tuple = (0, 0, 1)


@dataclass
class URDFModel:
    name: str
    links: dict[str, URDFLink] = field(default_factory=dict)
    joints: dict[str, URDFJoint] = field(default_factory=dict)

    def link_tree(self) -> dict[str, list[dict]]:
        """返回 {parent_link: [{child, joint, type, origin_xyz, origin_rpy}, ...]}。"""
        tree: dict[str, list[dict]] = {}
        for j in self.joints.values():
            tree.setdefault(j.parent_link, []).append({
                "child": j.child_link,
                "joint": j.name,
                "type": j.type,
                "origin_xyz": list(j.origin_xyz),
                "origin_rpy": list(j.origin_rpy),
            })
        return tree

    def root_link(self) -> str | None:
        """找到根 link（不是任何 joint 的 child）。"""
        children = {j.child_link for j in self.joints.values()}
        for name in self.links:
            if name not in children:
                return name
        return None

    def world_transforms(self) -> dict[str, list[float]]:
        """计算每个 link 的世界系 4x4 矩阵（rest pose, 所有 joint 角度=0）。
        返回 {link_name: [16 floats, column-major]}。"""
        result = {}
        root = self.root_link()
        if not root:
            return result

        # BFS
        identity = _mat4_identity()
        queue = [(root, identity)]
        result[root] = identity

        tree = self.link_tree()
        while queue:
            parent, parent_tf = queue.pop(0)
            for child_info in tree.get(parent, []):
                xyz = child_info["origin_xyz"]
                rpy = child_info["origin_rpy"]
                joint_tf = _mat4_from_xyz_rpy(xyz, rpy)
                world_tf = _mat4_mul(parent_tf, joint_tf)
                result[child_info["child"]] = world_tf

                # visual origin 偏移（如果有）
                child_link = self.links.get(child_info["child"])
                if child_link and (child_link.visual_origin_xyz != (0, 0, 0) or
                                   child_link.visual_origin_rpy != (0, 0, 0)):
                    vis_tf = _mat4_from_xyz_rpy(
                        list(child_link.visual_origin_xyz),
                        list(child_link.visual_origin_rpy),
                    )
                    result[child_info["child"] + "/__visual__"] = _mat4_mul(world_tf, vis_tf)

                queue.append((child_info["child"], world_tf))
        return result

    def to_dict(self) -> dict:
        """序列化为 JSON-friendly dict（供 API 输出）。"""
        return {
            "robot_name": self.name,
            "links": [
                {
                    "name": l.name,
                    "mesh": l.mesh_file,
                    "has_mesh": l.mesh_file is not None,
                    "visual_origin_xyz": list(l.visual_origin_xyz),
                    "visual_origin_rpy": list(l.visual_origin_rpy),
                }
                for l in self.links.values()
            ],
            "joints": [
                {
                    "name": j.name,
                    "type": j.type,
                    "parent": j.parent_link,
                    "child": j.child_link,
                    "origin_xyz": list(j.origin_xyz),
                    "origin_rpy": list(j.origin_rpy),
                    "axis": list(j.axis),
                }
                for j in self.joints.values()
            ],
            "tree": self.link_tree(),
            "root": self.root_link(),
        }


def parse_urdf(path: str | Path) -> URDFModel:
    """解析 URDF 文件，返回 URDFModel。"""
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    name = root.get("name", path.stem)
    model = URDFModel(name=name)

    for link_el in root.findall("link"):
        link_name = link_el.get("name", "")
        mesh_file = None
        vis_xyz = (0, 0, 0)
        vis_rpy = (0, 0, 0)

        visual = link_el.find("visual")
        if visual is not None:
            origin = visual.find("origin")
            if origin is not None:
                vis_xyz = _parse_vec(origin.get("xyz", "0 0 0"))
                vis_rpy = _parse_vec(origin.get("rpy", "0 0 0"))

            geom = visual.find("geometry")
            if geom is not None:
                mesh = geom.find("mesh")
                if mesh is not None:
                    filename = mesh.get("filename", "")
                    # 提取文件名（去掉 package:// 等前缀）
                    if "/" in filename:
                        mesh_file = filename.rsplit("/", 1)[-1]
                    else:
                        mesh_file = filename

        model.links[link_name] = URDFLink(
            name=link_name,
            mesh_file=mesh_file,
            visual_origin_xyz=vis_xyz,
            visual_origin_rpy=vis_rpy,
        )

    for joint_el in root.findall("joint"):
        joint_name = joint_el.get("name", "")
        joint_type = joint_el.get("type", "fixed")

        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        parent_link = parent_el.get("link", "") if parent_el is not None else ""
        child_link = child_el.get("link", "") if child_el is not None else ""

        origin_xyz = (0, 0, 0)
        origin_rpy = (0, 0, 0)
        origin_el = joint_el.find("origin")
        if origin_el is not None:
            origin_xyz = _parse_vec(origin_el.get("xyz", "0 0 0"))
            origin_rpy = _parse_vec(origin_el.get("rpy", "0 0 0"))

        axis_vec = (0, 0, 1)
        axis_el = joint_el.find("axis")
        if axis_el is not None:
            axis_vec = _parse_vec(axis_el.get("xyz", "0 0 1"))

        model.joints[joint_name] = URDFJoint(
            name=joint_name,
            type=joint_type,
            parent_link=parent_link,
            child_link=child_link,
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            axis=axis_vec,
        )

    return model


# ── 内部工具函数 ────────────────────────────────────────────────────────────

def _parse_vec(s: str) -> tuple:
    parts = s.strip().split()
    return tuple(float(x) for x in parts)


def _mat4_identity() -> list[float]:
    """4x4 单位矩阵（column-major flat list）。"""
    return [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]


def _mat4_from_xyz_rpy(xyz: list, rpy: list) -> list[float]:
    """从平移+RPY欧拉角构造 4x4 矩阵（column-major）。"""
    r, p, y = rpy[0], rpy[1], rpy[2]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # Rotation = Rz(yaw) * Ry(pitch) * Rx(roll)
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    # Column-major: [col0, col1, col2, col3]
    return [
        r00, r10, r20, 0,
        r01, r11, r21, 0,
        r02, r12, r22, 0,
        xyz[0], xyz[1], xyz[2], 1,
    ]


def _mat4_mul(a: list[float], b: list[float]) -> list[float]:
    """4x4 矩阵乘法（column-major）。"""
    result = [0.0] * 16
    for col in range(4):
        for row in range(4):
            s = 0.0
            for k in range(4):
                s += a[k * 4 + row] * b[col * 4 + k]
            result[col * 4 + row] = s
    return result
