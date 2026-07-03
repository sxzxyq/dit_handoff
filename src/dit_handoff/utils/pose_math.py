"""Small wxyz quaternion and pose helpers for anchored EE actions."""

from __future__ import annotations

import math
from typing import Any


def _vec(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ValueError(f"{label} must be {length}D")
    return [float(v) for v in value]


def normalize_quat_wxyz(quat: Any) -> list[float]:
    q = _vec(quat, 4, "quat")
    norm = math.sqrt(sum(v * v for v in q))
    if norm <= 0.0:
        raise ValueError("quaternion norm must be positive")
    q = [v / norm for v in q]
    if q[0] < 0.0:
        q = [-v for v in q]
    return q


def quat_conjugate_wxyz(quat: Any) -> list[float]:
    w, x, y, z = normalize_quat_wxyz(quat)
    return [w, -x, -y, -z]


def quat_multiply_wxyz(lhs: Any, rhs: Any, *, normalize: bool = True) -> list[float]:
    aw, ax, ay, az = normalize_quat_wxyz(lhs) if normalize else _vec(lhs, 4, "lhs")
    bw, bx, by, bz = normalize_quat_wxyz(rhs) if normalize else _vec(rhs, 4, "rhs")
    out = [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
    return normalize_quat_wxyz(out) if normalize else out


def quat_apply_wxyz(quat: Any, vec: Any) -> list[float]:
    q = normalize_quat_wxyz(quat)
    v = _vec(vec, 3, "vec")
    pure = [0.0, *v]
    rotated = quat_multiply_wxyz(
        quat_multiply_wxyz(q, pure, normalize=False), quat_conjugate_wxyz(q), normalize=False
    )
    return rotated[1:]


def quat_to_axis_angle_wxyz(quat: Any) -> list[float]:
    q = normalize_quat_wxyz(quat)
    w = max(-1.0, min(1.0, q[0]))
    xyz = q[1:]
    sin_half = math.sqrt(sum(v * v for v in xyz))
    if sin_half < 1e-8:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(sin_half, w)
    if angle > math.pi:
        angle -= 2.0 * math.pi
    scale = angle / sin_half
    return [v * scale for v in xyz]


def axis_angle_to_quat_wxyz(axis_angle: Any) -> list[float]:
    aa = _vec(axis_angle, 3, "axis_angle")
    angle = math.sqrt(sum(v * v for v in aa))
    if angle < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    half = 0.5 * angle
    scale = math.sin(half) / angle
    return normalize_quat_wxyz([math.cos(half), aa[0] * scale, aa[1] * scale, aa[2] * scale])


def pose_world_to_root(
    pos_w: Any,
    quat_w: Any,
    root_pos_w: Any,
    root_quat_w: Any,
) -> tuple[list[float], list[float]]:
    pos = _vec(pos_w, 3, "pos_w")
    root_pos = _vec(root_pos_w, 3, "root_pos_w")
    root_inv = quat_conjugate_wxyz(root_quat_w)
    rel_pos_w = [pos[i] - root_pos[i] for i in range(3)]
    pos_b = quat_apply_wxyz(root_inv, rel_pos_w)
    quat_b = quat_multiply_wxyz(root_inv, quat_w)
    return pos_b, quat_b


def pose_delta_axis_angle(
    source_pos: Any,
    source_quat: Any,
    target_pos: Any,
    target_quat: Any,
) -> list[float]:
    src_pos = _vec(source_pos, 3, "source_pos")
    tgt_pos = _vec(target_pos, 3, "target_pos")
    dpos = [tgt_pos[i] - src_pos[i] for i in range(3)]
    q_err = quat_multiply_wxyz(target_quat, quat_conjugate_wxyz(source_quat))
    return dpos + quat_to_axis_angle_wxyz(q_err)


def apply_pose_delta_axis_angle(
    source_pos: Any,
    source_quat: Any,
    delta6: Any,
) -> tuple[list[float], list[float]]:
    delta = _vec(delta6, 6, "delta6")
    src_pos = _vec(source_pos, 3, "source_pos")
    target_pos = [src_pos[i] + delta[i] for i in range(3)]
    delta_quat = axis_angle_to_quat_wxyz(delta[3:6])
    target_quat = quat_multiply_wxyz(delta_quat, source_quat)
    return target_pos, target_quat
