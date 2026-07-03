"""State and action adapters for the state26 -> absjoint18 contract."""

from __future__ import annotations

from typing import Any

from dit_handoff.constants import ACTION_DIM, STATE_DIM
from dit_handoff.convert.handoff_state26_absjoint18 import state26_from_raw_step


def state26_from_live_snapshot(snapshot: dict[str, Any]) -> list[float]:
    row = {"pre_observation": snapshot}
    state = state26_from_raw_step(row)
    if len(state) != STATE_DIM:
        raise ValueError(f"state adapter produced {len(state)} values")
    return state


def action18_to_env_action(action: Any):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("closed-loop action adapter requires torch") from exc
    tensor = torch.as_tensor(action, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected 18D absolute joint target, got {tuple(tensor.shape)}")
    return tensor



def pose14_anchor_from_live_snapshot(snapshot: dict[str, Any]) -> dict[str, tuple[list[float], list[float]]]:
    from dit_handoff.convert.handoff_state26_relee_pose14 import arm_tcp_pose_root

    return {side: arm_tcp_pose_root(snapshot, side) for side in ("left", "right")}


def relee_pose14_to_ik_env_action(
    action: Any,
    anchor_pose: dict[str, tuple[list[float], list[float]]],
    current_snapshot: dict[str, Any],
    *,
    ik_action_scale: float = 0.5,
):
    from dit_handoff.constants import POSE14_ACTION_DIM
    from dit_handoff.convert.handoff_state26_relee_pose14 import arm_tcp_pose_root
    from dit_handoff.utils.pose_math import apply_pose_delta_axis_angle, pose_delta_axis_angle

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("closed-loop pose14 adapter requires torch") from exc

    tensor = torch.as_tensor(action, dtype=torch.float32).flatten()
    if tensor.numel() != POSE14_ACTION_DIM:
        raise ValueError(f"expected 14D relative EE pose action, got {tuple(tensor.shape)}")
    values = [float(v) for v in tensor.detach().cpu().tolist()]
    env_action: list[float] = []
    for side, start in (("left", 0), ("right", 7)):
        anchor_pos, anchor_quat = anchor_pose[side]
        target_pos, target_quat = apply_pose_delta_axis_angle(anchor_pos, anchor_quat, values[start : start + 6])
        current_pos, current_quat = arm_tcp_pose_root(current_snapshot, side)
        live_delta = pose_delta_axis_angle(current_pos, current_quat, target_pos, target_quat)
        if ik_action_scale <= 0.0:
            raise ValueError("ik_action_scale must be positive")
        env_action.extend([v / float(ik_action_scale) for v in live_delta])
        env_action.append(values[start + 6])
    return torch.tensor([env_action], dtype=torch.float32)
