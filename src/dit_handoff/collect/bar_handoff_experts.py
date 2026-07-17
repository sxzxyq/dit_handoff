"""Scripted experts for the rectangular-bar aerial handoff task."""

from __future__ import annotations

from typing import Any

from dit_handoff.constants import LEFT_ARM_ASSET, PANDA_JOINT_POS_ACTION_NAMES, RIGHT_ARM_ASSET

TCP_OFFSET = (0.0, 0.0, 0.107)
LEFT_ARM = LEFT_ARM_ASSET
RIGHT_ARM = RIGHT_ARM_ASSET
OPEN_GRIPPER = 0.04
CLOSE_GRIPPER = 0.0
GRIPPER_OPEN_THRESHOLD = 0.01

BAR_PHASES = (
    "right_open_rest",
    "right_move_above_right_end",
    "right_descend_to_right_end",
    "right_close_gripper",
    "right_lift_bar",
    "right_rotate_bar_horizontal",
    "right_move_to_air_handoff",
    "right_hold_handoff",
    "left_open_rest",
    "left_orient_for_handoff",
    "left_move_to_left_end",
    "left_descend_to_left_end",
    "left_close_gripper",
    "dual_hold",
    "right_release",
    "right_clear_handoff",
    "right_retreat",
    "left_rotate_bar_vertical",
    "left_lift_bar",
    "left_move_above_red",
    "left_descend_to_red",
    "left_release_on_red",
    "left_retreat",
    "wait_red_stable",
    "post_success_hold",
    "done",
)

BAR_SCRIPTED_DEFAULTS = {
    "bar_size_xyz": (0.04, 0.04, 0.22),
    "grasp_inset": 0.040,
    "right_pick_offset_z": 0.065,
    "left_pick_offset_z": 0.035,
    "left_grasp_tcp_z_bias": -0.015,
    "red_xy": (0.50, 0.30),
    "red_descend_xy": (0.50, 0.30),
    "red_size_xy": (0.33, 0.33),
    "handoff_center_xy": (0.50, 0.00),
    "hover_z": 0.20,
    "approach_lift_z": 0.12,
    "left_approach_lift_z": 0.045,
    "handoff_z": 0.30,
    "lift_z": 0.30,
    "release_z": 0.17,
    "max_delta": 0.022,
    "max_rot_delta": 0.03,
    "rotate_max_rot_delta": 0.06,
    "max_joint_delta": 0.12,
    "damping": 0.05,
    "orientation_weight": 0.45,
    "pos_threshold": 0.014,
    "left_approach_threshold": 0.045,
    "left_descend_threshold": 0.025,
    "left_descend_steps": 110,
    "left_orient_safe_xy": (0.50, 0.24),
    "left_orient_safe_z": 0.42,
    "left_orient_hold_steps": 35,
    "left_orient_rot_threshold": 0.35,
    "left_orient_phase_timeout": 520,
    "rot_threshold": 0.20,
    "rest_steps": 20,
    "close_steps": 130,
    "open_steps": 35,
    "lift_hold_steps": 45,
    "rotate_hold_steps": 45,
    "left_rotate_hold_steps": 80,
    "handoff_hold_steps": 20,
    "dual_hold_steps": 5,
    "right_clear_xy": (0.50, -0.24),
    "right_clear_z": 0.38,
    "right_clear_steps": 45,
    "handoff_min_bar_z": 0.18,
    "left_grip_min_opening": 0.025,
    "left_grip_max_opening": 0.055,
    "left_close_min_steps": 20,
    "stable_steps": 12,
    "post_success_hold_steps": 100,
    "phase_timeout": 520,
}

_BODY_ID_CACHE: dict[tuple[int, str], int] = {}
_JOINT_ID_CACHE: dict[int, list[int]] = {}


def _asset(env: Any, name: str) -> Any:
    return env.unwrapped.scene[name]


def _body_id(env: Any, arm_name: str, body_name: str = "panda_hand") -> int:
    asset = _asset(env, arm_name)
    key = (id(asset), body_name)
    if key not in _BODY_ID_CACHE:
        body_ids, body_names = asset.find_bodies(body_name)
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one body for {arm_name}:{body_name}, got {body_names}")
        _BODY_ID_CACHE[key] = int(body_ids[0])
    return _BODY_ID_CACHE[key]


def _arm_joint_ids(env: Any, arm_name: str) -> list[int]:
    asset = _asset(env, arm_name)
    key = id(asset)
    if key not in _JOINT_ID_CACHE:
        joint_ids, joint_names = asset.find_joints(list(PANDA_JOINT_POS_ACTION_NAMES[:7]), preserve_order=True)
        if len(joint_ids) != 7:
            raise RuntimeError(f"Expected 7 arm joints for {arm_name}, got {joint_names}")
        _JOINT_ID_CACHE[key] = [int(idx) for idx in joint_ids]
    return _JOINT_ID_CACHE[key]


def _tcp_pose_w(env: Any, arm_name: str):
    import torch
    from isaaclab.utils import math as math_utils

    robot = _asset(env, arm_name)
    body_idx = _body_id(env, arm_name)
    hand_pos = robot.data.body_pos_w[:, body_idx, :]
    hand_quat = robot.data.body_quat_w[:, body_idx, :]
    offset = torch.tensor(TCP_OFFSET, device=env.unwrapped.device, dtype=hand_pos.dtype).repeat(env.unwrapped.num_envs, 1)
    tcp_pos = hand_pos + math_utils.quat_apply(hand_quat, offset)
    return tcp_pos, hand_quat


def _bar_pose_w(env: Any):
    obj = env.unwrapped.scene["object"]
    return obj.data.root_pos_w[:, :3], obj.data.root_quat_w[:, :4]


def _quat_conj(q):
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _quat_mul(q1, q2):
    import torch

    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _world_axis_angle_quat(env: Any, axis: tuple[float, float, float], angle_rad: float):
    import math
    import torch

    quat = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    half = 0.5 * float(angle_rad)
    quat[:, 0] = math.cos(half)
    quat[:, 1] = float(axis[0]) * math.sin(half)
    quat[:, 2] = float(axis[1]) * math.sin(half)
    quat[:, 3] = float(axis[2]) * math.sin(half)
    return quat


def _horizontal_handoff_quat(env: Any, grasp_quat):
    import math

    # The right gripper holds the upper end of the vertical bar. Rotate about world X so
    # that the held upper end becomes the right/negative-Y end and the lower end swings
    # toward the left arm for handoff.
    return _quat_mul(_world_axis_angle_quat(env, (1.0, 0.0, 0.0), 0.67 * math.pi), grasp_quat)


def _left_cross_grip_quat(env: Any, grasp_quat):
    import math

    # Mirror the right arm's horizontal handoff wrist rotation. The left gripper should be
    # the symmetric partner of the right gripper, not a copied hand frame that points the
    # wrist camera outward away from the bar and the right arm.
    return _quat_mul(_world_axis_angle_quat(env, (1.0, 0.0, 0.0), -0.67 * math.pi), grasp_quat)


def _quat_error_axis_angle(current_q, desired_q):
    import torch

    current_q = current_q / torch.linalg.vector_norm(current_q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    desired_q = desired_q / torch.linalg.vector_norm(desired_q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    q_err = _quat_mul(desired_q, _quat_conj(current_q))
    q_err = torch.where(q_err[:, :1] < 0.0, -q_err, q_err)
    vec = q_err[:, 1:]
    vec_norm = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vec_norm, q_err[:, :1].clamp_min(1.0e-8))
    axis = vec / vec_norm.clamp_min(1.0e-8)
    return axis * angle


def _clamp_vec_norm(vec, max_norm: float):
    import torch

    norm = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_norm) / norm.clamp_min(1.0e-8), max=1.0)
    return vec * scale, norm.squeeze(-1)


def _bar_axis_z_w(env: Any):
    import torch
    from isaaclab.utils import math as math_utils

    bar_pos, bar_quat = _bar_pose_w(env)
    axis = torch.zeros_like(bar_pos)
    axis[:, 2] = 1.0
    return math_utils.quat_apply(bar_quat, axis)


def _bar_endpoints_w(env: Any, cfg: dict[str, Any]):
    bar_pos, _ = _bar_pose_w(env)
    axis_z = _bar_axis_z_w(env)
    half = float(cfg["bar_size_xyz"][2]) * 0.5 - float(cfg["grasp_inset"])
    right_pick_offset = float(cfg.get("right_pick_offset_z", -half * 0.5))
    lower_end = bar_pos - axis_z * half
    upper_end = bar_pos + axis_z * half
    right_pick = bar_pos + axis_z * right_pick_offset
    return {"right": lower_end, "right_pick": right_pick, "left": upper_end, "free": lower_end, "held": upper_end, "axis_z": axis_z}


def _target_from_xy_z(env: Any, xy: tuple[float, float], z: float):
    import torch

    out = torch.zeros((env.unwrapped.num_envs, 3), device=env.unwrapped.device)
    out[:, 0] = float(xy[0])
    out[:, 1] = float(xy[1])
    out[:, 2] = float(z)
    return out


def _nominal_vertical_grasp_at_object_center(env: Any, center_xy: tuple[float, float], z: float, side: str, cfg: dict[str, Any]):
    import torch

    axis = torch.zeros((env.unwrapped.num_envs, 3), device=env.unwrapped.device)
    axis[:, 2] = 1.0
    center = _target_from_xy_z(env, center_xy, z)
    offset = float(cfg["left_pick_offset_z"] if side == "left" else cfg["right_pick_offset_z"])
    return center + axis * offset


def _nominal_horizontal_grasp_at_object_center(env: Any, center_xy: tuple[float, float], z: float, side: str, cfg: dict[str, Any]):
    import torch

    axis = torch.zeros((env.unwrapped.num_envs, 3), device=env.unwrapped.device)
    axis[:, 1] = 1.0
    center = _target_from_xy_z(env, center_xy, z)
    half = float(cfg["bar_size_xyz"][2]) * 0.5 - float(cfg["grasp_inset"])
    sign = 1.0 if side == "left" else -1.0
    return center + axis * (sign * half)


def _gripper_is_open(env: Any, arm_name: str):
    import torch

    robot = _asset(env, arm_name)
    finger_pos = torch.abs(robot.data.joint_pos[:, 7:9])
    return torch.all(finger_pos >= OPEN_GRIPPER - GRIPPER_OPEN_THRESHOLD, dim=1)


def _object_on_red(env: Any, cfg: dict[str, Any]):
    import torch

    bar_pos, _ = _bar_pose_w(env)
    red = _target_from_xy_z(env, cfg["red_xy"], float(cfg["bar_size_xyz"][0]) * 0.5 + 0.0005)
    xy_error = torch.abs(bar_pos[:, :2] - red[:, :2])
    inside = torch.logical_and(
        xy_error[:, 0] <= float(cfg["red_size_xy"][0]) * 0.5,
        xy_error[:, 1] <= float(cfg["red_size_xy"][1]) * 0.5,
    )
    low = torch.abs(bar_pos[:, 2] - red[:, 2]) <= 0.04
    return inside & low & _gripper_is_open(env, LEFT_ARM)


def _jacobian_6d_w(env: Any, arm_name: str):
    import torch
    from isaaclab.utils import math as math_utils

    robot = _asset(env, arm_name)
    body_idx = _body_id(env, arm_name)
    joint_ids = _arm_joint_ids(env, arm_name)
    jacobi_body_idx = body_idx - 1 if robot.is_fixed_base else body_idx
    jacobi_joint_ids = joint_ids if robot.is_fixed_base else [idx + 6 for idx in joint_ids]
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, jacobi_joint_ids].clone()
    hand_quat = robot.data.body_quat_w[:, body_idx, :]
    hand_pos = robot.data.body_pos_w[:, body_idx, :]
    offset = torch.tensor(TCP_OFFSET, device=env.unwrapped.device, dtype=jacobian.dtype).repeat(env.unwrapped.num_envs, 1)
    offset_w = math_utils.quat_apply(hand_quat, offset)
    jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_w), jacobian[:, 3:6, :])
    return jacobian


def _dls_joint_delta(jacobian, error, damping: float):
    import torch

    jj_t = torch.bmm(jacobian, jacobian.transpose(1, 2))
    eye = torch.eye(jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0).repeat(jacobian.shape[0], 1, 1)
    solved = torch.linalg.solve(jj_t + float(damping) ** 2 * eye, error.unsqueeze(-1))
    return torch.bmm(jacobian.transpose(1, 2), solved).squeeze(-1)


def _joint_limits(device: Any, dtype: Any):
    import torch

    lower = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0], device=device, dtype=dtype)
    upper = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04], device=device, dtype=dtype)
    return lower, upper


def _arm_jointpose_target(env: Any, arm_name: str, desired_pos_w, desired_quat_w, gripper_target: float, cfg: dict[str, Any]):
    import torch

    robot = _asset(env, arm_name)
    current = robot.data.joint_pos[:, :9].detach().clone()
    tcp_pos, tcp_quat = _tcp_pose_w(env, arm_name)
    pos_error, pos_distance = _clamp_vec_norm(desired_pos_w - tcp_pos, float(cfg["max_delta"]))
    rot_error, rot_distance = _clamp_vec_norm(_quat_error_axis_angle(tcp_quat, desired_quat_w), float(cfg["max_rot_delta"]))
    rot_error = rot_error * float(cfg["orientation_weight"])
    error = torch.cat([pos_error, rot_error], dim=-1)
    jacobian = _jacobian_6d_w(env, arm_name).clone()
    jacobian[:, 3:6, :] *= float(cfg["orientation_weight"])
    dq = _dls_joint_delta(jacobian, error, float(cfg["damping"]))
    dq = torch.clamp(dq, -float(cfg["max_joint_delta"]), float(cfg["max_joint_delta"]))
    target = current.clone()
    target[:, :7] = current[:, :7] + dq
    target[:, 7:9] = float(gripper_target)
    lower, upper = _joint_limits(target.device, target.dtype)
    target = torch.maximum(torch.minimum(target, upper), lower)
    return target, pos_distance, rot_distance


def _hold_current_jointpos_action(env: Any):
    import torch

    left = _asset(env, LEFT_ARM).data.joint_pos[:, :9]
    right = _asset(env, RIGHT_ARM).data.joint_pos[:, :9]
    return torch.cat([left, right], dim=-1).detach().clone()


def _park_positions(env: Any, cfg: dict[str, Any]):
    import torch

    left_pos, left_quat = _tcp_pose_w(env, LEFT_ARM)
    right_pos, right_quat = _tcp_pose_w(env, RIGHT_ARM)
    parks = {
        LEFT_ARM: {"pos": left_pos.clone(), "quat": left_quat.clone()},
        RIGHT_ARM: {"pos": right_pos.clone(), "quat": right_quat.clone()},
    }
    for item in parks.values():
        item["pos"][:, 2] = torch.maximum(item["pos"][:, 2], torch.full_like(item["pos"][:, 2], float(cfg["hover_z"])))
    return parks


def _scripted_bar_state(env: Any, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = dict(BAR_SCRIPTED_DEFAULTS)
    cfg.update(context.get("scripted_bar_jointpos_cfg", {}))
    state = context.get("_scripted_bar_state")
    if state is None:
        left_pos, left_quat = _tcp_pose_w(env, LEFT_ARM)
        right_pos, right_quat = _tcp_pose_w(env, RIGHT_ARM)
        endpoints = _bar_endpoints_w(env, cfg)
        state = {
            "phase_idx": 0,
            "phase_steps": 0,
            "last_step_index": None,
            "last_distance": 1.0e9,
            "last_rot_distance": 1.0e9,
            "last_arm_distances": {},
            "last_arm_rot_distances": {},
            "red_stable_steps": 0,
            "red_achieved": False,
            "park_positions": _park_positions(env, cfg),
            "grasp_quats": {LEFT_ARM: left_quat.clone(), RIGHT_ARM: right_quat.clone()},
            "right_pick_pos_w": endpoints["right_pick"].clone(),
            "left_handoff_hold_pose": None,
            "right_handoff_hold_pose": None,
            "done": False,
            "expert_success": False,
            "quality_failure": None,
        }
        context["_scripted_bar_state"] = state
    return state, cfg


def _left_handoff_hold_pose(env: Any, state: dict[str, Any]):
    pose = state.get("left_handoff_hold_pose")
    if pose is None:
        pos, quat = _tcp_pose_w(env, LEFT_ARM)
        pose = {"pos": pos.detach().clone(), "quat": quat.detach().clone()}
        state["left_handoff_hold_pose"] = pose
    return pose["pos"].clone(), pose["quat"].clone()


def _right_handoff_hold_pose(env: Any, state: dict[str, Any]):
    pose = state.get("right_handoff_hold_pose")
    if pose is None:
        pos, quat = _tcp_pose_w(env, RIGHT_ARM)
        pose = {"pos": pos.detach().clone(), "quat": quat.detach().clone()}
        state["right_handoff_hold_pose"] = pose
    return pose["pos"].clone(), pose["quat"].clone()


def _gripper_opening(env: Any, arm_name: str):
    import torch

    robot = _asset(env, arm_name)
    return torch.abs(robot.data.joint_pos[:, 7:9]).sum(dim=1)


def _left_grip_secure(env: Any, cfg: dict[str, Any]):
    import torch

    opening = _gripper_opening(env, LEFT_ARM)
    return torch.logical_and(
        opening >= float(cfg["left_grip_min_opening"]),
        opening <= float(cfg["left_grip_max_opening"]),
    )


def _handoff_bar_high(env: Any, cfg: dict[str, Any]):
    bar_pos, _ = _bar_pose_w(env)
    return bar_pos[:, 2] >= float(cfg["handoff_min_bar_z"])


def _mark_quality_failure(state: dict[str, Any], reason: str) -> None:
    state["quality_failure"] = reason
    state["expert_success"] = False
    state["done"] = True


def _desired_targets_for_phase(env: Any, phase_name: str, state: dict[str, Any], cfg: dict[str, Any]):
    endpoints = _bar_endpoints_w(env, cfg)
    targets: dict[str, tuple[Any, Any, float]] = {}
    left_grasp_quat = state["grasp_quats"][LEFT_ARM]
    left_cross_quat = _left_cross_grip_quat(env, left_grasp_quat)
    right_grasp_quat = state["grasp_quats"][RIGHT_ARM]
    right_horizontal_quat = _horizontal_handoff_quat(env, right_grasp_quat)

    right_support_phases = {"left_open_rest", "left_orient_for_handoff", "left_move_to_left_end", "left_descend_to_left_end", "left_close_gripper", "dual_hold"}
    if phase_name.startswith("right") or phase_name in right_support_phases:
        right_pos = endpoints["right"].clone()
        right_quat = right_grasp_quat
        if phase_name in {"right_move_above_right_end", "right_descend_to_right_end", "right_close_gripper", "right_lift_bar"}:
            right_pos = state["right_pick_pos_w"].clone()
        if "move_above_right_end" in phase_name:
            right_pos[:, 2] = state["right_pick_pos_w"][:, 2] + float(cfg["approach_lift_z"])
        elif "descend_to_right_end" in phase_name or "close_gripper" in phase_name:
            right_pos = state["right_pick_pos_w"].clone()
        elif "lift_bar" in phase_name:
            right_pos = state["right_pick_pos_w"].clone()
            right_pos[:, 2] = float(cfg["lift_z"])
        elif "rotate_bar_horizontal" in phase_name:
            right_pos = state["right_pick_pos_w"].clone()
            right_pos[:, 2] = float(cfg["lift_z"])
            right_quat = right_horizontal_quat
        elif "move_to_air_handoff" in phase_name:
            right_pos = _nominal_horizontal_grasp_at_object_center(env, cfg["handoff_center_xy"], float(cfg["handoff_z"]), "right", cfg)
            right_quat = right_horizontal_quat
        elif "hold_handoff" in phase_name or phase_name in {
            "left_open_rest",
            "left_orient_for_handoff",
            "left_move_to_left_end",
            "left_descend_to_left_end",
            "left_close_gripper",
            "dual_hold",
            "right_release",
        }:
            # Freeze the actual right TCP after it reaches handoff. Otherwise the right arm keeps
            # tracking a nominal pose while the left gripper closes, and the contact makes both arms wiggle.
            right_pos, right_quat = _right_handoff_hold_pose(env, state)
        elif phase_name == "right_clear_handoff":
            right_pos = _target_from_xy_z(env, cfg["right_clear_xy"], float(cfg["right_clear_z"]))
            right_quat = right_horizontal_quat
        elif "retreat" in phase_name:
            right_pos = state["park_positions"][RIGHT_ARM]["pos"].clone()
        right_closed_phases = {
            "right_close_gripper",
            "right_lift_bar",
            "right_rotate_bar_horizontal",
            "right_move_to_air_handoff",
            "right_hold_handoff",
            "left_open_rest",
            "left_orient_for_handoff",
            "left_move_to_left_end",
            "left_descend_to_left_end",
            "left_close_gripper",
            "dual_hold",
        }
        grip = CLOSE_GRIPPER if phase_name in right_closed_phases else OPEN_GRIPPER
        targets[RIGHT_ARM] = (right_pos, right_quat, grip)

    if phase_name.startswith("left") or phase_name in {"dual_hold", "right_release", "right_clear_handoff", "right_retreat"}:
        left_pos = endpoints["left"].clone()
        left_quat = left_grasp_quat
        if phase_name in {"left_open_rest"}:
            left_pos = state["park_positions"][LEFT_ARM]["pos"].clone()
        elif phase_name == "left_orient_for_handoff":
            # Rotate in a safe parked pose first. Moving toward the bar while changing wrist
            # orientation sweeps the gripper through the object and knocks it out of the right hand.
            left_pos = _target_from_xy_z(env, cfg["left_orient_safe_xy"], float(cfg["left_orient_safe_z"]))
            left_quat = left_cross_quat
        elif "move_to_left_end" in phase_name:
            # Approach the live free end from above first; going straight to the endpoint can push
            # the right-held bar away before the left gripper closes.
            left_pos = endpoints["free"].clone()
            left_pos[:, 2] = left_pos[:, 2] + float(cfg["left_approach_lift_z"])
            left_quat = left_cross_quat
        elif "descend_to_left_end" in phase_name:
            left_pos = endpoints["free"].clone()
            left_pos[:, 2] = left_pos[:, 2] + float(cfg["left_grasp_tcp_z_bias"])
            left_quat = left_cross_quat
        elif phase_name in {"left_close_gripper", "dual_hold", "right_release", "right_clear_handoff", "right_retreat"}:
            # As soon as the left gripper starts closing, actively own the bar with a fixed
            # airborne TCP pose. Do not chase the live bar end while contact settles.
            left_pos, left_quat = _left_handoff_hold_pose(env, state)
        elif phase_name == "left_rotate_bar_vertical":
            left_pos, _ = _left_handoff_hold_pose(env, state)
            left_pos[:, 2] = float(cfg["lift_z"])
            left_quat = left_grasp_quat
        elif "lift_bar" in phase_name:
            left_pos, _ = _left_handoff_hold_pose(env, state)
            left_pos[:, 2] = float(cfg["lift_z"])
            left_quat = left_grasp_quat
        elif "move_above_red" in phase_name:
            left_pos = _target_from_xy_z(env, cfg["red_xy"], float(cfg["lift_z"]))
            left_quat = left_grasp_quat
        elif "descend_to_red" in phase_name or "release_on_red" in phase_name:
            left_pos = _target_from_xy_z(env, cfg["red_descend_xy"], float(cfg["release_z"]))
            left_quat = left_grasp_quat
        elif "retreat" in phase_name:
            left_pos = state["park_positions"][LEFT_ARM]["pos"].clone()
        left_closed_phases = {
            "left_close_gripper",
            "dual_hold",
            "right_release",
            "right_clear_handoff",
            "right_retreat",
            "left_rotate_bar_vertical",
            "left_lift_bar",
            "left_move_above_red",
            "left_descend_to_red",
        }
        grip = CLOSE_GRIPPER if phase_name in left_closed_phases else OPEN_GRIPPER
        targets[LEFT_ARM] = (left_pos, left_quat, grip)

    return targets


def _advance_bar_state(env: Any, state: dict[str, Any], cfg: dict[str, Any]) -> None:
    phase_name = BAR_PHASES[state["phase_idx"]]
    red_now = bool(_object_on_red(env, cfg)[0].detach().cpu().item())
    state["red_stable_steps"] = state["red_stable_steps"] + 1 if red_now else 0
    if state["red_stable_steps"] >= int(cfg["stable_steps"]):
        state["red_achieved"] = True
    last_distance = float(state.get("last_distance", 1.0e9))
    arm_distances = state.get("last_arm_distances", {})
    arm_rot_distances = state.get("last_arm_rot_distances", {})
    left_distance = float(arm_distances.get(LEFT_ARM, last_distance))
    left_rot_distance = float(arm_rot_distances.get(LEFT_ARM, state.get("last_rot_distance", 1.0e9)))
    reached = last_distance <= float(cfg["pos_threshold"])
    left_approach_reached = left_distance <= float(cfg["left_approach_threshold"])
    left_descend_reached = left_distance <= float(cfg["left_descend_threshold"])
    rot_reached = float(state.get("last_rot_distance", 1.0e9)) <= float(cfg["rot_threshold"])
    left_orient_reached = left_rot_distance <= float(cfg["left_orient_rot_threshold"])
    left_grip_secure = bool(_left_grip_secure(env, cfg)[0].detach().cpu().item())
    handoff_bar_high = bool(_handoff_bar_high(env, cfg)[0].detach().cpu().item())

    if phase_name in {"left_descend_to_left_end", "left_close_gripper", "dual_hold"} and not handoff_bar_high and state["phase_steps"] > 10:
        _mark_quality_failure(state, "bar_low_during_handoff")
        return

    advance = False
    success = False
    if phase_name.endswith("open_rest"):
        advance = state["phase_steps"] >= int(cfg["rest_steps"])
    elif phase_name == "left_close_gripper":
        if state["phase_steps"] >= int(cfg["phase_timeout"]) and not left_grip_secure:
            _mark_quality_failure(state, "left_gripper_not_secure")
            return
        advance = left_grip_secure and state["phase_steps"] >= int(cfg["left_close_min_steps"])
    elif "close_gripper" in phase_name:
        advance = state["phase_steps"] >= int(cfg["close_steps"])
    elif phase_name in {"right_lift_bar"}:
        # Lift clear of the table before rotating; otherwise the lower end can scrape the tabletop.
        advance = (reached and state["phase_steps"] >= int(cfg["lift_hold_steps"])) or state["phase_steps"] >= int(cfg["phase_timeout"])
    elif phase_name in {"right_rotate_bar_horizontal"}:
        # Do not start translating to the handoff center until the bar has actually rotated horizontal.
        advance = (reached and rot_reached and state["phase_steps"] >= int(cfg["rotate_hold_steps"])) or state["phase_steps"] >= int(cfg["phase_timeout"])
    elif phase_name in {"right_hold_handoff"}:
        advance = state["phase_steps"] >= int(cfg["handoff_hold_steps"])
    elif phase_name == "left_orient_for_handoff":
        advance = (
            left_orient_reached
            and state["phase_steps"] >= int(cfg["left_orient_hold_steps"])
        ) or state["phase_steps"] >= int(cfg["left_orient_phase_timeout"])
    elif phase_name == "left_move_to_left_end":
        advance = left_approach_reached or state["phase_steps"] >= int(cfg["phase_timeout"])
    elif phase_name == "left_descend_to_left_end":
        if state["phase_steps"] >= int(cfg["phase_timeout"]) and not left_descend_reached:
            _mark_quality_failure(state, "left_descend_not_reached")
            return
        advance = left_descend_reached
    elif phase_name in {"dual_hold"}:
        if state["phase_steps"] >= int(cfg["phase_timeout"]) and not left_grip_secure:
            _mark_quality_failure(state, "dual_hold_left_grip_not_secure")
            return
        advance = left_grip_secure and handoff_bar_high and state["phase_steps"] >= int(cfg["dual_hold_steps"])
    elif phase_name == "right_clear_handoff":
        advance = reached or state["phase_steps"] >= int(cfg["right_clear_steps"])
    elif phase_name == "left_rotate_bar_vertical":
        advance = (reached and rot_reached and state["phase_steps"] >= int(cfg["left_rotate_hold_steps"])) or state["phase_steps"] >= int(cfg["phase_timeout"])
    elif "release" in phase_name:
        advance = state["phase_steps"] >= int(cfg["open_steps"])
    elif phase_name == "wait_red_stable":
        success = bool(state.get("red_achieved", False))
        advance = success or state["phase_steps"] >= int(cfg["phase_timeout"])
    elif phase_name == "post_success_hold":
        success = bool(state.get("red_achieved", False))
        advance = state["phase_steps"] >= int(cfg["post_success_hold_steps"])
    elif phase_name == "done":
        state["done"] = True
        return
    else:
        advance = reached or state["phase_steps"] >= int(cfg["phase_timeout"])

    if advance:
        state["phase_idx"] = min(state["phase_idx"] + 1, len(BAR_PHASES) - 1)
        state["phase_steps"] = 0
        if BAR_PHASES[state["phase_idx"]] == "done":
            state["done"] = True
            state["expert_success"] = bool(success or state.get("red_achieved", False))
    else:
        state["phase_steps"] += 1


def scripted_bar_handoff_jointpos_action(env: Any, obs: Any, step_index: int, episode_id: int, context: dict[str, Any]):
    """Return the actual 18D Joint-Pos action sent to the bar-handoff env."""
    import torch

    state, cfg = _scripted_bar_state(env, context)
    if state["last_step_index"] is not None and step_index > int(state["last_step_index"]):
        _advance_bar_state(env, state, cfg)

    action = _hold_current_jointpos_action(env)
    if state.get("done"):
        context["expert_done"] = True
        context["expert_success"] = bool(state.get("expert_success", False))
        context["quality_failure"] = state.get("quality_failure")
        return action

    phase_name = BAR_PHASES[state["phase_idx"]]
    distances = []
    rot_distances = []
    arm_distances = {}
    arm_rot_distances = {}
    for arm_name, (desired_pos, desired_quat, gripper) in _desired_targets_for_phase(env, phase_name, state, cfg).items():
        arm_cfg = cfg
        if arm_name == RIGHT_ARM and phase_name == "right_rotate_bar_horizontal":
            arm_cfg = dict(cfg)
            arm_cfg["max_rot_delta"] = float(cfg["rotate_max_rot_delta"])
        target, distance, rot_distance = _arm_jointpose_target(env, arm_name, desired_pos, desired_quat, gripper, arm_cfg)
        if arm_name == LEFT_ARM:
            action[:, 0:9] = target
        else:
            action[:, 9:18] = target
        distances.append(distance)
        rot_distances.append(rot_distance)
        arm_distances[arm_name] = float(distance.detach().cpu().max().item())
        arm_rot_distances[arm_name] = float(rot_distance.detach().cpu().max().item())

    if distances:
        state["last_distance"] = float(torch.stack(distances, dim=0).max().detach().cpu().item())
    else:
        state["last_distance"] = 0.0
    if rot_distances:
        state["last_rot_distance"] = float(torch.stack(rot_distances, dim=0).max().detach().cpu().item())
    else:
        state["last_rot_distance"] = 0.0
    state["last_arm_distances"] = arm_distances
    state["last_arm_rot_distances"] = arm_rot_distances
    state["last_step_index"] = int(step_index)
    context["expert_done"] = False
    context["expert_success"] = False
    context["quality_failure"] = state.get("quality_failure")
    return action.detach().clone()
