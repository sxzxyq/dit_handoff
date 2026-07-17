"""Expert/action providers for the generic raw collector."""

from __future__ import annotations

from typing import Any

from dit_handoff.constants import LEFT_ARM_ASSET, PANDA_JOINT_POS_ACTION_NAMES, RIGHT_ARM_ASSET

TCP_OFFSET = (0.0, 0.0, 0.107)
LEFT_ARM = LEFT_ARM_ASSET
RIGHT_ARM = RIGHT_ARM_ASSET
OPEN_GRIPPER = 0.04
CLOSE_GRIPPER = 0.0
AREA_SIZE_XY = (0.12, 0.12)
OBJECT_CENTER_Z = 0.0205
HEIGHT_TOLERANCE = 0.03
GRIPPER_OPEN_THRESHOLD = 0.01

PHASES = (
    "right_open_rest",
    "right_move_above_cube",
    "right_descend_to_grasp",
    "right_close_gripper",
    "right_lift_cube",
    "right_move_above_yellow",
    "right_descend_to_yellow",
    "right_release_on_yellow",
    "right_retreat",
    "wait_yellow_stable",
    "left_open_rest",
    "left_move_above_cube",
    "left_descend_to_grasp",
    "left_close_gripper",
    "left_lift_cube",
    "left_move_above_red",
    "left_descend_to_red",
    "left_release_on_red",
    "left_retreat",
    "wait_red_stable",
    "done",
)

SCRIPTED_DEFAULTS = {
    "yellow_xy": (0.50, 0.00),
    "red_xy": (0.50, 0.30),
    "hover_z": 0.20,
    "lift_z": 0.19,
    "grasp_z": 0.015,
    "release_z": 0.085,
    "max_delta": 0.018,
    "max_joint_delta": 0.08,
    "damping": 0.05,
    "pos_threshold": 0.018,
    "rest_steps": 20,
    "close_steps": 35,
    "open_steps": 35,
    "stable_steps": 12,
    "phase_timeout": 320,
}

_BODY_ID_CACHE: dict[tuple[int, str], int] = {}
_JOINT_ID_CACHE: dict[int, list[int]] = {}


def _asset(env: Any, name: str) -> Any:
    return env.unwrapped.scene[name]


def hold_current_jointpos_action(env: Any, obs: Any, step_index: int, episode_id: int, context: dict[str, Any]):
    """Command the current left/right joint positions as a valid 18D Joint-Pos target.

    This is intended for smoke tests and contract validation. A real data run should pass an
    expert callable that returns the commanded Joint-Pos targets used for task execution.
    """
    import torch

    left = _asset(env, LEFT_ARM).data.joint_pos[:, :9]
    right = _asset(env, RIGHT_ARM).data.joint_pos[:, :9]
    return torch.cat([left, right], dim=-1).detach().clone()


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


def _tcp_pos_w(env: Any, arm_name: str):
    import torch
    from isaaclab.utils import math as math_utils

    robot = _asset(env, arm_name)
    body_idx = _body_id(env, arm_name)
    hand_pos = robot.data.body_pos_w[:, body_idx, :]
    hand_quat = robot.data.body_quat_w[:, body_idx, :]
    offset = torch.tensor(TCP_OFFSET, device=env.unwrapped.device, dtype=hand_pos.dtype).repeat(env.unwrapped.num_envs, 1)
    return hand_pos + math_utils.quat_apply(hand_quat, offset)


def _cube_pos_w(env: Any):
    return env.unwrapped.scene["object"].data.root_pos_w[:, :3]


def _gripper_is_open(env: Any, arm_name: str):
    import torch

    robot = _asset(env, arm_name)
    finger_pos = torch.abs(robot.data.joint_pos[:, 7:9])
    return torch.all(finger_pos >= OPEN_GRIPPER - GRIPPER_OPEN_THRESHOLD, dim=1)


def _area_center(env: Any, area: str, cfg: dict[str, Any]):
    import torch

    if area == "yellow":
        xy = cfg["yellow_xy"]
    elif area == "red":
        xy = cfg["red_xy"]
    else:
        raise ValueError(f"Unknown area: {area}")
    center = torch.zeros((env.unwrapped.num_envs, 3), device=env.unwrapped.device)
    center[:, 0] = float(xy[0])
    center[:, 1] = float(xy[1])
    center[:, 2] = OBJECT_CENTER_Z
    return center


def _object_on_area(env: Any, center_w: Any, gripper_arm: str):
    import torch

    cube_pos = _cube_pos_w(env)
    xy_error = torch.abs(cube_pos[:, :2] - center_w[:, :2])
    inside = torch.logical_and(xy_error[:, 0] <= AREA_SIZE_XY[0] * 0.5, xy_error[:, 1] <= AREA_SIZE_XY[1] * 0.5)
    low = torch.abs(cube_pos[:, 2] - center_w[:, 2]) <= HEIGHT_TOLERANCE
    released = _gripper_is_open(env, gripper_arm)
    return inside & low & released


def _active_arm_for_phase(phase_name: str) -> str | None:
    if phase_name.startswith("right"):
        return RIGHT_ARM
    if phase_name.startswith("left"):
        return LEFT_ARM
    return None


def _target_area_for_phase(phase_name: str) -> str | None:
    if "yellow" in phase_name:
        return "yellow"
    if "red" in phase_name:
        return "red"
    return None


def _gripper_target_for_phase(phase_name: str) -> float:
    if "close" in phase_name or "lift" in phase_name or "move_above_yellow" in phase_name:
        return CLOSE_GRIPPER
    if "descend_to_yellow" in phase_name or "move_above_red" in phase_name or "descend_to_red" in phase_name:
        return CLOSE_GRIPPER
    return OPEN_GRIPPER


def _desired_pos_for_phase(env: Any, phase_name: str, cfg: dict[str, Any], park_positions: dict[str, Any]):
    active_arm = _active_arm_for_phase(phase_name)
    if active_arm is None:
        return None

    desired = _tcp_pos_w(env, active_arm).clone()
    cube_pos = _cube_pos_w(env)
    area_name = _target_area_for_phase(phase_name)
    target = _area_center(env, area_name, cfg) if area_name else None

    if "move_above_cube" in phase_name:
        desired[:, :2] = cube_pos[:, :2]
        desired[:, 2] = cfg["hover_z"]
    elif "descend_to_grasp" in phase_name or "close_gripper" in phase_name:
        desired[:, :2] = cube_pos[:, :2]
        desired[:, 2] = cfg["grasp_z"]
    elif "lift_cube" in phase_name:
        desired[:, :2] = cube_pos[:, :2]
        desired[:, 2] = cfg["lift_z"]
    elif "move_above_yellow" in phase_name or "move_above_red" in phase_name:
        desired[:, :2] = target[:, :2]
        desired[:, 2] = cfg["lift_z"]
    elif "descend_to_yellow" in phase_name or "descend_to_red" in phase_name:
        desired[:, :2] = target[:, :2]
        desired[:, 2] = cfg["release_z"]
    elif "release_on_yellow" in phase_name or "release_on_red" in phase_name:
        desired[:, :2] = target[:, :2]
        desired[:, 2] = cfg["release_z"]
    elif "retreat" in phase_name:
        desired = park_positions[active_arm].clone()
    return desired


def _position_jacobian_w(env: Any, arm_name: str):
    import torch
    from isaaclab.utils import math as math_utils

    robot = _asset(env, arm_name)
    body_idx = _body_id(env, arm_name)
    joint_ids = _arm_joint_ids(env, arm_name)
    jacobi_body_idx = body_idx - 1 if robot.is_fixed_base else body_idx
    jacobi_joint_ids = joint_ids if robot.is_fixed_base else [idx + 6 for idx in joint_ids]
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, jacobi_joint_ids].clone()
    hand_quat = robot.data.body_quat_w[:, body_idx, :]
    offset = torch.tensor(TCP_OFFSET, device=env.unwrapped.device, dtype=jacobian.dtype).repeat(env.unwrapped.num_envs, 1)
    offset_w = math_utils.quat_apply(hand_quat, offset)
    jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_w), jacobian[:, 3:6, :])
    return jacobian[:, 0:3, :]


def _dls_joint_delta(jac_pos: Any, delta_w: Any, damping: float):
    import torch

    jj_t = torch.bmm(jac_pos, jac_pos.transpose(1, 2))
    eye = torch.eye(3, device=jac_pos.device, dtype=jac_pos.dtype).unsqueeze(0).repeat(jac_pos.shape[0], 1, 1)
    rhs = delta_w.unsqueeze(-1)
    solved = torch.linalg.solve(jj_t + float(damping) ** 2 * eye, rhs)
    return torch.bmm(jac_pos.transpose(1, 2), solved).squeeze(-1)


def _joint_limits(device: Any, dtype: Any):
    import torch

    lower = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0], device=device, dtype=dtype)
    upper = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04], device=device, dtype=dtype)
    return lower, upper


def _arm_jointpos_target(env: Any, arm_name: str, desired_pos_w: Any, gripper_target: float, cfg: dict[str, Any]):
    import torch

    robot = _asset(env, arm_name)
    current = robot.data.joint_pos[:, :9].detach().clone()
    tcp_pos = _tcp_pos_w(env, arm_name)
    delta_w = desired_pos_w - tcp_pos
    distance = torch.linalg.vector_norm(delta_w, dim=1)
    scale = torch.clamp(float(cfg["max_delta"]) / (distance + 1.0e-8), max=1.0).unsqueeze(-1)
    clipped_delta_w = delta_w * scale
    jac_pos = _position_jacobian_w(env, arm_name)
    dq = _dls_joint_delta(jac_pos, clipped_delta_w, float(cfg["damping"]))
    dq = torch.clamp(dq, -float(cfg["max_joint_delta"]), float(cfg["max_joint_delta"]))
    target = current.clone()
    target[:, :7] = current[:, :7] + dq
    target[:, 7:9] = float(gripper_target)
    lower, upper = _joint_limits(target.device, target.dtype)
    target = torch.maximum(torch.minimum(target, upper), lower)
    return target, distance


def _park_positions(env: Any, cfg: dict[str, Any]):
    import torch

    parks = {LEFT_ARM: _tcp_pos_w(env, LEFT_ARM).clone(), RIGHT_ARM: _tcp_pos_w(env, RIGHT_ARM).clone()}
    for value in parks.values():
        value[:, 2] = torch.maximum(value[:, 2], torch.full_like(value[:, 2], float(cfg["hover_z"])))
    return parks


def _advance_state(env: Any, state: dict[str, Any], cfg: dict[str, Any]) -> None:
    phase_name = PHASES[state["phase_idx"]]
    yellow = _object_on_area(env, _area_center(env, "yellow", cfg), RIGHT_ARM)
    red = _object_on_area(env, _area_center(env, "red", cfg), LEFT_ARM)
    yellow_now = bool(yellow[0].detach().cpu().item())
    red_now = bool(red[0].detach().cpu().item())
    state["yellow_stable_steps"] = state["yellow_stable_steps"] + 1 if yellow_now else 0
    state["red_stable_steps"] = state["red_stable_steps"] + 1 if red_now else 0
    if state["yellow_stable_steps"] >= int(cfg["stable_steps"]):
        state["yellow_achieved"] = True
    if state["red_stable_steps"] >= int(cfg["stable_steps"]):
        state["red_achieved"] = True
    reached = float(state.get("last_distance", 1.0e9)) <= float(cfg["pos_threshold"])

    advance = False
    success = False
    if phase_name.endswith("open_rest"):
        advance = state["phase_steps"] >= int(cfg["rest_steps"])
    elif "close_gripper" in phase_name:
        advance = state["phase_steps"] >= int(cfg["close_steps"])
    elif "release_on" in phase_name:
        advance = state["phase_steps"] >= int(cfg["open_steps"])
    elif "retreat" in phase_name:
        advance = reached
    elif phase_name == "wait_yellow_stable":
        advance = bool(state.get("yellow_achieved", False)) or state["phase_steps"] >= int(cfg["phase_timeout"])
    elif phase_name == "wait_red_stable":
        success = bool(state.get("red_achieved", False))
        advance = success or state["phase_steps"] >= int(cfg["phase_timeout"])
    elif phase_name == "done":
        state["done"] = True
        return
    else:
        advance = reached or state["phase_steps"] >= int(cfg["phase_timeout"])

    if advance:
        state["phase_idx"] = min(state["phase_idx"] + 1, len(PHASES) - 1)
        state["phase_steps"] = 0
        if PHASES[state["phase_idx"]] == "done":
            state["done"] = True
            state["expert_success"] = bool(success and state.get("yellow_achieved", False))
    else:
        state["phase_steps"] += 1


def _scripted_state(env: Any, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = dict(SCRIPTED_DEFAULTS)
    cfg.update(context.get("scripted_jointpos_cfg", {}))
    state = context.get("_scripted_jointpos_state")
    if state is None:
        state = {
            "phase_idx": 0,
            "phase_steps": 0,
            "last_step_index": None,
            "last_distance": 1.0e9,
            "yellow_stable_steps": 0,
            "red_stable_steps": 0,
            "yellow_achieved": False,
            "red_achieved": False,
            "park_positions": _park_positions(env, cfg),
            "done": False,
            "expert_success": False,
        }
        context["_scripted_jointpos_state"] = state
    return state, cfg


def _scripted_pose14_state(env: Any, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = dict(SCRIPTED_DEFAULTS)
    cfg.update(context.get("scripted_pose14_cfg", {}))
    state = context.get("_scripted_pose14_state")
    if state is None:
        state = {
            "phase_idx": 0,
            "phase_steps": 0,
            "last_step_index": None,
            "last_distance": 1.0e9,
            "yellow_stable_steps": 0,
            "red_stable_steps": 0,
            "yellow_achieved": False,
            "red_achieved": False,
            "park_positions": _park_positions(env, cfg),
            "done": False,
            "expert_success": False,
        }
        context["_scripted_pose14_state"] = state
    return state, cfg


def _pose14_gripper_sign_for_phase(phase_name: str) -> float:
    return 1.0 if _gripper_target_for_phase(phase_name) >= OPEN_GRIPPER * 0.5 else -1.0


def _arm_pose14_delta_action(env: Any, arm_name: str, desired_pos_w: Any, gripper_sign: float, cfg: dict[str, Any]):
    import torch

    tcp_pos = _tcp_pos_w(env, arm_name)
    delta_w = desired_pos_w - tcp_pos
    distance = torch.linalg.vector_norm(delta_w, dim=1)
    scale = torch.clamp(float(cfg["max_delta"]) / (distance + 1.0e-8), max=1.0).unsqueeze(-1)
    physical_delta = delta_w * scale
    ik_action_scale = float(cfg.get("ik_action_scale", 0.5))
    if ik_action_scale <= 0.0:
        raise ValueError("ik_action_scale must be positive")
    action = torch.zeros((env.unwrapped.num_envs, 7), device=env.unwrapped.device, dtype=tcp_pos.dtype)
    action[:, :3] = physical_delta / ik_action_scale
    action[:, 6] = float(gripper_sign)
    return action, distance


def scripted_handoff_relee_pose14_action(env: Any, obs: Any, step_index: int, episode_id: int, context: dict[str, Any]):
    """Scripted yellow-to-red handoff expert for 14D IK-relative EE pose actions.

    The returned action is the raw env command. The arm delta components are divided by
    the IK action scale so that the controller receives the intended physical TCP delta.
    """
    import torch

    state, cfg = _scripted_pose14_state(env, context)
    if state["last_step_index"] is not None and step_index > int(state["last_step_index"]):
        _advance_state(env, state, cfg)

    action = torch.zeros((env.unwrapped.num_envs, 14), device=env.unwrapped.device)
    action[:, 6] = 1.0
    action[:, 13] = 1.0
    if state.get("done"):
        context["expert_done"] = True
        context["expert_success"] = bool(state.get("expert_success", False))
        return action

    phase_name = PHASES[state["phase_idx"]]
    active_arm = _active_arm_for_phase(phase_name)
    desired_pos = _desired_pos_for_phase(env, phase_name, cfg, state["park_positions"])
    distance = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    if active_arm is not None and desired_pos is not None:
        arm_action, distance = _arm_pose14_delta_action(
            env, active_arm, desired_pos, _pose14_gripper_sign_for_phase(phase_name), cfg
        )
        if active_arm == LEFT_ARM:
            action[:, 0:7] = arm_action
            action[:, 13] = 1.0
        else:
            action[:, 7:14] = arm_action
            action[:, 6] = 1.0

    state["last_distance"] = float(distance[0].detach().cpu().item()) if distance.numel() else 0.0
    state["last_step_index"] = int(step_index)
    context["expert_done"] = False
    context["expert_success"] = False
    return action.detach().clone()


def scripted_handoff_jointpos_action(env: Any, obs: Any, step_index: int, episode_id: int, context: dict[str, Any]):
    """Scripted yellow-to-red handoff expert that commands 18D absolute Joint-Pos targets.

    The phase and waypoint logic mirrors the read-only legacy IK scripted expert, but this
    callable converts Cartesian TCP waypoints to commanded joint targets with a DLS positional
    Jacobian solve. Phase state stays in the collector context and is not written into raw steps.
    """
    import torch

    state, cfg = _scripted_state(env, context)
    if state["last_step_index"] is not None and step_index > int(state["last_step_index"]):
        _advance_state(env, state, cfg)

    action = hold_current_jointpos_action(env, obs, step_index, episode_id, context)
    if state.get("done"):
        context["expert_done"] = True
        context["expert_success"] = bool(state.get("expert_success", False))
        return action

    phase_name = PHASES[state["phase_idx"]]
    active_arm = _active_arm_for_phase(phase_name)
    desired_pos = _desired_pos_for_phase(env, phase_name, cfg, state["park_positions"])
    distance = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    if active_arm is not None and desired_pos is not None:
        target, distance = _arm_jointpos_target(env, active_arm, desired_pos, _gripper_target_for_phase(phase_name), cfg)
        if active_arm == LEFT_ARM:
            action[:, 0:9] = target
        else:
            action[:, 9:18] = target

    state["last_distance"] = float(distance[0].detach().cpu().item()) if distance.numel() else 0.0
    state["last_step_index"] = int(step_index)
    context["expert_done"] = False
    context["expert_success"] = False
    return action.detach().clone()
