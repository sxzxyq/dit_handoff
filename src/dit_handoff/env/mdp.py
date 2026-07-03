"""MDP terms for the local dual-Franka handoff task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _target_center_w(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float],
    target_cube_center_z: float,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    target = torch.zeros((env.num_envs, 3), device=env.device, dtype=robot.data.root_pos_w.dtype)
    target_xy_offset = torch.tensor(target_xy, device=env.device, dtype=robot.data.root_pos_w.dtype)
    target[:, :2] = robot.data.root_pos_w[:, :2] + target_xy_offset
    target[:, 2] = robot.data.root_pos_w[:, 2] + target_cube_center_z
    return target


def _gripper_is_open(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    open_command: float,
    open_threshold: float,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids = robot_cfg.joint_ids
    if joint_ids is None:
        joint_ids, _ = robot.find_joints(["panda_finger.*"])
    finger_pos = torch.abs(robot.data.joint_pos[:, joint_ids])
    return torch.all(finger_pos >= open_command - open_threshold, dim=1)


def reset_object_pose_around_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    target_xy: tuple[float, float] = (0.50, 0.00),
    radius_range: tuple[float, float] = (0.12, 0.22),
    angle_range: tuple[float, float] = (-2.61799387799, -0.52359877559),
    object_center_z: float = 0.0205,
    yaw_range: tuple[float, float] = (0.0, 0.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset the cube around a target in the selected robot root frame."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    count = len(env_ids)
    device = env.device

    target_xy_t = torch.tensor(target_xy, device=device, dtype=robot.data.root_pos_w.dtype)
    radius = torch.empty(count, device=device).uniform_(*radius_range)
    angle = torch.empty(count, device=device).uniform_(*angle_range)

    positions = torch.zeros((count, 3), device=device, dtype=robot.data.root_pos_w.dtype)
    positions[:, :2] = robot.data.root_pos_w[env_ids, :2] + target_xy_t
    positions[:, 0] += radius * torch.cos(angle)
    positions[:, 1] += radius * torch.sin(angle)
    positions[:, 2] = robot.data.root_pos_w[env_ids, 2] + object_center_z

    yaw = torch.empty(count, device=device).uniform_(*yaw_range)
    zero = torch.zeros_like(yaw)
    orientations = quat_from_euler_xyz(zero, zero, yaw)
    velocities = torch.zeros((count, 6), device=device, dtype=robot.data.root_pos_w.dtype)

    object_asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    object_asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def target_area_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (0.50, 0.22),
    target_cube_center_z: float = 0.0205,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    target_pos_w = _target_center_w(env, target_xy, target_cube_center_z, robot_cfg)
    target_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w)
    return target_pos_b


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w)
    return ee_pos_b


def ee_quat_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    _, ee_quat_b = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w)
    return ee_quat_b


def object_target_xy_tracking(
    env: ManagerBasedRLEnv,
    std: float,
    lifted_height: float,
    target_xy: tuple[float, float] = (0.50, 0.22),
    target_cube_center_z: float = 0.0205,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object_asset: RigidObject = env.scene[object_cfg.name]
    target_pos_w = _target_center_w(env, target_xy, target_cube_center_z, robot_cfg)
    xy_distance = torch.linalg.vector_norm(object_asset.data.root_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    lifted = object_asset.data.root_pos_w[:, 2] > target_cube_center_z + lifted_height
    return lifted.float() * (1.0 - torch.tanh(xy_distance / std))


def object_on_target_area(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (0.50, 0.22),
    target_size_xy: tuple[float, float] = (0.12, 0.12),
    target_cube_center_z: float = 0.0205,
    height_tolerance: float = 0.03,
    require_release: bool = True,
    gripper_open_command: float = 0.04,
    gripper_open_threshold: float = 0.01,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object_asset: RigidObject = env.scene[object_cfg.name]
    target_pos_w = _target_center_w(env, target_xy, target_cube_center_z, robot_cfg)
    object_pos_w = object_asset.data.root_pos_w[:, :3]

    xy_error = torch.abs(object_pos_w[:, :2] - target_pos_w[:, :2])
    inside_target = torch.logical_and(xy_error[:, 0] <= target_size_xy[0] * 0.5, xy_error[:, 1] <= target_size_xy[1] * 0.5)
    low_enough = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2]) <= height_tolerance

    if require_release:
        released = _gripper_is_open(env, gripper_cfg, gripper_open_command, gripper_open_threshold)
    else:
        released = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    return inside_target & low_enough & released


def object_on_target_area_reward(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (0.50, 0.22),
    target_size_xy: tuple[float, float] = (0.12, 0.12),
    target_cube_center_z: float = 0.0205,
    height_tolerance: float = 0.03,
    require_release: bool = True,
    gripper_open_command: float = 0.04,
    gripper_open_threshold: float = 0.01,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return object_on_target_area(
        env,
        target_xy=target_xy,
        target_size_xy=target_size_xy,
        target_cube_center_z=target_cube_center_z,
        height_tolerance=height_tolerance,
        require_release=require_release,
        gripper_open_command=gripper_open_command,
        gripper_open_threshold=gripper_open_threshold,
        robot_cfg=robot_cfg,
        gripper_cfg=gripper_cfg,
        object_cfg=object_cfg,
    ).float()


class object_stably_placed_on_target(ManagerTermBase):
    """Terminate only after the red-target placement condition is true for consecutive steps."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._success_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._success_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        stable_steps: int = 10,
        target_xy: tuple[float, float] = (0.50, 0.22),
        target_size_xy: tuple[float, float] = (0.12, 0.12),
        target_cube_center_z: float = 0.0205,
        height_tolerance: float = 0.03,
        require_release: bool = True,
        gripper_open_command: float = 0.04,
        gripper_open_threshold: float = 0.01,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        placed = object_on_target_area(
            env,
            target_xy=target_xy,
            target_size_xy=target_size_xy,
            target_cube_center_z=target_cube_center_z,
            height_tolerance=height_tolerance,
            require_release=require_release,
            gripper_open_command=gripper_open_command,
            gripper_open_threshold=gripper_open_threshold,
            robot_cfg=robot_cfg,
            gripper_cfg=gripper_cfg,
            object_cfg=object_cfg,
        )
        self._success_steps = torch.where(placed, self._success_steps + 1, torch.zeros_like(self._success_steps))
        return self._success_steps >= stable_steps

