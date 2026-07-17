"""Dual-arm rectangular-bar aerial handoff IsaacLab config."""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from dit_handoff.constants import BAR_HANDOFF_LANGUAGE_INSTRUCTION, PANDA_JOINT_POS_ACTION_NAMES

from . import mdp as handoff_mdp
from .env_cfg import CubePickPlaceRedTargetFrankaIKRelVisuomotorEnvCfg
from .env_cfg import _cube_yaw_range_from_env
from .env_cfg import _parse_float_pair_env
from .handoff_env_cfg import _xy_world_to_robot_root_xy


def _bar_angle_range_from_env() -> tuple[float, float]:
    radians = os.environ.get("BAR_ANGLE_RANGE_RAD")
    if radians:
        return _parse_float_pair_env("BAR_ANGLE_RANGE_RAD", (-math.pi, math.pi))
    degrees = _parse_float_pair_env("BAR_ANGLE_RANGE_DEG", (-180.0, 180.0))
    return math.radians(degrees[0]), math.radians(degrees[1])


@configclass
class BarHandoffDualFrankaJointPosVisuomotorEnvCfg(CubePickPlaceRedTargetFrankaIKRelVisuomotorEnvCfg):
    """Rectangular bar handoff task with 18D absolute joint-position actions."""

    def __post_init__(self):
        super().__post_init__()

        self.task_name = "bar_handoff_dual_franka_joint_pos"
        self.instruction = BAR_HANDOFF_LANGUAGE_INSTRUCTION
        self.bar_size_xyz = (0.04, 0.04, 0.22)
        self.bar_grasp_inset_m = 0.025
        self.object_center_z = self.bar_size_xyz[2] * 0.5 + 0.0005
        self.target_object_center_z = self.bar_size_xyz[0] * 0.5 + 0.0005
        self.target_area_size_xy = (0.33, 0.33)
        self.red_area_center_world_xy = (0.50, 0.30)
        self.bar_init_center_world_xy = (0.50, -0.30)

        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(PANDA_JOINT_POS_ACTION_NAMES),
            scale=1.0,
            offset=0.0,
            preserve_order=True,
            use_default_offset=False,
        )
        self.actions.gripper_action = None
        self.actions.observer_arm_action = JointPositionActionCfg(
            asset_name="observer_robot",
            joint_names=list(PANDA_JOINT_POS_ACTION_NAMES),
            scale=1.0,
            offset=0.0,
            preserve_order=True,
            use_default_offset=False,
        )
        self.actions.observer_gripper_action = None

        actor_root_xy = (self.scene.robot.init_state.pos[0], self.scene.robot.init_state.pos[1])
        observer_root_xy = (self.scene.observer_robot.init_state.pos[0], self.scene.observer_robot.init_state.pos[1])
        red_target_xy_actor = _xy_world_to_robot_root_xy(self.red_area_center_world_xy, actor_root_xy)
        bar_center_xy_observer = _xy_world_to_robot_root_xy(self.bar_init_center_world_xy, observer_root_xy)

        self.target_area_center_xy = red_target_xy_actor
        self.cube_reset_target_xy = bar_center_xy_observer
        self.observations.policy.target_area_position.params["target_xy"] = red_target_xy_actor
        self.observations.policy.target_area_position.params["target_cube_center_z"] = self.target_object_center_z

        self.scene.object.init_state.pos = (
            self.bar_init_center_world_xy[0],
            self.bar_init_center_world_xy[1],
            self.object_center_z,
        )
        self.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.object.spawn.size = self.bar_size_xyz
        self.scene.object.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.055)
        self.scene.object.spawn.rigid_props = RigidBodyPropertiesCfg(
            solver_position_iteration_count=24,
            solver_velocity_iteration_count=2,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        self.scene.object.spawn.collision_props = CollisionPropertiesCfg()
        self.scene.object.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="min",
        )
        self.scene.object.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.15, 1.0),
            roughness=0.45,
            metallic=0.0,
        )
        self.scene.object.spawn.semantic_tags = [("class", "rectangular_bar")]

        self.scene.target_area.init_state.pos = (
            self.red_area_center_world_xy[0],
            self.red_area_center_world_xy[1],
            0.0015,
        )
        self.scene.target_area.spawn.size = (self.target_area_size_xy[0], self.target_area_size_xy[1], 0.001)
        self.scene.target_area.spawn.semantic_tags = [("class", "red_target_area")]

        self.events.reset_object_position = EventTerm(
            func=handoff_mdp.reset_object_pose_around_target,
            mode="reset",
            params={
                "target_xy": bar_center_xy_observer,
                "radius_range": _parse_float_pair_env("BAR_RADIUS_RANGE", (0.0, 0.08)),
                "angle_range": _bar_angle_range_from_env(),
                "object_center_z": self.object_center_z,
                "yaw_range": _cube_yaw_range_from_env(),
                "robot_cfg": SceneEntityCfg("observer_robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        for reward_name in ("object_goal_tracking", "object_goal_tracking_fine_grained"):
            getattr(self.rewards, reward_name).params["target_xy"] = red_target_xy_actor
            getattr(self.rewards, reward_name).params["target_cube_center_z"] = self.target_object_center_z

        self.rewards.placed_on_target.params["target_xy"] = red_target_xy_actor
        self.rewards.placed_on_target.params["target_size_xy"] = self.target_area_size_xy
        self.rewards.placed_on_target.params["target_cube_center_z"] = self.target_object_center_z
        self.terminations.success = None
