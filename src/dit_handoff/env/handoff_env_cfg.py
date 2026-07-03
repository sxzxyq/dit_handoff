"""Dual-arm yellow-to-red handoff IsaacLab configs."""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.utils import configclass

from dit_handoff.constants import LANGUAGE_INSTRUCTION, PANDA_JOINT_POS_ACTION_NAMES

from . import mdp as handoff_mdp
from .env_cfg import CubePickPlaceRedTargetFrankaIKRelVisuomotorEnvCfg
from .env_cfg import _cube_yaw_range_from_env
from .env_cfg import _parse_float_pair_env


def _handoff_cube_angle_range_from_env() -> tuple[float, float]:
    radians = os.environ.get("CUBE_ANGLE_RANGE_RAD")
    if radians:
        return _parse_float_pair_env("CUBE_ANGLE_RANGE_RAD", (-math.pi, math.pi))
    degrees = _parse_float_pair_env("CUBE_ANGLE_RANGE_DEG", (-180.0, 180.0))
    return math.radians(degrees[0]), math.radians(degrees[1])


def _xy_world_to_robot_root_xy(world_xy: tuple[float, float], robot_root_xy: tuple[float, float]) -> tuple[float, float]:
    return world_xy[0] - robot_root_xy[0], world_xy[1] - robot_root_xy[1]


@configclass
class CubeHandoffYellowRedDualFrankaIKRelVisuomotorEnvCfg(CubePickPlaceRedTargetFrankaIKRelVisuomotorEnvCfg):
    """Scene layout for staged dual-arm handoff with IK relative pose actions."""

    def __post_init__(self):
        super().__post_init__()

        self.task_name = "cube_handoff_yellow_red_dual_franka"
        self.instruction = LANGUAGE_INSTRUCTION
        self.actions.observer_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="observer_robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        self.actions.observer_gripper_action = BinaryJointPositionActionCfg(
            asset_name="observer_robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.yellow_area_center_world_xy = (0.50, 0.00)
        self.red_area_center_world_xy = (0.50, 0.30)
        self.cube_init_center_world_xy = (0.50, -0.30)
        self.handoff_area_size_xy = (0.12, 0.12)

        actor_root_xy = (self.scene.robot.init_state.pos[0], self.scene.robot.init_state.pos[1])
        observer_root_xy = (self.scene.observer_robot.init_state.pos[0], self.scene.observer_robot.init_state.pos[1])
        red_target_xy_actor = _xy_world_to_robot_root_xy(self.red_area_center_world_xy, actor_root_xy)
        cube_center_xy_observer = _xy_world_to_robot_root_xy(self.cube_init_center_world_xy, observer_root_xy)

        self.target_area_center_xy = red_target_xy_actor
        self.cube_reset_target_xy = cube_center_xy_observer
        self.target_area_size_xy = (0.12, 0.12)
        self.observations.policy.target_area_position.params["target_xy"] = red_target_xy_actor

        self.scene.object.init_state.pos = (
            self.cube_init_center_world_xy[0],
            self.cube_init_center_world_xy[1],
            self.object_center_z,
        )
        self.scene.target_area.init_state.pos = (
            self.red_area_center_world_xy[0],
            self.red_area_center_world_xy[1],
            0.0015,
        )
        self.scene.target_area.spawn.semantic_tags = [("class", "red_target_area")]
        self.scene.yellow_area = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/YellowHandoffArea",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(self.yellow_area_center_world_xy[0], self.yellow_area_center_world_xy[1], 0.0016)
            ),
            spawn=sim_utils.CuboidCfg(
                size=(self.handoff_area_size_xy[0], self.handoff_area_size_xy[1], 0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.85, 0.0),
                    roughness=0.5,
                    metallic=0.0,
                ),
                collision_props=CollisionPropertiesCfg(collision_enabled=False),
                semantic_tags=[("class", "yellow_handoff_area")],
            ),
        )

        self.events.reset_object_position = EventTerm(
            func=handoff_mdp.reset_object_pose_around_target,
            mode="reset",
            params={
                "target_xy": cube_center_xy_observer,
                "radius_range": _parse_float_pair_env("CUBE_RADIUS_RANGE", (0.0, 0.10)),
                "angle_range": _handoff_cube_angle_range_from_env(),
                "object_center_z": self.object_center_z,
                "yaw_range": _cube_yaw_range_from_env(),
                "robot_cfg": SceneEntityCfg("observer_robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        for reward_name in ("object_goal_tracking", "object_goal_tracking_fine_grained"):
            getattr(self.rewards, reward_name).params["target_xy"] = red_target_xy_actor

        self.rewards.placed_on_target.params["target_xy"] = red_target_xy_actor
        self.rewards.placed_on_target.params["target_size_xy"] = self.target_area_size_xy
        self.terminations.success = None


@configclass
class CubeHandoffYellowRedDualFrankaJointPosVisuomotorEnvCfg(CubeHandoffYellowRedDualFrankaIKRelVisuomotorEnvCfg):
    """Handoff variant with absolute joint-position actions for both Franka arms.

    Action order is left/actor 9D followed by right/observer 9D. Each 9D arm block is
    panda_joint1..7, panda_finger_joint1, panda_finger_joint2.
    """

    def __post_init__(self):
        super().__post_init__()

        self.task_name = "cube_handoff_yellow_red_dual_franka_joint_pos"
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

