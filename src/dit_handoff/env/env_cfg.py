"""IsaacLab config for the base red-target Franka cube task used by handoff."""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import ObservationsCfg as LiftObservationsCfg

from . import mdp as handoff_mdp


def _parse_float_pair_env(name: str, default: tuple[float, float]) -> tuple[float, float]:
    value = os.environ.get(name)
    if not value:
        return default
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be formatted as 'min,max', got {value!r}")
    return float(parts[0]), float(parts[1])


def _parse_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    return float(value)


def _cube_radius_range_from_env() -> tuple[float, float]:
    return _parse_float_pair_env("CUBE_RADIUS_RANGE", (0.12, 0.22))


def _cube_angle_range_from_env() -> tuple[float, float]:
    radians = os.environ.get("CUBE_ANGLE_RANGE_RAD")
    if radians:
        return _parse_float_pair_env("CUBE_ANGLE_RANGE_RAD", (-2.61799387799, -0.52359877559))
    degrees = _parse_float_pair_env("CUBE_ANGLE_RANGE_DEG", (-150.0, -30.0))
    return math.radians(degrees[0]), math.radians(degrees[1])


def _cube_yaw_range_from_env(default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    radians = os.environ.get("CUBE_YAW_RANGE_RAD")
    if radians:
        return _parse_float_pair_env("CUBE_YAW_RANGE_RAD", default)
    degrees = _parse_float_pair_env("CUBE_YAW_RANGE_DEG", (math.degrees(default[0]), math.degrees(default[1])))
    return math.radians(degrees[0]), math.radians(degrees[1])


def _target_xy_from_env() -> tuple[float, float]:
    return _parse_float_pair_env("TARGET_XY", (0.50, 0.00))


def _cube_reset_target_xy_from_env(default: tuple[float, float]) -> tuple[float, float]:
    return _parse_float_pair_env("CUBE_RESET_TARGET_XY", default)


@configclass
class PickPlaceObservationsCfg(LiftObservationsCfg):
    """Policy observations as named fields, including three RGB camera terms."""

    @configclass
    class PolicyCfg(LiftObservationsCfg.PolicyCfg):
        target_object_position = None
        target_area_position = ObsTerm(
            func=handoff_mdp.target_area_position_in_robot_root_frame,
            params={"target_xy": (0.50, 0.00), "target_cube_center_z": 0.0205},
        )
        ee_position = ObsTerm(func=handoff_mdp.ee_position_in_robot_root_frame)
        ee_quat = ObsTerm(func=handoff_mdp.ee_quat_in_robot_root_frame)
        wrist_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        observer_wrist_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("observer_wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        global_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("global_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class CubePickPlaceRedTargetFrankaIKRelVisuomotorEnvCfg(FrankaCubeLiftEnvCfg):
    """Lift-derived Franka env with a red target patch and three visuomotor cameras."""

    observations: PickPlaceObservationsCfg = PickPlaceObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.task_name = "cube_pick_place_red_target"
        self.instruction = "Pick up the cube and place it on the red target area."
        self.commands.object_pose = None
        self.observations.policy.target_object_position = None
        self.target_area_center_xy = _target_xy_from_env()
        self.cube_reset_target_xy = _cube_reset_target_xy_from_env(self.target_area_center_xy)
        self.target_area_size_xy = (0.12, 0.12)
        self.cube_size_m = _parse_float_env("CUBE_SIZE_M", 0.04)
        self.object_center_z = self.cube_size_m * 0.5 + 0.0005
        self.observations.policy.target_area_position.params["target_xy"] = self.target_area_center_xy
        self.observations.policy.target_area_position.params["target_cube_center_z"] = self.object_center_z
        self.events.reset_object_position = EventTerm(
            func=handoff_mdp.reset_object_pose_around_target,
            mode="reset",
            params={
                "target_xy": self.cube_reset_target_xy,
                "radius_range": _cube_radius_range_from_env(),
                "angle_range": _cube_angle_range_from_env(),
                "object_center_z": self.object_center_z,
                "yaw_range": _cube_yaw_range_from_env(),
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.events.reset_all.params = {"reset_joint_targets": True}
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 30.0
        self.terminations.time_out = None

        self.scene.table.spawn.semantic_tags = [("class", "table")]
        self.scene.table.spawn.scale = (1.35, 1.80, 1.0)

        actor_ready_joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.73,
            "panda_joint3": 0.0,
            "panda_joint4": -2.83,
            "panda_joint5": 0.0,
            "panda_joint6": 3.12,
            "panda_joint7": 0.8,
            "panda_finger_joint.*": 0.04,
        }
        self.scene.robot.init_state.pos = (0.0, 0.30, 0.0)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = actor_ready_joint_pos

        self.scene.observer_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/ObserverRobot")
        self.scene.observer_robot.init_state.pos = (0.0, -0.30, 0.0)
        self.scene.observer_robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.observer_robot.init_state.joint_pos = dict(actor_ready_joint_pos)
        self.scene.observer_robot.spawn.semantic_tags = [("class", "observer_robot")]

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.50, 0.00, self.object_center_z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(self.cube_size_m, self.cube_size_m, self.cube_size_m),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.15, 1.0),
                    roughness=0.45,
                    metallic=0.0,
                ),
                semantic_tags=[("class", "cube")],
            ),
        )

        self.scene.target_area = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TargetArea",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(self.target_area_center_xy[0], 0.30 + self.target_area_center_xy[1], 0.0015)
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.12, 0.12, 0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                    roughness=0.5,
                    metallic=0.0,
                ),
                collision_props=CollisionPropertiesCfg(collision_enabled=False),
                semantic_tags=[("class", "target_area")],
            ),
        )

        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.03, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.08, 0.0, 0.0375),
                rot=(0.66014, -0.25340, -0.25340, 0.66014),
                convention="ros",
            ),
        )
        self.scene.observer_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/ObserverRobot/panda_hand/observer_wrist_cam",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.03, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.08, 0.0, 0.0375),
                rot=(0.66014, -0.25340, -0.25340, 0.66014),
                convention="ros",
            ),
        )
        self.scene.global_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/GlobalCamera",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=14.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 3.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.20, 0.00, 1.00),
                rot=(0.69527, 0.12886, -0.12886, -0.69527),
                convention="opengl",
            ),
        )
        self.image_obs_list = ["wrist_cam", "observer_wrist_cam", "global_cam"]

        placement_params = {
            "target_xy": self.target_area_center_xy,
            "target_size_xy": self.target_area_size_xy,
            "target_cube_center_z": self.object_center_z,
            "height_tolerance": 0.03,
            "require_release": True,
            "gripper_open_command": 0.04,
            "gripper_open_threshold": 0.01,
            "robot_cfg": SceneEntityCfg("robot"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
            "object_cfg": SceneEntityCfg("object"),
        }
        self.rewards.reaching_object = RewTerm(func=lift_mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
        self.rewards.lifting_object = RewTerm(func=lift_mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=8.0)
        self.rewards.object_goal_tracking = RewTerm(
            func=handoff_mdp.object_target_xy_tracking,
            params={
                "std": 0.20,
                "lifted_height": self.cube_size_m,
                "target_xy": self.target_area_center_xy,
                "target_cube_center_z": self.object_center_z,
            },
            weight=10.0,
        )
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=handoff_mdp.object_target_xy_tracking,
            params={
                "std": 0.04,
                "lifted_height": self.cube_size_m,
                "target_xy": self.target_area_center_xy,
                "target_cube_center_z": self.object_center_z,
            },
            weight=6.0,
        )
        self.rewards.placed_on_target = RewTerm(
            func=handoff_mdp.object_on_target_area_reward,
            params=placement_params,
            weight=25.0,
        )
        self.rewards.action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)
        self.rewards.joint_vel = RewTerm(
            func=base_mdp.joint_vel_l2,
            weight=-1e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.object_dropping = DoneTerm(
            func=base_mdp.root_height_below_minimum,
            params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
        )
        self.terminations.success = DoneTerm(
            func=handoff_mdp.object_stably_placed_on_target,
            params={**placement_params, "stable_steps": 10},
        )

