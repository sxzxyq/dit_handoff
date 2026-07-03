from pathlib import Path

from dit_handoff.constants import CAMERA_OBS_FEATURES
from dit_handoff.data.raw_validator import validate_raw_dataset
from dit_handoff.data.schema import episode_meta, raw_dataset_manifest
from dit_handoff.utils.io import write_json


def _write_dataset(root: Path, source: str = "commanded_env_action"):
    write_json(root / "dataset_manifest.json", raw_dataset_manifest("test_raw"))
    ep = root / "episodes" / "episode_000000"
    (ep / "images").mkdir(parents=True)
    for camera in CAMERA_OBS_FEATURES:
        d = ep / "images" / camera
        d.mkdir(parents=True)
        (d / "000000.png").write_bytes(b"")
    write_json(
        ep / "episode_meta.json",
        episode_meta(dataset_name="test_raw", episode_id=0, seed=1, max_steps=1, randomization={"enabled": False}),
    )
    arm = {
        "joint_pos": [0.0] * 9,
        "joint_vel": [0.0] * 9,
        "tcp_pos_w": [0.0, 0.0, 0.0],
        "tcp_quat_w": [1.0, 0.0, 0.0, 0.0],
        "gripper_opening": 0.08,
    }
    row = {
        "schema_version": "dit_raw_handoff_v1",
        "dataset_name": "test_raw",
        "episode_id": 0,
        "step_index": 0,
        "sim_time": 0.0,
        "pre_observation": {
            "arms": {"left": arm, "right": arm},
            "images": {camera: f"images/{camera}/000000.png" for camera in CAMERA_OBS_FEATURES},
        },
        "action": {
            "raw_env_action": [0.1] * 18,
            "action_interface": "Joint-Pos absolute target 18D",
            "joint_target_18_commanded": [0.1] * 18,
            "joint_target_18_commanded_source": source,
        },
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "env_info": {},
        "post_observation": {"arms": {"left": arm, "right": arm}},
    }
    (ep / "steps.jsonl").write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")


def test_raw_validator_accepts_commanded_action(tmp_path):
    _write_dataset(tmp_path)
    report = validate_raw_dataset(tmp_path)
    assert report["valid"]


def test_raw_validator_rejects_post_joint_source(tmp_path):
    _write_dataset(tmp_path, source="post_observed_joint_pos")
    report = validate_raw_dataset(tmp_path)
    assert not report["valid"]
    assert report["error_count"] >= 1

