import json
from pathlib import Path

import numpy as np

from dit_handoff.collect.raw_collector_relee_pose14 import _action_row
from dit_handoff.convert.handoff_state26_relee_pose14 import _write_action_chunk_sidecar, action14_from_raw_step


def _arm(pos, quat=None, opening=0.08):
    return {
        "joint_pos": [0.0] * 9,
        "joint_vel": [0.0] * 9,
        "tcp_pos_w": list(pos),
        "tcp_quat_w": list(quat or [1.0, 0.0, 0.0, 0.0]),
        "gripper_opening": opening,
    }


def _row(pre_x, commanded_dx, *, left_grip=1.0, right_grip=-1.0):
    commanded = [commanded_dx, 0.0, 0.0, 0.0, 0.0, 0.0, left_grip, commanded_dx, 0.0, 0.0, 0.0, 0.0, 0.0, right_grip]
    return {
        "pre_observation": {
            "arms": {
                "left": _arm([pre_x, 0.30, 0.50]),
                "right": _arm([pre_x, -0.30, 0.50]),
            },
            "images": {"wrist_rgb": "images/wrist_rgb/000000.png", "observer_wrist_rgb": "images/observer_wrist_rgb/000000.png", "global_rgb": "images/global_rgb/000000.png"},
        },
        "post_observation": {
            "arms": {
                "left": _arm([pre_x + commanded_dx, 0.30, 0.50]),
                "right": _arm([pre_x + commanded_dx, -0.30, 0.50]),
            }
        },
        "action": {
            "raw_env_action": commanded,
            "pose14_delta_commanded": commanded,
            "pose14_delta_commanded_source": "commanded_expert_action",
        },
    }


def test_pose14_collector_action_row_records_commanded_physical_delta():
    row = _action_row([[0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], 0.5)
    assert row["action_interface"] == "Dual Franka IK relative EE pose 14D"
    assert row["pose14_delta_commanded_source"] == "commanded_expert_action"
    assert len(row["raw_env_action"]) == 14
    assert len(row["pose14_delta_commanded"]) == 14
    assert np.allclose(row["pose14_delta_commanded"][:3], [0.01, 0.0, 0.0])
    assert np.allclose(row["pose14_delta_commanded"][7:10], [-0.02, 0.0, 0.0])
    assert row["pose14_delta_commanded"][6] == 1.0
    assert row["pose14_delta_commanded"][13] == -1.0


def test_action14_from_raw_step_prefers_commanded_pose14():
    row = _row(0.40, 0.03, left_grip=-1.0, right_grip=1.0)
    action = action14_from_raw_step(row)
    assert np.allclose(action[:3], [0.03, 0.0, 0.0])
    assert action[6] == -1.0
    assert np.allclose(action[7:10], [0.03, 0.0, 0.0])
    assert action[13] == 1.0


def test_commanded_pose14_sidecar_reanchors_future_commanded_targets(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    ep_dir = raw_dir / "episodes" / "episode_000000"
    ep_dir.mkdir(parents=True)
    rows = [_row(0.40, 0.01), _row(0.45, 0.02), _row(0.50, 0.03)]
    with (ep_dir / "steps.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    out_dir = tmp_path / "lerobot"
    meta = _write_action_chunk_sidecar(raw_dir, out_dir, [ep_dir], horizon=4, n_obs_steps=2)
    actions = np.load(out_dir / meta["actions_path"])
    mask = np.load(out_dir / meta["action_is_pad_path"])
    assert actions.shape == (3, 4, 14)
    assert mask[0].tolist() == [True, False, False, False]
    assert np.allclose(actions[0, 1, :3], [0.01, 0.0, 0.0])
    assert np.allclose(actions[0, 2, :3], [0.07, 0.0, 0.0])
    assert np.allclose(actions[0, 3, :3], [0.13, 0.0, 0.0])
    assert actions[0, 1, 6] == 1.0
    assert actions[0, 1, 13] == -1.0
