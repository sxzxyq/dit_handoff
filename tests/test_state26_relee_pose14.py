import json
import math
from pathlib import Path

import numpy as np

from dit_handoff.convert.handoff_state26_relee_pose14 import (
    _write_action_chunk_sidecar,
    action14_from_raw_step,
    gripper_opening_to_sign,
)
from dit_handoff.utils.pose_math import axis_angle_to_quat_wxyz, apply_pose_delta_axis_angle, pose_delta_axis_angle


def _arm(pos, quat=None, opening=0.08):
    return {
        "joint_pos": [0.0] * 9,
        "tcp_pos_w": list(pos),
        "tcp_quat_w": list(quat or [1.0, 0.0, 0.0, 0.0]),
        "gripper_opening": opening,
    }


def _row(pre_x, post_x, *, left_open=0.08, right_open=0.0):
    return {
        "pre_observation": {
            "arms": {
                "left": _arm([pre_x, 0.30, 0.50], opening=left_open),
                "right": _arm([pre_x, -0.30, 0.50], opening=right_open),
            }
        },
        "post_observation": {
            "arms": {
                "left": _arm([post_x, 0.30, 0.50], opening=left_open),
                "right": _arm([post_x, -0.30, 0.50], opening=right_open),
            }
        },
        "action": {
            "joint_target_18_commanded": [0.0] * 7 + [left_open / 2, left_open / 2] + [0.0] * 7 + [right_open / 2, right_open / 2],
        },
    }


def test_pose_math_delta_round_trip():
    source_pos = [0.1, 0.2, 0.3]
    source_quat = [1.0, 0.0, 0.0, 0.0]
    target_pos = [0.2, 0.1, 0.5]
    target_quat = axis_angle_to_quat_wxyz([0.0, 0.0, math.pi / 2])
    delta = pose_delta_axis_angle(source_pos, source_quat, target_pos, target_quat)
    recovered_pos, recovered_quat = apply_pose_delta_axis_angle(source_pos, source_quat, delta)
    assert np.allclose(recovered_pos, target_pos)
    assert np.allclose(recovered_quat, target_quat)


def test_action14_uses_root_frame_and_gripper_sign():
    action = action14_from_raw_step(_row(0.40, 0.45, left_open=0.08, right_open=0.0))
    assert len(action) == 14
    assert np.allclose(action[:3], [0.05, 0.0, 0.0])
    assert np.allclose(action[7:10], [0.05, 0.0, 0.0])
    assert action[6] == 1.0
    assert action[13] == -1.0
    assert gripper_opening_to_sign(0.0) == -1.0
    assert gripper_opening_to_sign(0.08) == 1.0


def test_action_chunk_sidecar_anchors_future_to_same_current_pose(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    ep_dir = raw_dir / "episodes" / "episode_000000"
    ep_dir.mkdir(parents=True)
    rows = [_row(0.40, 0.45), _row(0.45, 0.50), _row(0.50, 0.55)]
    with (ep_dir / "steps.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    out_dir = tmp_path / "lerobot"
    meta = _write_action_chunk_sidecar(raw_dir, out_dir, [ep_dir], horizon=4, n_obs_steps=2)
    actions = np.load(out_dir / meta["actions_path"])
    mask = np.load(out_dir / meta["action_is_pad_path"])
    assert actions.shape == (3, 4, 14)
    assert mask[0].tolist() == [True, False, False, False]
    assert np.allclose(actions[0, 1, :3], [0.05, 0.0, 0.0])
    assert np.allclose(actions[0, 2, :3], [0.10, 0.0, 0.0])
    assert np.allclose(actions[0, 3, :3], [0.15, 0.0, 0.0])
    assert mask[2].tolist() == [True, False, True, True]
