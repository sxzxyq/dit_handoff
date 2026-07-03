from dit_handoff.convert.handoff_state26_absjoint18 import action18_from_raw_step, state26_from_raw_step


def _row():
    left_joints = [float(i) for i in range(9)]
    right_joints = [float(i + 10) for i in range(9)]
    return {
        "pre_observation": {
            "arms": {
                "left": {"joint_pos": left_joints, "tcp_pos_w": [1, 2, 3], "gripper_opening": 0.08},
                "right": {"joint_pos": right_joints, "tcp_pos_w": [4, 5, 6], "gripper_opening": 0.07},
            }
        },
        "action": {
            "joint_target_18_commanded": [float(i) for i in range(18)],
            "joint_target_18_commanded_source": "commanded_env_action",
        },
    }


def test_state26_layout():
    state = state26_from_raw_step(_row())
    assert len(state) == 26
    assert state[:9] == [float(i) for i in range(9)]
    assert state[9:13] == [1.0, 2.0, 3.0, 0.08]
    assert state[13:22] == [float(i + 10) for i in range(9)]
    assert state[22:] == [4.0, 5.0, 6.0, 0.07]


def test_action18_uses_commanded_source():
    action = action18_from_raw_step(_row())
    assert len(action) == 18
    assert action == [float(i) for i in range(18)]

