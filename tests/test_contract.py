from dit_handoff import constants as c


def test_state_and_action_layout_lengths():
    assert len(c.STATE26_LAYOUT) == 26
    assert len(c.ACTION18_LAYOUT) == 18
    assert len(c.JOINT_NAMES_18) == 18
    assert c.CAMERA_OBS_FEATURES == ("wrist_rgb", "observer_wrist_rgb", "global_rgb")


def test_action_order_left_then_right():
    assert c.ACTION18_LAYOUT[0].endswith("left/panda_joint1")
    assert c.ACTION18_LAYOUT[8].endswith("left/panda_finger_joint2")
    assert c.ACTION18_LAYOUT[9].endswith("right/panda_joint1")
    assert c.ACTION18_LAYOUT[17].endswith("right/panda_finger_joint2")

