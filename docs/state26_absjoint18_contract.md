# State26 -> AbsJoint18 Contract

This is the only converter contract implemented in this workspace.

Images:

1. `observation.images.wrist_rgb`
2. `observation.images.observer_wrist_rgb`
3. `observation.images.global_rgb`

`observation.state` is 26D:

1. left actor joint positions, 9D
2. left actor TCP position in world frame, 3D
3. left gripper opening scalar
4. right observer joint positions, 9D
5. right observer TCP position in world frame, 3D
6. right gripper opening scalar

`action` is 18D absolute Joint-Pos target:

1. left actor target, 9D
2. right observer target, 9D

Each 9D arm block is:

```text
panda_joint1
panda_joint2
panda_joint3
panda_joint4
panda_joint5
panda_joint6
panda_joint7
panda_finger_joint1
panda_finger_joint2
```

The action label must come from the actual 18D target sent to the Joint-Pos action space. It must not be backfilled from post-step observed joint positions.

Quaternion convention for raw pose fields is `wxyz`. The current state26 contract does not consume quaternions; future rotation features should prefer explicit axis-angle representation and must mark derived values in the converter manifest.

`gripper_opening` is defined as `sum(abs(panda_finger_joint1), abs(panda_finger_joint2))` unless an env-provided scalar with a different definition is explicitly documented.

