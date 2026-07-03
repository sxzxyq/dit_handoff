# State26 -> Relative EE Pose14 Contract

Contract: `state26_relee_pose14_v1`

Input stays identical to the previous contract: `observation.state` is state26 and the visual inputs are `wrist_rgb`, `observer_wrist_rgb`, and `global_rgb`. TCP quaternion is not added to the model observation.

The model output is a 14D anchored relative end-effector pose action:

- left arm: `dx, dy, dz, dax, day, daz, gripper_sign`
- right arm: `dx, dy, dz, dax, day, daz, gripper_sign`

The pose delta is expressed in each arm's own robot root frame. Rotation is axis-angle, computed as target quaternion times source quaternion inverse. `gripper_sign` is `+1` for open and `-1` for close.

For a policy call at raw frame `t`, every executed action in the chunk is anchored to the same current TCP pose at `t`. It is not chained and it is not relative to future rows. The sidecar stores `actions.npy` and `action_is_pad.npy`; the slot rule is:

```text
target_raw_index = anchor_raw_index + slot - (n_obs_steps - 1)
```

With the default `n_obs_steps=2`, slot 0 is masked and slot 1 is `a0`. MultiTask DiT executes from slot `n_obs_steps - 1`, so this matches inference.

LeRobot's official RelativeActionsProcessorStep is not used here because it subtracts `observation.state` from `action` dimension-wise. Our `state26` does not align with EE pose14. Instead, the converter writes relative labels directly and the task-specific training wrapper injects the anchored action sidecar into the official LeRobot training loop.

## Commands

Convert:

```bash
/home/qsh/dit/scripts/convert/convert_handoff_state26_relee_pose14.sh \
  /data/shared_folder/datasets/dit/raw/handoff_jointpos_scripted_success120_0701_225643 \
  --output-dir /data/shared_folder/datasets/dit/lerobot/handoff_jointpos_scripted_success120_0701_225643_state26_relee_pose14_h50_obs2 \
  --repo-id local/handoff_jointpos_scripted_success120_0701_225643_state26_relee_pose14_h50_obs2 \
  --horizon 50 \
  --n-obs-steps 2
```

Train on two GPUs:

```bash
/home/qsh/dit/scripts/train/train_handoff_state26_relee_pose14_mtdp.sh \
  /data/shared_folder/datasets/dit/lerobot/handoff_jointpos_scripted_success120_0701_225643_state26_relee_pose14_h50_obs2 \
  --profile /home/qsh/dit/configs/train/handoff_state26_relee_pose14_2gpu_bs16_accum4_50k.json \
  --run-name handoff_state26_relee_pose14_mtdp_2gpu_bs16_accum4_50k \
  --device cuda \
  --num-processes 2 \
  --gpu-ids 0,1
```

Closed-loop eval:

```bash
/home/qsh/dit/scripts/eval/closed_loop_handoff_state26_relee_pose14.sh \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --task Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0 \
  --checkpoint /data/shared_folder/datasets/dit/runs/<run_name>/checkpoints/050000 \
  --policy-python /home/qsh/miniconda3/envs/dit_lerobot_main/bin/python \
  --policy-device cuda:0 \
  --output-dir /data/shared_folder/datasets/dit/eval/<eval_name> \
  --max-steps 2200 \
  --save-video
```
