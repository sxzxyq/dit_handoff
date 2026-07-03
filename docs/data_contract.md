# Raw Data Contract

This workspace records generic raw handoff data under `/data/shared_folder/datasets/dit/raw/<dataset_name>`.

The collector records:

- directly observed env data: joint positions, joint velocities, TCP pose, gripper opening, cube/target assets when available, and env policy observation terms
- commanded collector action: `raw_env_action` and `joint_target_18_commanded`
- env-returned data: reward, terminated, truncated, and serializable `info`
- post observation after `env.step(action)`
- image paths for `wrist_rgb`, `observer_wrist_rgb`, and `global_rgb`

The collector must not store manually inferred `subtask`, `phase`, `active_arm`, TCP-cube distance, first-close flags, relative pose, progress, or red/yellow stage labels. If an env provides a value directly in observation or info, it may be saved under `env_policy_observation` or `env_info` and remains marked env-provided by its source.

Raw layout:

```text
/data/shared_folder/datasets/dit/raw/<dataset_name>/
  dataset_manifest.json
  validation_report.json
  train_val_split.json
  episodes/
    episode_000000/
      episode_meta.json
      steps.jsonl
      images/
        wrist_rgb/
        observer_wrist_rgb/
        global_rgb/
```

Each step contains `pre_observation`, `action`, `post_observation`, `reward`, `terminated`, `truncated`, and `env_info`. For Joint-Pos 18D collection, `action.joint_target_18_commanded_source` must be `commanded_env_action`.

Randomization is recorded in `dataset_manifest.json` and `episode_meta.json`. Unimplemented randomization items must be marked `not_implemented` or `disabled`.

