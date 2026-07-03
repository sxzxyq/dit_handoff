# Known Pain Points

- The old repository is not local and is only a read-only GitHub reference.
- The new workspace must not depend on old server paths, old experiments, old raw demos, or old checkpoints.
- 18D absolute joint targets must come from commanded Joint-Pos actions, not post-step observed joint state.
- If Joint-Pos commanded expert replay is unstable, training failure does not prove the policy is bad; first check the data/action contract.
- IsaacLab collect/eval and LeRobot training may use different Python environments or launchers.
- The raw collector only saves env-direct fields and commanded actions; it does not save manual phase, active arm, subtask, progress, or distance labels.
- Randomization switches and parameters must be recorded in the manifest. Unimplemented items must be marked clearly.
- Camera names and order are fixed: `wrist_rgb`, `observer_wrist_rgb`, `global_rgb`.
- Train/val split must be by episode, not by frame.
- Horizon and `n_action_steps` should be computed from fps. Do not blindly fix step counts.
- Long action chunks can be risky for contact-rich handoff. Short chunks such as 8 or 16 steps are diagnostic configs, not the formal default.
- Raw images, datasets, videos, checkpoints, reports, and caches must stay under `/data/shared_folder/datasets/dit`, not `/home/qsh/dit`.
- Quaternion convention is `wxyz`; do not silently mix conventions.
- `gripper_opening` must be documented as finger-joint sum or an env-provided scalar.
- LeRobot CLI/API may change. Record observed differences in this file or `setup_notes.md`.
- Directly importing some `omni.*` modules before `AppLauncher(...).app` can fail even when Isaac Sim is installed; launch Kit first, then import IsaacLab task utilities.
- Avoid `CUDA_VISIBLE_DEVICES` for Isaac Sim on this server; pass `--device cuda:N` and let Omniverse enumerate GPUs.
- LeRobot 0.5.2 source requires Python >=3.12, while Isaac Sim 5.0 uses Python 3.11. Closed-loop checkpoint eval must use the `--policy-python` worker bridge instead of installing LeRobot into `env_isaaclab`.


## Observed on 2026-07-01

- PyPI `lerobot==0.4.4` does not provide a working MultiTask DiT extra/module; this workspace uses source LeRobot `0.5.2` in `dit_lerobot_main`.
- `lerobot-train` may not be on `PATH` when using an env Python by absolute path. The wrappers resolve executables from `sys.executable`'s `bin` directory first.
- LeRobot 0.5.2 requires training `output_dir` not to exist when `resume=false`; the wrapper writes preflight config under `/data/shared_folder/datasets/dit/checkpoints/<run>` before launch and copies config into the run dir after training.
- Single-GPU official-time smoke with batch 32 can OOM on a 4090; two-GPU Accelerate DDP with per-process batch 16 completed the smoke.
- Under DDP, LeRobot's `batch_size` is per process. Do not read it as total batch size.
- Isaac Sim 5.0 pip runtime in `env_isaaclab` passes headless reset, raw collect, hold-current closed-loop, and checkpoint closed-loop smoke when imports are ordered after `AppLauncher`.

- Isaac Sim camera observations can contain stale/unrefreshed buffers immediately after `env.reset`; raw collector defaults to one unrecorded no-op camera warmup step before saving `step_index=0`.
