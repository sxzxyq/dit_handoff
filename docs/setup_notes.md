# Setup Notes

Official LeRobot MultiTask DiT docs checked on 2026-07-01 list:

- install extra: `pip install "lerobot[multi_task_dit]"`
- train entry: `lerobot-train`
- policy type: `policy.type=multi_task_dit`
- 30 Hz reference: horizon about 32 and `n_action_steps=24`
- default objective: diffusion with DDPM

Observed on this server on 2026-07-01:

- `scripts/setup/create_lerobot_env.sh` created conda env `dit_lerobot`.
- `pip install "lerobot[multi_task_dit]"` installed PyPI `lerobot==0.4.4`.
- pip warned: `lerobot 0.4.4 does not provide the extra multi-task-dit`.
- In `dit_lerobot`, `torch==2.10.0+cu128`, CUDA is available, and `lerobot-train` exists.
- In `dit_lerobot`, `lerobot.common.policies.multi_task_dit` and `lerobot.policies.multi_task_dit` are not importable.
- Therefore formal MultiTask DiT training needs a LeRobot version newer than the current PyPI 0.4.4 package, likely the official main/v0.5.1 line or a source install matching the docs.
- IsaacLab is intentionally not installed in `dit_lerobot`; collect/closed-loop eval should use the IsaacLab launcher/Python with `PYTHONPATH=/home/qsh/dit/src`.
- A second source env `dit_lerobot_main` was created with Python 3.12.13 and GitHub LeRobot commit `8414188db0b178b947985a7a9a91314708837315`, producing `lerobot==0.5.2`.
- In `dit_lerobot_main`, `lerobot-train`, torch/CUDA, and `lerobot.policies.multi_task_dit` are available after installing `training,multi-task-dit` extras.

Create the LeRobot environment:

```bash
cd /home/qsh/dit
DIT_LEROBOT_ENV=dit_lerobot_main DIT_LEROBOT_PYTHON_VERSION=3.12 bash scripts/setup/create_lerobot_env.sh
```

Check setup:

```bash
cd /home/qsh/dit
bash scripts/setup/check_setup.sh --headless-reset --device cuda:0
```


Wrapper behavior on 2026-07-01:

- Raw collection defaults to `camera_warmup_steps=1`: after each reset, the collector sends one no-op current Joint-Pos action and does not record that warmup frame, avoiding stale Isaac Sim camera buffers at recorded step 0.
- `scripts/setup/check_setup.sh` is an aggregate checker. It runs LeRobot checks in `/home/qsh/miniconda3/envs/dit_lerobot_main/bin/python` and IsaacLab checks in `/home/qsh/miniconda3/envs/env_isaaclab/bin/python`.
- `scripts/validate`, `scripts/convert`, `scripts/train`, and `scripts/eval/open_loop` use the LeRobot Python by default.
- `scripts/collect` and `scripts/eval/closed_loop` use the IsaacLab Python by default.
- The old repository reference was cloned to gitignored `/home/qsh/dit/_reference/dykj_lbm` only for parity checks; runtime scripts do not import it.

IsaacLab collect and closed-loop eval use `/home/qsh/miniconda3/envs/env_isaaclab` with Isaac Sim 5.0 pip packages. Training and LeRobot policy loading use `/home/qsh/miniconda3/envs/dit_lerobot_main`. The shared interface between the two environments is file-based:

- raw data: `/data/shared_folder/datasets/dit/raw`
- converted dataset: `/data/shared_folder/datasets/dit/lerobot`
- train config and runs: `/data/shared_folder/datasets/dit/runs`
- checkpoint config/checkpoints: `/data/shared_folder/datasets/dit/checkpoints`
- eval outputs: `/data/shared_folder/datasets/dit/eval`

If the installed LeRobot API changes and `LeRobotDataset.create/add_frame/save_episode/finalize` is not available, the converter records the failure in `conversion_summary.json`. The portable fallback is for smoke validation only; formal training should use an official LeRobot-readable dataset.

IsaacLab notes on 2026-07-01:

- `/home/qsh/IsaacLab` is version 2.2.1 and matches Isaac Sim 5.0.
- `/home/qsh/miniconda3/envs/env_isaaclab` has Python 3.11.13, torch 2.7.0+cu128, Isaac Sim 5.0.0.0 packages, editable IsaacLab packages, and CUDA device count 4.
- `omni.physics` is provided after Kit starts; direct pre-launch import can fail. Scripts must call `AppLauncher(...).app` before importing task utilities that need Omni extensions.
- Do not use `CUDA_VISIBLE_DEVICES` for Isaac Sim smoke runs on this machine; Omniverse logs warn that it can confuse CUDA/Omniverse device enumeration. Use `--device cuda:0` for IsaacLab and `--policy-device cuda:1` for the LeRobot worker instead.
- `env_isaaclab` cannot install this LeRobot source directly because the verified LeRobot commit requires Python >=3.12. Closed-loop checkpoint eval therefore starts a policy subprocess with `/home/qsh/miniconda3/envs/dit_lerobot_main/bin/python`.


## Verified on 2026-07-01

LeRobot source env:

- Env: `/home/qsh/miniconda3/envs/dit_lerobot_main`, Python 3.12.13.
- LeRobot: `0.5.2`; current check reports torch `2.11.0+cu130` and CUDA device count 4.
- `lerobot-train` is available at `/home/qsh/miniconda3/envs/dit_lerobot_main/bin/lerobot-train`.
- MultiTask DiT module is `lerobot.policies.multi_task_dit`.
- `LeRobotDataset.create/add_frame/save_episode/finalize` successfully wrote an official dataset for `synthetic_train_smoke_long_state26_absjoint18`.

Training smoke:

- Single GPU 0 with official-time model and per-process batch 32 reached real forward but OOMed at about 21.23 GiB used.
- Two GPUs 0 and 1 via `accelerate launch --multi_gpu --num_processes=2 --gpu_ids=0,1` completed a 2-step official-time smoke with per-process batch 16.
- Smoke run: `/data/shared_folder/datasets/dit/runs/synthetic_train_smoke_2gpu_0701a`.
- Checkpoint: `/data/shared_folder/datasets/dit/runs/synthetic_train_smoke_2gpu_0701a/checkpoints/000002/pretrained_model`.
- LeRobot 0.5.2 treats `batch_size` as per process under Accelerate DDP; effective batch is `batch_size * num_processes`.
- LeRobot 0.5.2 train CLI has no `dataset.push_to_hub` or gradient accumulation flag in the exposed training args. The wrapper disables `policy.push_to_hub` and `save_checkpoint_to_hub`, and records desired gradient accumulation separately.

Open-loop eval:

- Target-only eval still reports zero MAE/MSE for contract smoke.
- Real checkpoint eval loads `TrainPipelineConfig`, official temporal `LeRobotDataset`, `make_policy`, and `make_pre_post_processors`.
- For 2 samples from the validation episode of the two-GPU smoke checkpoint, `first_action_mae=0.025819997303187847` and `chunk_mae=0.024106363765895367`.

IsaacLab / Isaac Sim:

- `dit_lerobot_main` is for training/policy loading and does not import IsaacLab.
- `/home/qsh/miniconda3/envs/env_isaaclab` is for IsaacLab collect/eval and has pip Isaac Sim 5.0.0.0.
- Aggregate setup check passed: `/data/shared_folder/datasets/dit/reports/setup_check_full_noheadless_0701b.json` with `ok=true` and `failures=[]`.
- Headless reset passed through aggregate setup check: `/data/shared_folder/datasets/dit/reports/setup_check_full_headless_0701b.json` with `ok=true`, `failures=[]`, and `isaaclab.headless_reset.ok=true`.
- Real IsaacLab raw collector smoke passed: `/data/shared_folder/datasets/dit/raw/handoff_jointpos_raw_isaac_smoke_0701b`, 1 episode, 2 steps, 3 cameras.
- Raw validator passed with 0 errors and 2 hold-current warnings; official LeRobot conversion passed at `/data/shared_folder/datasets/dit/lerobot/handoff_jointpos_raw_isaac_smoke_0701b_state26_absjoint18`.
- Closed-loop hold-current smoke passed at `/data/shared_folder/datasets/dit/eval/closed_loop_hold_current_isaac_smoke_0701b`, 2 steps and 2 policy calls.
- Closed-loop checkpoint smoke passed at `/data/shared_folder/datasets/dit/eval/closed_loop_policy_synthetic_smoke_0701a`, using `env_isaaclab` on `cuda:0` and a LeRobot policy worker on `cuda:1`.
