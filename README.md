# DIT Handoff Workspace

Clean workspace for the IsaacLab dual-arm yellow-to-red handoff task and a single LeRobot MultiTask DiT pipeline:

```text
generic raw collector
-> raw validator
-> LeRobot conversion
-> 3 cameras + state26 -> absjoint18
-> official LeRobot MultiTask DiT training
-> open-loop eval
-> closed-loop eval
```

The old `sxzxyq/dykj_lbm` GitHub repo is read-only reference only. This workspace does not rely on old `/home/ubuntu/...` paths, old raw demos, old experiments, or old checkpoints.

Large outputs are fixed under `/data/shared_folder/datasets/dit`:

```text
raw/ lerobot/ runs/ checkpoints/ eval/ reports/ videos/ cache/
```

## Layout

```text
/home/qsh/dit/
  configs/
  docs/
  scripts/
  src/dit_handoff/
  tests/
```

## Setup

```bash
cd /home/qsh/dit
DIT_LEROBOT_ENV=dit_lerobot_main bash scripts/setup/create_lerobot_env.sh
bash scripts/setup/check_setup.sh --headless-reset --device cuda:0
```

The wrapper scripts choose the correct Python environment themselves. Collect and closed-loop eval use `/home/qsh/miniconda3/envs/env_isaaclab`; validate, convert, train, open-loop eval, and policy loading use `/home/qsh/miniconda3/envs/dit_lerobot_main`. The old repo reference, when present, lives only under gitignored `/home/qsh/dit/_reference/dykj_lbm` and is not a runtime dependency.

## Collect

Smoke collection with the Joint-Pos hold-current expert. The collector performs one no-op camera warmup step after every reset by default, so the recorded `step_index=0` image is from a refreshed camera buffer.

```bash
cd /home/qsh/dit
bash scripts/collect/collect_raw_jointpos.sh --headless --enable_cameras --device cuda:0 --overwrite
```

For real data, pass a callable expert that returns the actual 18D Joint-Pos target sent to the env:

```bash
bash scripts/collect/collect_raw_jointpos.sh \
  --dataset-name handoff_jointpos_train001 \
  --expert your_module:your_jointpos_expert \
  --episodes 100 \
  --max-steps 800 \
  --headless --enable_cameras --device cuda:0
```

## Validate

```bash
bash scripts/validate/validate_raw.sh /data/shared_folder/datasets/dit/raw/handoff_jointpos_train001
```

The validator rejects missing manifests, broken image paths, non-contiguous steps, bad action dims, and 18D labels not marked as commanded env actions.

## Convert

```bash
bash scripts/convert/convert_handoff_state26_absjoint18.sh \
  /data/shared_folder/datasets/dit/raw/handoff_jointpos_train001
```

Output goes to `/data/shared_folder/datasets/dit/lerobot/<dataset>_state26_absjoint18`.

## Train

Smoke train command generation:

```bash
bash scripts/train/train_handoff_state26_absjoint18_mtdp.sh \
  /data/shared_folder/datasets/dit/lerobot/handoff_jointpos_train001_state26_absjoint18 \
  --run-name smoke_state26_absjoint18 \
  --smoke --dry-run
```

Formal default uses the official-time profile:

```bash
bash scripts/train/train_handoff_state26_absjoint18_mtdp.sh \
  /data/shared_folder/datasets/dit/lerobot/handoff_jointpos_train001_state26_absjoint18 \
  --run-name formal_state26_absjoint18_30k \
  --device cuda --wandb
```

Two-GPU smoke on cards 0 and 1, preserving the official-time horizon/chunk/model shape with per-process batch 16:

```bash
bash scripts/train/train_handoff_state26_absjoint18_mtdp.sh \
  /data/shared_folder/datasets/dit/lerobot/handoff_jointpos_train001_state26_absjoint18 \
  --profile /home/qsh/dit/configs/train/handoff_state26_absjoint18_2gpu_smoke.json \
  --run-name smoke_2gpu_state26_absjoint18 \
  --device cuda --num-processes 2 --gpu-ids 0,1 --smoke
```

The wrapper computes `horizon=round(fps*1.0)` and `n_action_steps=round(fps*0.8)` by default. It records fps, horizon seconds, action seconds, state/action dims, image features, batch size, desired gradient accumulation, actual distributed batch size, objective, diffusion steps, output dir, checkpoint dir, cache dirs, and seed.

## Open-Loop Eval

```bash
bash scripts/eval/open_loop_handoff_state26_absjoint18.sh \
  /data/shared_folder/datasets/dit/lerobot/handoff_jointpos_train001_state26_absjoint18 \
  --checkpoint /data/shared_folder/datasets/dit/runs/formal_state26_absjoint18_30k/checkpoints/last/pretrained_model
```

For a real checkpoint, the evaluator loads LeRobot 0.5.x policy/pre/post processors and reports both `first_action_mae` and `chunk_mae`.

Smoke metric path without a checkpoint:

```bash
bash scripts/eval/open_loop_handoff_state26_absjoint18.sh <dataset_dir> --target-only
```

## Closed-Loop Eval

Hold-current IsaacLab smoke:

```bash
bash scripts/eval/closed_loop_handoff_state26_absjoint18.sh \
  --output-dir /data/shared_folder/datasets/dit/eval/closed_loop_hold_current_smoke \
  --headless --enable_cameras --device cuda:0 --max-steps 2 --smoke-hold-current
```

Checkpoint smoke uses an external LeRobot policy worker because LeRobot 0.5.2 requires Python 3.12 while Isaac Sim 5.0 uses Python 3.11:

```bash
bash scripts/eval/closed_loop_handoff_state26_absjoint18.sh \
  --checkpoint /data/shared_folder/datasets/dit/checkpoints/formal_state26_absjoint18_30k/pretrained_model \
  --output-dir /data/shared_folder/datasets/dit/eval/closed_loop_formal_state26_absjoint18 \
  --headless --enable_cameras --device cuda:0 --max-steps 200 \
  --policy-python /home/qsh/miniconda3/envs/dit_lerobot_main/bin/python --policy-device cuda:1
```

Every policy call records action chunk shape and executed steps in `policy_calls.jsonl`; every env step records the sent 18D action in `steps.jsonl`. Worker request tensors are stored under `policy_worker_inputs/` in the eval output directory.

## Contracts

- Raw schema: [docs/data_contract.md](docs/data_contract.md)
- State/action contract: [docs/state26_absjoint18_contract.md](docs/state26_absjoint18_contract.md)
- Setup notes: [docs/setup_notes.md](docs/setup_notes.md)
- Known pain points: [docs/known_pain_points.md](docs/known_pain_points.md)


## Verified Smoke Artifacts

On 2026-07-01 this workspace verified:

- raw validation: `/data/shared_folder/datasets/dit/raw/synthetic_train_smoke_long`
- official LeRobot conversion: `/data/shared_folder/datasets/dit/lerobot/synthetic_train_smoke_long_state26_absjoint18`
- two-GPU official MultiTask DiT smoke run: `/data/shared_folder/datasets/dit/runs/synthetic_train_smoke_2gpu_0701a`
- checkpoint open-loop eval: `/data/shared_folder/datasets/dit/reports/open_loop_synthetic_train_smoke_2gpu_0701a`
- setup reports: `/data/shared_folder/datasets/dit/reports/setup_check_full_noheadless_0701b.json` and `/data/shared_folder/datasets/dit/reports/setup_check_full_headless_0701b.json`
- real IsaacLab raw smoke: `/data/shared_folder/datasets/dit/raw/handoff_jointpos_raw_isaac_smoke_0701b`
- real IsaacLab raw -> official LeRobot conversion: `/data/shared_folder/datasets/dit/lerobot/handoff_jointpos_raw_isaac_smoke_0701b_state26_absjoint18`
- closed-loop hold-current smoke: `/data/shared_folder/datasets/dit/eval/closed_loop_hold_current_isaac_smoke_0701b`
- closed-loop checkpoint smoke with LeRobot worker: `/data/shared_folder/datasets/dit/eval/closed_loop_policy_synthetic_smoke_0701a`

Isaac Sim/Omni runtime is available through pip Isaac Sim 5.0 in `env_isaaclab`; direct `omni.physics` imports still require launching Kit first through `AppLauncher`.
