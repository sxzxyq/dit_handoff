"""Subprocess LeRobot policy worker for IsaacLab closed-loop evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dit_handoff.constants import BAR_HANDOFF_LANGUAGE_INSTRUCTION, CAMERA_OBS_FEATURES
from dit_handoff.eval.closed_loop_bar_handoff_state26_absjoint18 import _load_policy_stack, _repeat_obs_window


def _action_chunk_from_npz(policy_stack: dict[str, Any], input_npz: Path, language: str) -> list[list[list[float]]]:
    cfg = policy_stack["cfg"]
    policy = policy_stack["policy"]
    preprocessor = policy_stack["preprocessor"]
    postprocessor = policy_stack["postprocessor"]
    device = str(policy.config.device)
    n_obs_steps = int(cfg.policy.n_obs_steps)

    with np.load(input_npz) as data:
        batch: dict[str, Any] = {
            "observation.state": torch.as_tensor(data["state"], dtype=torch.float32, device=device).unsqueeze(0),
            "task": [language],
        }
        for camera in CAMERA_OBS_FEATURES:
            if camera in data:
                batch[f"observation.images.{camera}"] = torch.as_tensor(
                    data[camera], dtype=torch.float32, device=device
                ).unsqueeze(0)
    batch = _repeat_obs_window(batch, n_obs_steps)
    batch = preprocessor(batch)
    with torch.no_grad():
        prepared = policy._prepare_batch(batch)
        normalized_chunk = policy._generate_actions(prepared)
        chunk = postprocessor(normalized_chunk)
    if chunk.ndim == 2:
        chunk = chunk.unsqueeze(1)
    return chunk.detach().cpu().tolist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LeRobot policy subprocess worker.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    policy_stack = _load_policy_stack(args.checkpoint, args.device)
    ready = {
        "ready": True,
        "policy_dir": str(policy_stack["policy_dir"]),
        "n_action_steps": int(policy_stack["cfg"].policy.n_action_steps),
    }
    print(json.dumps(ready), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            chunk = _action_chunk_from_npz(
                policy_stack,
                Path(request["input_npz"]),
                request.get("language") or BAR_HANDOFF_LANGUAGE_INSTRUCTION,
            )
            print(json.dumps({"ok": True, "request_id": request.get("request_id"), "action_chunk": chunk}), flush=True)
        except Exception as exc:  # pragma: no cover - exercised through subprocess smoke tests
            print(
                json.dumps(
                    {
                        "ok": False,
                        "request_id": request.get("request_id") if "request" in locals() else None,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
