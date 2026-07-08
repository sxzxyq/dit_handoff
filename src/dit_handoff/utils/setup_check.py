"""Environment setup checks for the DIT handoff workspace."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from dit_handoff.constants import CONDA_ROOT, DATA_ROOT, ISAACLAB_ROOT, JOINT_POS_TASK_ID, WORKSPACE_ROOT
from dit_handoff.env import register_tasks
from dit_handoff.utils.io import read_json, write_json

WORKSPACE_SRC = WORKSPACE_ROOT / "src"
ISAACLAB_SOURCE = ISAACLAB_ROOT / "source"


def _check_import(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"ok": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _active_env_executable(name: str) -> str | None:
    env_local = Path(sys.executable).resolve().parent / name
    if env_local.is_file():
        return str(env_local)
    return shutil.which(name)


def _conda_env_exists(env_name: str) -> bool | None:
    conda = shutil.which("conda")
    if conda is None:
        return None
    try:
        out = subprocess.check_output([conda, "env", "list"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return any(line.split() and line.split()[0] == env_name for line in out.splitlines())


def _torch_cuda() -> dict[str, Any]:
    try:
        import torch

        return {"available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def _multi_task_dit_entry() -> dict[str, Any]:
    try:
        importlib.import_module("lerobot.common.policies.multi_task_dit")
        return {"ok": True, "module": "lerobot.common.policies.multi_task_dit"}
    except Exception as first_exc:
        try:
            importlib.import_module("lerobot.policies.multi_task_dit")
            return {"ok": True, "module": "lerobot.policies.multi_task_dit"}
        except Exception as second_exc:
            return {"ok": False, "errors": [repr(first_exc), repr(second_exc)]}


def _handoff_task_registration() -> dict[str, Any]:
    try:
        import gymnasium as gym

        register_tasks()
        gym.spec(JOINT_POS_TASK_ID)
        return {"ok": True, "task_id": JOINT_POS_TASK_ID}
    except Exception as exc:
        return {"ok": False, "task_id": JOINT_POS_TASK_ID, "error": repr(exc)}


def _headless_reset(device: str, early_output: Path | None, results: dict[str, Any]) -> dict[str, Any]:
    app = None
    env = None
    try:
        from isaaclab.app import AppLauncher
        import gymnasium as gym

        launcher_args = {
            "headless": True,
            "enable_cameras": True,
            "device": device,
            "renderer": "RayTracedLighting",
            "livestream": -1,
            "xr": False,
        }
        app = AppLauncher(launcher_args).app
        from isaaclab_tasks.utils import parse_env_cfg

        register_tasks()
        env_cfg = parse_env_cfg(JOINT_POS_TASK_ID, device=device, num_envs=1, use_fabric=True)
        env = gym.make(JOINT_POS_TASK_ID, cfg=env_cfg)
        env.reset()
        headless = {"requested": True, "ok": True}
    except Exception as exc:
        headless = {"requested": True, "ok": False, "error": repr(exc)}
    finally:
        results["headless_reset"] = headless
        # Isaac Sim may terminate the process during app.close(), so make the
        # early report a complete report before closing the Kit app.
        results["failures"] = _failures(results, "isaaclab")
        results["ok"] = not results["failures"]
        if early_output is not None:
            write_json(early_output, results)
        if env is not None:
            env.close()
        if app is not None:
            app.close()
    return headless


def run_lerobot_checks(env_name: str) -> dict[str, Any]:
    results: dict[str, Any] = {
        "mode": "lerobot",
        "conda_env": {"name": env_name, "exists": _conda_env_exists(env_name)},
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "cli": {
            "lerobot-train": _active_env_executable("lerobot-train"),
            "accelerate": _active_env_executable("accelerate"),
        },
        "imports": {
            "torch": _check_import("torch"),
            "lerobot": _check_import("lerobot"),
        },
        "torch_cuda": _torch_cuda(),
        "multi_task_dit_entry": _multi_task_dit_entry(),
        "handoff_task_registration": {"skipped": True, "reason": "checked in IsaacLab Python"},
        "headless_reset": {"requested": False, "ok": None, "skipped": True},
    }
    results["ok"] = not _failures(results, "lerobot")
    return results


def run_isaaclab_checks(
    env_name: str,
    *,
    headless_reset: bool = False,
    device: str = "cuda:0",
    early_output: Path | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "mode": "isaaclab",
        "conda_env": {"name": env_name, "exists": _conda_env_exists(env_name)},
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "cli": {"lerobot-train": {"skipped": True, "reason": "checked in LeRobot Python"}},
        "imports": {
            "torch": _check_import("torch"),
            "gymnasium": _check_import("gymnasium"),
            "isaaclab": _check_import("isaaclab"),
            "isaaclab_tasks": _check_import("isaaclab_tasks"),
        },
        "torch_cuda": _torch_cuda(),
        "multi_task_dit_entry": {"skipped": True, "reason": "checked in LeRobot Python"},
        "handoff_task_registration": _handoff_task_registration(),
        "headless_reset": {"requested": headless_reset, "ok": None},
    }
    if headless_reset:
        _headless_reset(device, early_output, results)
    results["ok"] = not _failures(results, "isaaclab")
    return results


def _failures(results: dict[str, Any], mode: str) -> list[str]:
    failures: list[str] = []
    if mode == "full":
        for label in ("lerobot", "isaaclab"):
            sub = results.get(label, {})
            if not sub.get("ok"):
                failures.append(label)
        return failures

    if results.get("conda_env", {}).get("exists") is False:
        failures.append("conda_env")
    if not results.get("imports", {}).get("torch", {}).get("ok"):
        failures.append("torch")
    if not results.get("torch_cuda", {}).get("available"):
        failures.append("torch_cuda")
    if mode == "lerobot":
        if not results.get("imports", {}).get("lerobot", {}).get("ok"):
            failures.append("lerobot")
        if not results.get("multi_task_dit_entry", {}).get("ok"):
            failures.append("multi_task_dit")
        if not results.get("cli", {}).get("lerobot-train"):
            failures.append("lerobot-train")
    elif mode == "isaaclab":
        for name in ("gymnasium", "isaaclab", "isaaclab_tasks"):
            if not results.get("imports", {}).get(name, {}).get("ok"):
                failures.append(name)
        if not results.get("handoff_task_registration", {}).get("ok"):
            failures.append("handoff_task_registration")
        headless = results.get("headless_reset", {})
        if headless.get("requested") and not headless.get("ok"):
            failures.append("headless_reset")
    return failures


def _subreport_path(output: Path, label: str) -> Path:
    return output.with_name(f"{output.stem}_{label}{output.suffix}")


def _subprocess_env(*, include_isaaclab_source: bool) -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(WORKSPACE_SRC)]
    if include_isaaclab_source:
        paths.append(str(ISAACLAB_SOURCE))
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = ":".join(paths)
    return env


def _run_subcheck(python: Path, label: str, mode: str, output: Path, extra_args: list[str]) -> dict[str, Any]:
    if not python.is_file():
        return {
            "mode": mode,
            "ok": False,
            "subprocess": {"python": str(python), "returncode": None, "error": "python executable not found"},
        }
    command = [str(python), "-m", "dit_handoff.utils.setup_check", "--mode", mode, "--output", str(output), *extra_args]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=_subprocess_env(include_isaaclab_source=(mode == "isaaclab")),
    )
    if output.exists():
        report = read_json(output)
    else:
        report = {"mode": mode, "ok": False, "error": "subcheck did not write report"}
    report["subprocess"] = {
        "label": label,
        "python": str(python),
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        report["ok"] = False
    return report


def run_full_checks(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    lerobot_report_path = _subreport_path(output, "lerobot")
    isaaclab_report_path = _subreport_path(output, "isaaclab")
    lerobot = _run_subcheck(
        Path(args.lerobot_python),
        "lerobot",
        "lerobot",
        lerobot_report_path,
        ["--env-name", args.env_name],
    )
    isaaclab_args = ["--env-name", args.isaaclab_env_name, "--device", args.device]
    if args.headless_reset:
        isaaclab_args.append("--headless-reset")
    isaaclab = _run_subcheck(
        Path(args.isaaclab_python),
        "isaaclab",
        "isaaclab",
        isaaclab_report_path,
        isaaclab_args,
    )
    results = {
        "mode": "full",
        "ok": bool(lerobot.get("ok")) and bool(isaaclab.get("ok")),
        "lerobot_python": args.lerobot_python,
        "isaaclab_python": args.isaaclab_python,
        "lerobot_report": str(lerobot_report_path),
        "isaaclab_report": str(isaaclab_report_path),
        "lerobot": lerobot,
        "isaaclab": isaaclab,
    }
    results["failures"] = _failures(results, "full")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check DIT handoff setup.")
    parser.add_argument("--mode", choices=("full", "lerobot", "isaaclab"), default="full")
    parser.add_argument("--env-name", default="dit_lerobot_main")
    parser.add_argument("--isaaclab-env-name", default="env_isaaclab")
    parser.add_argument("--lerobot-python", default=str(CONDA_ROOT / "envs/dit_lerobot_main/bin/python"))
    parser.add_argument("--isaaclab-python", default=str(CONDA_ROOT / "envs/env_isaaclab/bin/python"))
    parser.add_argument("--headless-reset", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "reports/setup_check.json")
    args = parser.parse_args(argv)
    if args.mode == "isaaclab" and args.env_name == parser.get_default("env_name"):
        args.env_name = args.isaaclab_env_name

    if args.mode == "lerobot":
        results = run_lerobot_checks(args.env_name)
    elif args.mode == "isaaclab":
        results = run_isaaclab_checks(
            args.env_name,
            headless_reset=args.headless_reset,
            device=args.device,
            early_output=args.output if args.headless_reset else None,
        )
    else:
        results = run_full_checks(args)

    results["failures"] = _failures(results, args.mode)
    if args.mode != "full":
        results["ok"] = not results["failures"]
    write_json(args.output, results)
    print(json.dumps(results, indent=2))
    return 0 if results.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
