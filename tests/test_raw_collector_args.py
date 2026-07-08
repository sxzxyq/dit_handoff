from dit_handoff.collect.raw_collector import parse_args
from dit_handoff.constants import WORKSPACE_ROOT


def test_camera_warmup_default_from_config():
    args = parse_args([
        "--config",
        str(WORKSPACE_ROOT / "configs/collect/raw_jointpos_default.json"),
        "--dataset-name",
        "unit_test_dataset",
    ])
    assert args.camera_warmup_steps == 1


def test_camera_warmup_cli_override():
    args = parse_args(["--camera-warmup-steps", "3"])
    assert args.camera_warmup_steps == 3
