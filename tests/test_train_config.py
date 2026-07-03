from pathlib import Path

from dit_handoff.train.handoff_state26_absjoint18_train import build_training_config
from dit_handoff.utils.io import write_json


def test_official_time_profile_scales_with_fps(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    write_json(
        dataset_dir / "manifest.json",
        {
            "contract": "state26_absjoint18_v1",
            "fps": 50.0,
            "repo_id": "local/test",
        },
    )
    cfg = build_training_config(
        dataset_dir,
        {
            "horizon_seconds": 1.0,
            "n_action_seconds": 0.8,
            "physical_batch_size": 16,
            "gradient_accumulation_steps": 4,
        },
        run_name="test",
        smoke=False,
    )
    assert cfg["horizon"] == 50
    assert cfg["n_action_steps"] == 40
    assert cfg["effective_batch_size"] == 64

