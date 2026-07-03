from pathlib import Path

from dit_handoff.train.handoff_state26_relee_pose14_train import build_training_config
from dit_handoff.utils.io import ensure_dir, write_json


def test_pose14_training_config_checks_sidecar_shape(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    ensure_dir(dataset_dir / "action_chunks")
    write_json(
        dataset_dir / "manifest.json",
        {
            "contract": "state26_relee_pose14_v1",
            "fps": 50.0,
            "repo_id": "local/test_pose14",
            "action_chunk_sidecar": {"metadata": "action_chunks/metadata.json"},
        },
    )
    write_json(
        dataset_dir / "action_chunks" / "metadata.json",
        {"horizon": 50, "n_obs_steps": 2, "actions_path": "action_chunks/actions.npy", "action_is_pad_path": "action_chunks/action_is_pad.npy"},
    )
    cfg = build_training_config(
        dataset_dir,
        {"horizon": 50, "n_obs_steps": 2, "n_action_steps": 40, "physical_batch_size": 16, "gradient_accumulation_steps": 4},
        run_name="test_pose14",
        smoke=False,
    )
    assert cfg["contract"] == "state26_relee_pose14_v1"
    assert cfg["action_dim"] == 14
    assert cfg["effective_batch_size"] == 64
    assert cfg["do_mask_loss_for_padding"] is True
