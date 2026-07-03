"""Dataset wrapper that injects anchored relative EE pose14 action chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from dit_handoff.constants import LEROBOT_RELEE_POSE14_CONTRACT_VERSION, POSE14_ACTION_DIM
from dit_handoff.utils.io import read_json


def _item_index(value: Any, fallback: int) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return int(value.detach().cpu().item())
        return int(value.reshape(-1)[0].detach().cpu().item())
    if isinstance(value, np.ndarray):
        return int(value.reshape(-1)[0])
    if isinstance(value, list | tuple):
        return int(value[0]) if value else fallback
    if value is None:
        return fallback
    return int(value)


class AnchoredRelativeEePose14Dataset(torch.utils.data.Dataset):
    """Wrap a LeRobotDataset and replace its action with anchored pose14 chunks."""

    def __init__(self, base_dataset: Any, dataset_root: str | Path | None = None):
        self.base_dataset = base_dataset
        self.root = Path(dataset_root or getattr(base_dataset, "root"))
        manifest = read_json(self.root / "manifest.json")
        if manifest.get("contract") != LEROBOT_RELEE_POSE14_CONTRACT_VERSION:
            raise ValueError(
                f"dataset contract must be {LEROBOT_RELEE_POSE14_CONTRACT_VERSION}, got {manifest.get('contract')!r}"
            )
        sidecar = manifest.get("action_chunk_sidecar") or {}
        metadata_rel = sidecar.get("metadata", "action_chunks/metadata.json")
        metadata = read_json(self.root / metadata_rel)
        actions_rel = sidecar.get("actions") or metadata["actions_path"]
        mask_rel = sidecar.get("action_is_pad") or metadata["action_is_pad_path"]
        self.action_chunks = np.load(self.root / actions_rel, mmap_mode="r")
        self.action_is_pad = np.load(self.root / mask_rel, mmap_mode="r")
        if self.action_chunks.ndim != 3 or self.action_chunks.shape[-1] != POSE14_ACTION_DIM:
            raise ValueError(f"invalid action chunk sidecar shape {self.action_chunks.shape}")
        if self.action_is_pad.shape != self.action_chunks.shape[:2]:
            raise ValueError(
                f"action_is_pad shape {self.action_is_pad.shape} does not match action chunks {self.action_chunks.shape}"
            )
        self.sidecar_metadata = metadata
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base_dataset[idx]
        absolute_index = _item_index(item.get("index"), idx)
        item["action"] = torch.from_numpy(np.asarray(self.action_chunks[absolute_index]).copy()).to(torch.float32)
        item["action_is_pad"] = torch.from_numpy(np.asarray(self.action_is_pad[absolute_index]).copy()).to(torch.bool)
        return item

    @property
    def meta(self):
        return self.base_dataset.meta

    @property
    def episodes(self):
        return self.base_dataset.episodes

    @property
    def num_frames(self):
        return self.base_dataset.num_frames

    @property
    def num_episodes(self):
        return self.base_dataset.num_episodes

    @property
    def absolute_to_relative_idx(self):
        return self.base_dataset.absolute_to_relative_idx

    @property
    def hf_dataset(self):
        return self.base_dataset.hf_dataset

    def __getattr__(self, name: str):
        return getattr(self.base_dataset, name)


def wrap_train_eval_datasets(train_dataset: Any, eval_dataset: Any | None, dataset_root: str | Path | None = None):
    wrapped_train = AnchoredRelativeEePose14Dataset(train_dataset, dataset_root=dataset_root)
    wrapped_eval = None if eval_dataset is None else AnchoredRelativeEePose14Dataset(eval_dataset, dataset_root=dataset_root)
    return wrapped_train, wrapped_eval
