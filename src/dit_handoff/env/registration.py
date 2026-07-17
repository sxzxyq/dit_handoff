"""Gymnasium registration for the DIT handoff IsaacLab tasks."""

from __future__ import annotations

from dit_handoff.constants import BAR_HANDOFF_JOINT_POS_TASK_ID, IK_TASK_ID, JOINT_POS_TASK_ID


def _is_registered(gym, task_id: str) -> bool:
    try:
        gym.spec(task_id)
    except Exception:
        return False
    return True


def register_tasks() -> list[str]:
    """Register the local handoff task ids if Gymnasium is available."""
    import gymnasium as gym

    registered: list[str] = []
    specs = (
        (
            IK_TASK_ID,
            "dit_handoff.env.handoff_env_cfg:CubeHandoffYellowRedDualFrankaIKRelVisuomotorEnvCfg",
        ),
        (
            JOINT_POS_TASK_ID,
            "dit_handoff.env.handoff_env_cfg:CubeHandoffYellowRedDualFrankaJointPosVisuomotorEnvCfg",
        ),
        (
            BAR_HANDOFF_JOINT_POS_TASK_ID,
            "dit_handoff.env.bar_handoff_env_cfg:BarHandoffDualFrankaJointPosVisuomotorEnvCfg",
        ),
    )
    for task_id, cfg_entry_point in specs:
        if not _is_registered(gym, task_id):
            gym.register(
                id=task_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                kwargs={"env_cfg_entry_point": cfg_entry_point},
                disable_env_checker=True,
            )
        registered.append(task_id)
    return registered

