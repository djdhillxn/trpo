from __future__ import annotations

from trpo_repro.envs.atari import make_atari_env
from trpo_repro.envs.mujoco import make_mujoco_env

def make_env(cfg, seed: int):
    env_type = cfg.env.type.lower()
    env_id = cfg.env.id
    if env_type == "atari":
        return make_atari_env(env_id, seed, cfg)
    if env_type in {"mujoco", "classic_control"}:
        return make_mujoco_env(env_id, seed, cfg)
    raise ValueError(f"Unsupported env type: {cfg.env.type}")