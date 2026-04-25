from __future__ import annotations

from trpo_repro.envs.atari import make_atari_env
from trpo_repro.envs.mujoco import make_mujoco_env
from trpo_repro.utils.utils import seed_env_spaces


def make_env(cfg, seed: int):
    env_type = cfg.env.type.lower()
    env_id = cfg.env.id
    if env_type == "atari":
        env = make_atari_env(env_id, seed, cfg)
    elif env_type in {"mujoco", "classic_control"}:
        env = make_mujoco_env(env_id, seed, cfg)
    else:
        raise ValueError(f"Unsupported env type: {cfg.env.type}")
    seed_env_spaces(env, seed)
    return env
