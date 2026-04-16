from __future__ import annotations


def make_mujoco_env(env_id: str, seed: int, cfg):
    import gymnasium as gym

    env = gym.make(env_id)
    env.reset(seed=seed)
    return env
