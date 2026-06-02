from collections import deque

import numpy as np


class FrameStackWrapper:
    """Small compatibility wrapper for stacked observations."""

    def __init__(self, env, num_stack: int = 4):
        self.env = env
        self.num_stack = num_stack
        self.frames: deque[np.ndarray] = deque(maxlen=num_stack)
        obs_space = env.observation_space
        self.observation_space = type(obs_space)(
            low=np.repeat(obs_space.low[None, ...], num_stack, axis=0),
            high=np.repeat(obs_space.high[None, ...], num_stack, axis=0),
            dtype=obs_space.dtype,
        )
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(list(self.frames), axis=0)

    def __getattr__(self, item):
        return getattr(self.env, item)


def make_atari_env(env_id: str, seed: int, cfg):
    import ale_py
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing

    gym.register_envs(ale_py)

    # To mimic older DeepMind-style pipelines more closely, we disable sticky actions
    # and use explicit preprocessing frame-skip instead of relying on v5 defaults.
    make_kwargs = dict(
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=float(cfg.env.get("repeat_action_probability", 0.0)),
        full_action_space=bool(cfg.env.get("full_action_space", False)),
    )
    render_mode = cfg.env.get("render_mode")
    if render_mode is not None:
        make_kwargs["render_mode"] = str(render_mode)
    env = gym.make(env_id, **make_kwargs)
    env.reset(seed=seed)
    env = AtariPreprocessing(
        env,
        noop_max=int(cfg.env.get("noop_max", 30)),
        frame_skip=int(cfg.env.get("frame_skip", 4)),
        screen_size=int(cfg.env.get("screen_size", 84)),
        terminal_on_life_loss=bool(cfg.env.get("terminal_on_life_loss", False)),
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = FrameStackWrapper(env, num_stack=int(cfg.env.get("frame_stack", 4)))
    return env
