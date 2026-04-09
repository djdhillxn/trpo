from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from trpo_repro.algos.trpo import TRPOAgent
from trpo_repro.data.buffer import TrajectoryBuffer
from trpo_repro.models.policies import make_policy
from trpo_repro.models.value_functions import ValueFunction
from trpo_repro.utils.io import ensure_dir
from trpo_repro.utils.logger import JsonlLogger
from trpo_repro.utils.torch_utils import RunningMeanStd, to_tensor

class Runner:
    def __init__(self, env, cfg, output_dir: str | Path, device: str = "cpu") -> None:
        self.env = env
        self.cfg = cfg
        self.output_dir = ensure_dir(output_dir)
        self.device = torch.device(device)
        self.logger = JsonlLogger(self.output_dir)
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")

        self.policy = make_policy(env.observation_space, env.action_space, cfg)
        self.value_fn = ValueFunction(tuple(env.observation_space.shape), cfg)
        self.agent = TRPOAgent(self.policy, self.value_fn, cfg, self.device)

        self.normalize_obs = bool(cfg.train.get("normalize_obs", False)) and len(env.observation_space.shape) == 1
        self.obs_rms = RunningMeanStd(shape=tuple(env.observation_space.shape)) if self.normalize_obs else None

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        if self.obs_rms is not None:
            self.obs_rms.update(obs[None, ...])
            obs = self.obs_rms.normalize(obs)
        return obs.astype(np.float32)
    
    def train(self) -> None:
        obs, _ = self.env.reset()
        obs = self._preprocess_obs(obs)
        ep_ret = 0.0
        ep_len = 0
        episode_returns: list[float] = []
        episode_lengths: list[int] = []

        obs_shape = tuple(self.env.observation_space.shape)
        if hasattr(self.env.action_space, "n"):
            act_shape = ()
            act_dtype = np.int64
        else:
            act_shape = tuple(self.env.action_space.shape)
            act_dtype = np.float32
        
        total_epochs = int(self.cfg.train.epochs)
        steps_per_epoch = int(self.cfg.train.steps_per_epoch)
        max_ep_len = int(self.cfg.train.get("max_ep_len", 1000))
        gamma = float(self.cfg.algo.gamma)
        lam = float(self.cfg.algo.get("lam", 1.0))
        advantage_mode = str(self.cfg.algo.get("advantage_mode", "mc"))

        for epoch in range(1, total_epochs + 1):
            start_time = time.time()
            buffer = TrajectoryBuffer(
                obs_shape=obs_shape,
                act_shape=act_shape,
                size=steps_per_epoch,
                gamma=gamma,
                lam=lam,
                advantage_mode=advantage_mode,
                act_dtype=act_dtype,
            )

            for t in range(steps_per_epoch):
                obs_tensor = to_tensor(obs[None, ...], self.device, dtype=torch.float32)
                action_t, value_t, logp_t = self.agent.step(obs_tensor)
                action = action_t.squeeze(0).cpu().numpy()
                value = float(value_t.squeeze(0).cpu().item())
                logp = float(logp_t.squeeze(0).cpu().item() if logp_t.ndim > 0 else float(logp_t.cpu().item()))

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_obs = self._preprocess_obs(next_obs)

                buffer.store(obs, action, float(reward), value, logp)
                ep_ret += float(reward)
                ep_len += 1

                timeout = ep_len >= max_ep_len
                terminal = terminated or truncated or timeout
                epoch_ended = t == steps_per_epoch - 1

                if terminal or epoch_ended:
                    if terminal:
                        last_val = 0.0 if terminated else float(
                            self.agent.value(to_tensor(next_obs[None, ...], self.device, dtype=torch.float32)).cpu().item()
                        )
                    else:
                        last_val = float(
                            self.agent.value(to_tensor(next_obs[None, ...], self.device, dtype=torch.float32)).cpu().item()
                        )
                    buffer.finish_path(last_val)

                    if terminal:
                        episode_returns.append(ep_ret)
                        episode_lengths.append(ep_len)
                        obs, _ = self.env.reset()
                        obs = self._preprocess_obs(obs)
                        ep_ret = 0.0
                        ep_len = 0
                    else:
                        obs = next_obs
                else:
                    obs = next_obs
            
            batch = buffer.get(self.device)
            stats = self.agent.update(batch)
            wall_time = time.time() - start_time

            record = {
                "epoch": epoch,
                "episodes_in_epoch": len(episode_returns),
                "ep_return_mean": float(np.mean(episode_returns)) if episode_returns else float("nan"),
                "ep_return_std": float(np.std(episode_returns)) if episode_returns else float("nan"),
                "ep_len_mean": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
                "policy_loss_before": stats.policy_loss_before,
                "policy_loss_after": stats.policy_loss_after,
                "value_loss_before": stats.value_loss_before,
                "value_loss_after": stats.value_loss_after,
                "entropy": stats.entropy,
                "approx_kl": stats.approx_kl,
                "line_search_success": int(stats.line_search_success),
                "cg_norm": stats.cg_norm,
                "wall_time_sec": wall_time,
            }
            self.logger.log(record)
            print(record)
            episode_returns.clear()
            episode_lengths.clear()

            save_every = int(self.cfg.train.get("save_interval", 10))
            if epoch % save_every == 0 or epoch == total_epochs:
                self.save_checkpoint(epoch)
        
        self.logger.close()

    def save_checkpoint(self, epoch: int) -> None:
        ckpt = {
            "epoch": epoch,
            "policy": self.policy.state_dict(),
            "value_fn": self.value_fn.state_dict(),
            "obs_rms_mean": None if self.obs_rms is None else self.obs_rms.mean,
            "obs_rms_var": None if self.obs_rms is None else self.obs_rms.var,
            "obs_rms_count": None if self.obs_rms is None else self.obs_rms.count,
        }
        torch.save(ckpt, self.checkpoint_dir / f"epoch_{epoch:04d}.pt")














