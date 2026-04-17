from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from trpo_repro.data.buffer import TrajectoryBuffer
from trpo_repro.methods import make_method, resolve_method_name
from trpo_repro.methods.base import MethodUpdateStats
from trpo_repro.utils.io import ensure_dir, write_json
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

        self.method_name = resolve_method_name(cfg)
        self.method = make_method(env.observation_space, env.action_space, cfg, self.device)
        self.method_variant = getattr(self.method, "variant", "default")

        self.estimator = getattr(self.method, "estimator", None)
        self.normalize_weights = bool(cfg.algo.get("normalize_weights", self.estimator != "paper_mc"))
        self.bootstrap_truncated_paths = bool(
            cfg.algo.get("bootstrap_truncated_paths", self.estimator != "paper_mc")
        )
        self.trainable = bool(getattr(self.method, "trainable", False))
        self.supports_checkpoints = bool(getattr(self.method, "supports_checkpoints", False))

        self.normalize_obs = bool(cfg.train.get("normalize_obs", False)) and len(env.observation_space.shape) == 1
        self.obs_rms = RunningMeanStd(shape=tuple(env.observation_space.shape)) if self.normalize_obs else None

        suite = str(cfg.env.get("type", "unknown")).lower()
        self.run_metadata = {
            "method": self.method_name,
            "method_variant": self.method_variant,
            "estimator": self.estimator,
            "env_id": str(cfg.env.id),
            "suite": suite,
            "seed": int(cfg.train.seed),
            "run_name": str(cfg.train.get("run_name", self.output_dir.name)),
            "trainable": self.trainable,
        }
        write_json(self.run_metadata, self.output_dir / "run_metadata.json")

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        if self.obs_rms is not None:
            self.obs_rms.update(obs[None, ...])
            obs = self.obs_rms.normalize(obs)
        return obs.astype(np.float32)

    def _make_buffer(self, target_steps: int, max_ep_len: int):
        obs_shape = tuple(self.env.observation_space.shape)
        if hasattr(self.env.action_space, "n"):
            act_shape = ()
            act_dtype = np.int64
        else:
            act_shape = tuple(self.env.action_space.shape)
            act_dtype = np.float32

        buffer_size = target_steps + (max_ep_len if self.estimator == "paper_mc" else 0)
        return TrajectoryBuffer(
            obs_shape=obs_shape,
            act_shape=act_shape,
            size=buffer_size,
            gamma=float(self.cfg.algo.gamma),
            lam=float(self.cfg.algo.get("lam", 1.0)),
            estimator=self.estimator,
            act_dtype=act_dtype,
            normalize_weights=self.normalize_weights,
        )

    def train(self) -> None:
        obs, _ = self.env.reset()
        obs = self._preprocess_obs(obs)
        ep_ret = 0.0
        ep_len = 0
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        total_env_steps = 0

        total_epochs = int(self.cfg.train.epochs)
        target_steps = int(self.cfg.train.steps_per_epoch)
        max_ep_len = int(self.cfg.train.get("max_ep_len", 1000))

        for epoch in range(1, total_epochs + 1):
            start_time = time.time()
            steps_in_epoch = 0
            buffer = self._make_buffer(target_steps, max_ep_len) if self.trainable else None

            while True:
                obs_tensor = to_tensor(obs[None, ...], self.device, dtype=torch.float32)
                action_t, value_t, logp_t = self.method.act(obs_tensor, deterministic=False)
                action = action_t.squeeze(0).cpu().numpy()
                value = float(value_t.squeeze(0).cpu().item()) if value_t.ndim > 0 else float(value_t.cpu().item())
                logp = float(logp_t.squeeze(0).cpu().item()) if logp_t.ndim > 0 else float(logp_t.cpu().item())

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_obs = self._preprocess_obs(next_obs)

                if buffer is not None:
                    buffer.store(obs, action, float(reward), value, logp)
                ep_ret += float(reward)
                ep_len += 1
                steps_in_epoch += 1
                total_env_steps += 1

                timeout = ep_len >= max_ep_len
                terminal = terminated or truncated or timeout
                reached_target = steps_in_epoch >= target_steps

                if terminal:
                    if buffer is not None:
                        buffer.finish_path(last_val=0.0)
                    episode_returns.append(ep_ret)
                    episode_lengths.append(ep_len)
                    obs, _ = self.env.reset()
                    obs = self._preprocess_obs(obs)
                    ep_ret = 0.0
                    ep_len = 0
                    if reached_target:
                        break
                    continue

                obs = next_obs
                if reached_target:
                    if not self.trainable or self.estimator == "paper_mc":
                        # Continue until this trajectory terminates so the logged batch contains complete paths.
                        continue

                    last_val = 0.0
                    if self.bootstrap_truncated_paths:
                        next_obs_tensor = to_tensor(next_obs[None, ...], self.device, dtype=torch.float32)
                        last_val = float(self.method.value(next_obs_tensor).cpu().item())
                    assert buffer is not None
                    buffer.finish_path(last_val=last_val)
                    break

            batch = buffer.get(self.device) if buffer is not None else None
            stats = self.method.update(batch)
            wall_time = time.time() - start_time

            train_return_mean = float(np.mean(episode_returns)) if episode_returns else float("nan")
            train_return_std = float(np.std(episode_returns)) if episode_returns else float("nan")
            train_len_mean = float(np.mean(episode_lengths)) if episode_lengths else float("nan")
            record = {
                "iteration": epoch,
                "epoch": epoch,
                "method": self.method_name,
                "method_variant": self.method_variant,
                "estimator": self.estimator,
                "env_id": str(self.cfg.env.id),
                "suite": str(self.cfg.env.type),
                "seed": int(self.cfg.train.seed),
                "trainable": int(self.trainable),
                "did_update": float(stats.did_update),
                "env_steps": int(total_env_steps),
                "batch_env_steps": int(steps_in_epoch),
                "episodes_in_batch": len(episode_returns),
                "episodes_in_epoch": len(episode_returns),
                "train_return_mean": train_return_mean,
                "train_return_std": train_return_std,
                "train_len_mean": train_len_mean,
                # Backward-compatible aliases.
                "ep_return_mean": train_return_mean,
                "ep_return_std": train_return_std,
                "ep_len_mean": train_len_mean,
                **stats.to_log_dict(),
                "wall_time_sec": wall_time,
            }
            self.logger.log(record)
            print(record)
            episode_returns.clear()
            episode_lengths.clear()

            save_every = int(self.cfg.train.get("save_interval", 10))
            if self.supports_checkpoints and (epoch % save_every == 0 or epoch == total_epochs):
                self.save_checkpoint(epoch)

        self.logger.close()

    def save_checkpoint(self, epoch: int) -> None:
        if not self.supports_checkpoints:
            return
        ckpt = {
            "epoch": epoch,
            "method": self.method_name,
            "method_variant": self.method_variant,
            "state": self.method.state_dict(),
            "obs_rms_mean": None if self.obs_rms is None else self.obs_rms.mean,
            "obs_rms_var": None if self.obs_rms is None else self.obs_rms.var,
            "obs_rms_count": None if self.obs_rms is None else self.obs_rms.count,
        }
        torch.save(ckpt, self.checkpoint_dir / f"epoch_{epoch:04d}.pt")
