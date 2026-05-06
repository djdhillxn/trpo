import csv
import time
from pathlib import Path

import numpy as np
import torch

from trpo_repro.config import save_config
from trpo_repro.data.buffer import RolloutBatch, TrajectoryBuffer
from trpo_repro.methods import make_method, resolve_method_name
from trpo_repro.rollouts import ParallelRolloutCollector
from trpo_repro.utils.run_metadata import (
    build_base_run_metadata,
    build_failure_record,
    collect_environment_metadata,
    collect_git_metadata,
    monotonic_time,
    utc_now_iso,
    write_failure,
    write_git_diff_snapshot,
    write_run_summary,
)
from trpo_repro.utils.utils import JsonlLogger, ensure_dir, get_tqdm, write_json
from trpo_repro.utils.torch_utils import RunningMeanStd, to_tensor


class Runner:
    def __init__(self, env, cfg, output_dir: str | Path, device: str = "cpu", launch_metadata: dict | None = None) -> None:
        self.env = env
        self.cfg = cfg
        self.output_dir = ensure_dir(output_dir)
        self.device = torch.device(device)
        self.run_start_monotonic = monotonic_time()
        self.logger = JsonlLogger(self.output_dir, mode="w")
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.buffer_dir = ensure_dir(self.output_dir / "_buffers")
        self.completed_epochs = 0
        self.current_epoch: int | None = None
        self.total_env_steps = 0
        self.latest_record: dict | None = None
        self.best_train_return_record: dict | None = None
        self.checkpoint_paths: list[str] = []
        self.resume_metadata: dict | None = None

        self.method_name = resolve_method_name(cfg)
        self.method = make_method(env.observation_space, env.action_space, cfg, self.device)
        self.method_variant = getattr(self.method, "variant", "default")

        self.estimator = getattr(self.method, "estimator", None)
        self.normalize_weights = bool(cfg.algo.get("normalize_weights", self.estimator != "trpo_paper"))
        self.bootstrap_truncated_paths = bool(cfg.algo.get("bootstrap_truncated_paths", self.estimator != "trpo_paper"))
        self.trainable = bool(getattr(self.method, "trainable", False))
        self.supports_checkpoints = bool(getattr(self.method, "supports_checkpoints", False))
        self.batch_obs_to_device = bool(getattr(self.method, "batch_obs_to_device", True))

        self.normalize_obs = bool(cfg.train.get("normalize_obs", False)) and len(env.observation_space.shape) == 1
        self.obs_rms = RunningMeanStd(shape=tuple(env.observation_space.shape)) if self.normalize_obs else None

        self.memory_mode = str(cfg.train.get("memory_mode", "standard")).lower()
        if self.memory_mode not in {"standard", "safe"}:
            raise ValueError(f"Unsupported memory_mode: {self.memory_mode}")
        configured_obs_storage = cfg.train.get("obs_storage", "auto")
        self.obs_storage = str(configured_obs_storage).lower()
        if self.obs_storage == "auto":
            self.obs_storage = "memmap" if self.memory_mode == "safe" else "ram"
        configured_chunk = cfg.algo.get("full_batch_chunk_size")
        if configured_chunk is None:
            self.full_batch_chunk_size = 2048 if self.memory_mode == "safe" and self.trainable and self.method_name != "ppo" else None
        else:
            configured_chunk = int(configured_chunk)
            self.full_batch_chunk_size = configured_chunk if configured_chunk > 0 else None
        configured_fvp_fraction = cfg.algo.get("fvp_subsample_fraction")
        if configured_fvp_fraction is None:
            self.fvp_subsample_fraction = None
        else:
            configured_fvp_fraction = float(configured_fvp_fraction)
            if configured_fvp_fraction <= 0.0:
                raise ValueError("fvp_subsample_fraction must be positive when provided.")
            if configured_fvp_fraction > 1.0:
                if configured_fvp_fraction <= 100.0:
                    configured_fvp_fraction = configured_fvp_fraction / 100.0
                else:
                    raise ValueError("fvp_subsample_fraction must be in (0, 1] or a percentage in (0, 100].")
            self.fvp_subsample_fraction = configured_fvp_fraction
        if self.memory_mode == "safe" and self.method_name in {"empirical_fim", "trpo_empirical_fim"}:
            raise ValueError("Empirical-FIM TRPO currently supports only memory_mode=standard.")
        self.cfg.train.obs_storage = self.obs_storage
        self.cfg.algo.full_batch_chunk_size = self.full_batch_chunk_size
        self.cfg.algo.fvp_subsample_fraction = self.fvp_subsample_fraction

        tqdm_impl, resolved_progress_mode = get_tqdm(str(cfg.train.get("progress_mode", "auto")))
        self.tqdm = tqdm_impl
        self.progress_mode = resolved_progress_mode

        self.num_workers = max(1, int(cfg.train.get("num_workers", 1)))
        self.parallel_rollouts = self.num_workers > 1
        if self.parallel_rollouts and self.normalize_obs:
            raise ValueError("Parallel rollout collection does not support normalize_obs=true in this repo.")
        if self.parallel_rollouts and self.obs_storage == "memmap":
            raise ValueError(
                "Parallel rollout collection currently supports obs_storage='ram' only. "
                "Use --obs_storage ram when enabling --num-workers."
            )
        self.collector = ParallelRolloutCollector(env, cfg) if self.parallel_rollouts else None

        if self.memory_mode == "safe" and self.trainable and self.method_name != "ppo":
            # Keep observations on CPU-backed storage in safe mode and stream them to the device in chunks.
            self.batch_obs_to_device = False

        suite = str(cfg.env.get("type", "unknown")).lower()

        self.fvp_estimator = str(cfg.algo.get("fvp_estimator", "analytic")).lower()
        save_config(self.cfg, self.output_dir / "config_runtime.yaml")

        environment_metadata = collect_environment_metadata(device=str(self.device))
        write_json(environment_metadata, self.output_dir / "environment.json")
        git_metadata = collect_git_metadata(Path(__file__).resolve())
        git_diff_path = write_git_diff_snapshot(self.output_dir, Path(__file__).resolve())
        if git_diff_path is not None:
            git_metadata["diff_path"] = git_diff_path

        runtime_metadata = {
            "device": str(self.device),
            "normalize_obs": self.normalize_obs,
            "memory_mode": self.memory_mode,
            "obs_storage": self.obs_storage,
            "full_batch_chunk_size": self.full_batch_chunk_size,
            "fvp_subsample_fraction": self.fvp_subsample_fraction,
            "fvp_estimator": self.fvp_estimator,
            "progress_mode": self.progress_mode,
            "num_workers": self.num_workers,
            "parallel_rollouts": self.parallel_rollouts,
            "target_epochs": int(cfg.train.epochs),
            "steps_per_epoch": int(cfg.train.steps_per_epoch),
            "max_ep_len": int(cfg.train.get("max_ep_len", 1000)),
            "save_interval": int(cfg.train.get("save_interval", 10)),
            "observation_space": repr(env.observation_space),
            "action_space": repr(env.action_space),
        }
        self.run_metadata = build_base_run_metadata(
            launch=launch_metadata or {},
            environment=environment_metadata,
            git=git_metadata,
            method=self.method_name,
            method_variant=self.method_variant,
            estimator=self.estimator,
            env_id=str(cfg.env.id),
            suite=suite,
            seed=int(cfg.train.seed),
            run_name=str(cfg.train.get("run_name", self.output_dir.name)),
            trainable=self.trainable,
            runtime=runtime_metadata,
        )
        self.git_commit_hash = self.run_metadata.get("git_commit_hash")
        self.run_metadata.update(runtime_metadata)
        write_json(self.run_metadata, self.output_dir / "run_metadata.json")
        print({
            "runner_method": self.method_name,
            "runner_variant": self.method_variant,
            "runner_estimator": self.estimator,
            "trainable": self.trainable,
            "supports_checkpoints": self.supports_checkpoints,
            "normalize_obs": self.normalize_obs,
            "memory_mode": self.memory_mode,
            "obs_storage": self.obs_storage,
            "full_batch_chunk_size": self.full_batch_chunk_size,
            "fvp_subsample_fraction": self.fvp_subsample_fraction,
            "fvp_estimator": self.fvp_estimator,
            "progress_mode": self.progress_mode,
            "num_workers": self.num_workers,
            "parallel_rollouts": self.parallel_rollouts,
            "git_commit_hash": self.git_commit_hash,
        })

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        if self.obs_rms is not None:
            self.obs_rms.update(obs[None, ...])
            obs = self.obs_rms.normalize(obs)
        return obs.astype(np.float32)

    def _make_buffer(self, target_steps: int, max_ep_len: int, epoch: int):
        obs_shape = tuple(self.env.observation_space.shape)
        if hasattr(self.env.action_space, "n"):
            act_shape = ()
            act_dtype = np.int64
        else:
            act_shape = tuple(self.env.action_space.shape)
            act_dtype = np.float32
        buffer_size = target_steps + (max_ep_len if self.estimator == "trpo_paper" else 0)
        return TrajectoryBuffer(
            obs_shape=obs_shape,
            act_shape=act_shape,
            size=buffer_size,
            gamma=float(self.cfg.algo.gamma),
            lam=float(self.cfg.algo.get("lam", 1.0)),
            estimator=self.estimator,
            act_dtype=act_dtype,
            normalize_weights=self.normalize_weights,
            obs_storage=self.obs_storage,
            storage_dir=self.buffer_dir,
            storage_prefix=f"epoch_{epoch:04d}",
        )

    def _batch_from_arrays(self, arrays: dict[str, np.ndarray]) -> RolloutBatch:
        weight = np.asarray(arrays["weight"], dtype=np.float32)
        if self.normalize_weights and len(weight) > 1:
            weight = (weight - float(weight.mean())) / (float(weight.std()) + 1e-8)
        if self.batch_obs_to_device:
            obs_obj = torch.as_tensor(arrays["obs"], dtype=torch.float32, device=self.device)
        else:
            obs_obj = np.asarray(arrays["obs"], dtype=np.float32)
        weights_t = torch.as_tensor(weight, dtype=torch.float32, device=self.device)
        return RolloutBatch(
            {
                "obs": obs_obj,
                "obs_storage": "ram",
                "size": len(weight),
                "act": torch.as_tensor(arrays["act"], device=self.device),
                "ret": torch.as_tensor(arrays["ret"], dtype=torch.float32, device=self.device),
                "weight": weights_t,
                "adv": weights_t,
                "logp": torch.as_tensor(arrays["logp"], dtype=torch.float32, device=self.device),
            }
        )

    def _validate_epoch_stats(self, stats) -> None:
        if self.method_name == "random":
            assert not self.trainable, "Random method should not be trainable."
            assert float(stats.did_update) == 0.0, "Random method should never report an update."
        elif self.method_name in {"natural_pg", "npg"}:
            assert np.isnan(stats.line_search_success), "Natural PG should not report line search success."
        elif self.method_name in {"trpo", "trpo_max_kl", "empirical_fim", "trpo_empirical_fim"}:
            assert float(stats.line_search_success) in {0.0, 1.0}, "TRPO methods should report line search success."
        elif self.method_name == "ppo":
            assert np.isnan(stats.line_search_success), "PPO should not report line search success."

    def _write_epoch_record(self, epoch_iter, record: dict, total_env_steps: int, train_return_mean: float) -> None:
        if self.progress_mode in {"terminal", "notebook"}:
            epoch_iter.set_postfix(
                method=self.method_name,
                ret=f"{train_return_mean:.2f}" if np.isfinite(train_return_mean) else "nan",
                kl=(
                    f"{record.get('approx_kl', float('nan')):.4f}"
                    if np.isfinite(record.get("approx_kl", float("nan")))
                    else "nan"
                ),
                steps=total_env_steps,
            )
        if self.progress_mode == "terminal":
            self.tqdm.write(str(record))
        else:
            print(record)

    @staticmethod
    def _load_checkpoint_file(path: Path, device: torch.device) -> dict:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint {path} did not contain a dictionary payload.")
        return checkpoint

    @staticmethod
    def _float_from_row(row: dict[str, str], key: str) -> float | None:
        value = row.get(key)
        if value in {None, "", "nan", "NaN"}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _read_source_metrics(cls, checkpoint_path: Path, resume_epoch: int) -> tuple[dict[str, str] | None, dict | None]:
        metrics_path = checkpoint_path.parent.parent / "metrics.csv"
        if not metrics_path.exists():
            return None, None
        checkpoint_row = None
        best_record = None
        try:
            with metrics_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_epoch = row.get("epoch") or row.get("iteration")
                    if raw_epoch in {None, ""}:
                        continue
                    try:
                        epoch = int(float(raw_epoch))
                    except ValueError:
                        continue
                    if epoch > resume_epoch:
                        continue
                    train_return = cls._float_from_row(row, "train_return_mean")
                    env_steps = cls._float_from_row(row, "env_steps")
                    train_return_std = cls._float_from_row(row, "train_return_std")
                    train_len_mean = cls._float_from_row(row, "train_len_mean")
                    if train_return is not None and np.isfinite(train_return):
                        if best_record is None or train_return > float(best_record["train_return_mean"]):
                            best_record = {
                                "epoch": epoch,
                                "env_steps": 0 if env_steps is None else int(env_steps),
                                "train_return_mean": train_return,
                                "train_return_std": float("nan") if train_return_std is None else train_return_std,
                                "train_len_mean": float("nan") if train_len_mean is None else train_len_mean,
                            }
                    if epoch == resume_epoch:
                        checkpoint_row = row
        except OSError:
            return None, None
        return checkpoint_row, best_record

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        if not self.supports_checkpoints:
            raise ValueError(f"Method {self.method_name!r} does not support checkpoint resume.")

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = self._load_checkpoint_file(checkpoint_path, self.device)
        state = checkpoint.get("state")
        if state is None and "policy" in checkpoint:
            state = {"policy": checkpoint.get("policy"), "value_fn": checkpoint.get("value_fn")}
        if state is None:
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain a method state.")
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint {checkpoint_path} has an unsupported method state payload.")

        checkpoint_method = checkpoint.get("method_name") or state.get("method_name")
        if checkpoint_method is not None and checkpoint_method != self.method.name:
            raise ValueError(
                f"Checkpoint method {checkpoint_method!r} does not match configured method {self.method.name!r}."
            )
        checkpoint_variant = checkpoint.get("method_variant") or state.get("method_variant")
        if checkpoint_variant is not None and checkpoint_variant != self.method.variant:
            raise ValueError(
                f"Checkpoint variant {checkpoint_variant!r} does not match configured variant {self.method.variant!r}."
            )

        self.method.load_state_dict(state)
        if self.obs_rms is not None:
            mean = checkpoint.get("obs_rms_mean")
            var = checkpoint.get("obs_rms_var")
            count = checkpoint.get("obs_rms_count")
            if mean is not None and var is not None and count is not None:
                self.obs_rms.mean = np.asarray(mean, dtype=np.float64)
                self.obs_rms.var = np.asarray(var, dtype=np.float64)
                self.obs_rms.count = float(count)

        resume_epoch = int(checkpoint.get("epoch", 0))
        source_row, best_record = self._read_source_metrics(checkpoint_path, resume_epoch)
        checkpoint_steps = checkpoint.get("total_env_steps")
        if checkpoint_steps is None and source_row is not None:
            checkpoint_steps = self._float_from_row(source_row, "env_steps")

        self.completed_epochs = resume_epoch
        self.current_epoch = resume_epoch
        self.total_env_steps = 0 if checkpoint_steps is None else int(checkpoint_steps)
        if best_record is not None:
            self.best_train_return_record = best_record

        self.resume_metadata = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_epoch": resume_epoch,
            "checkpoint_total_env_steps": self.total_env_steps,
            "checkpoint_method": checkpoint_method,
            "checkpoint_variant": checkpoint_variant,
        }
        self.run_metadata["resume"] = self.resume_metadata
        write_json(self.run_metadata, self.output_dir / "run_metadata.json")
        print(
            {
                "resumed_from": str(checkpoint_path),
                "resume_epoch": resume_epoch,
                "resume_total_env_steps": self.total_env_steps,
            }
        )

    def train(self) -> None:
        run_status = "completed"
        failure_record = None
        try:
            total_epochs = int(self.cfg.train.epochs)
            target_steps = int(self.cfg.train.steps_per_epoch)
            max_ep_len = int(self.cfg.train.get("max_ep_len", 1000))
            start_epoch = self.completed_epochs + 1
            if start_epoch > total_epochs:
                raise ValueError(
                    f"Resume checkpoint is already at epoch {self.completed_epochs}, "
                    f"but train.epochs is {total_epochs}. Set --epochs to the final target epoch."
                )

            obs, _ = self.env.reset()
            obs = self._preprocess_obs(obs)
            ep_ret = 0.0
            ep_len = 0
            episode_returns: list[float] = []
            episode_lengths: list[float] = []
            total_env_steps = int(self.total_env_steps)

            disable_progress = self.progress_mode == "off"
            epoch_iter = self.tqdm(
                range(start_epoch, total_epochs + 1),
                desc="Training",
                position=0,
                leave=True,
                disable=disable_progress,
            )
            save_interval = int(self.cfg.train.get("save_interval", 10))

            for epoch in epoch_iter:
                self.current_epoch = epoch
                epoch_start_time = time.time()
                self.method.set_training_progress(epoch, total_epochs)
                steps_in_epoch = 0
                collect_start_time = time.time()

                if self.parallel_rollouts:
                    assert self.collector is not None
                    arrays = self.collector.collect(self.method.state_dict())
                    batch = self._batch_from_arrays(arrays) if self.trainable else None
                    steps_in_epoch = int(arrays["batch_steps"])
                    total_env_steps += steps_in_epoch
                    self.total_env_steps = total_env_steps
                    episode_returns = list(arrays["episode_returns"])
                    episode_lengths = list(arrays["episode_lengths"])
                else:
                    buffer = self._make_buffer(target_steps, max_ep_len, epoch) if self.trainable else None
                    rollout_pbar = None
                    if self.progress_mode == "terminal":
                        rollout_pbar = self.tqdm(
                            total=target_steps,
                            desc=f"Epoch {epoch}/{total_epochs}",
                            position=1,
                            leave=False,
                            dynamic_ncols=True,
                            mininterval=0.5,
                        )

                    while True:
                        obs_tensor = to_tensor(obs[None, ...], self.device, dtype=torch.float32)
                        action_t, value_t, logp_t = self.method.act(obs_tensor, deterministic=False)
                        action = action_t.squeeze(0).cpu().numpy()
                        env_action = int(action) if hasattr(self.env.action_space, "n") else action
                        value = float(value_t.squeeze(0).cpu().item()) if value_t.ndim > 0 else float(value_t.cpu().item())
                        logp = float(logp_t.squeeze(0).cpu().item()) if logp_t.ndim > 0 else float(logp_t.cpu().item())

                        next_obs, reward, terminated, truncated, _ = self.env.step(env_action)
                        next_obs = self._preprocess_obs(next_obs)

                        if buffer is not None:
                            buffer.store(obs, action, float(reward), value, logp)
                        ep_ret += float(reward)
                        ep_len += 1
                        steps_in_epoch += 1
                        total_env_steps += 1
                        self.total_env_steps = total_env_steps

                        if rollout_pbar is not None and steps_in_epoch <= target_steps:
                            rollout_pbar.update(1)
                            rollout_pbar.set_postfix(ep_len=ep_len, ep_ret=f"{ep_ret:.2f}", env_steps=total_env_steps)

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
                            if not self.trainable or self.estimator == "trpo_paper":
                                continue
                            last_val = 0.0
                            if self.bootstrap_truncated_paths:
                                next_obs_tensor = to_tensor(next_obs[None, ...], self.device, dtype=torch.float32)
                                last_val = float(self.method.value(next_obs_tensor).cpu().item())
                            assert buffer is not None
                            buffer.finish_path(last_val=last_val)
                            break

                    if rollout_pbar is not None:
                        rollout_pbar.close()
                    batch = buffer.get(self.device, obs_to_device=self.batch_obs_to_device) if buffer is not None else None
                    del buffer

                collect_time = time.time() - collect_start_time
                update_start_time = time.time()
                stats = None
                try:
                    stats = self.method.update(batch)
                finally:
                    if batch is not None and hasattr(batch, "cleanup"):
                        batch.cleanup()
                assert stats is not None
                self._validate_epoch_stats(stats)
                update_time = time.time() - update_start_time

                checkpoint_time = 0.0
                if self.supports_checkpoints and epoch % save_interval == 0:
                    checkpoint_start_time = time.time()
                    self.save_checkpoint(epoch)
                    checkpoint_time = time.time() - checkpoint_start_time

                wall_time = time.time() - epoch_start_time
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
                    "num_workers": self.num_workers,
                    "parallel_rollouts": int(self.parallel_rollouts),
                    "train_return_mean": train_return_mean,
                    "train_return_std": train_return_std,
                    "train_len_mean": train_len_mean,
                    "ep_return_mean": train_return_mean,
                    "ep_return_std": train_return_std,
                    "ep_len_mean": train_len_mean,
                    **stats.to_log_dict(),
                    "collect_time_sec": collect_time,
                    "update_time_sec": update_time,
                    "checkpoint_time_sec": checkpoint_time,
                    "wall_time_sec": wall_time,
                }
                self.logger.log(record)
                self._write_epoch_record(epoch_iter, record, total_env_steps, train_return_mean)
                self.completed_epochs = epoch
                self.total_env_steps = total_env_steps
                self.latest_record = record
                if np.isfinite(train_return_mean):
                    if (
                        self.best_train_return_record is None
                        or train_return_mean > float(self.best_train_return_record["train_return_mean"])
                    ):
                        self.best_train_return_record = {
                            "epoch": epoch,
                            "env_steps": int(total_env_steps),
                            "train_return_mean": train_return_mean,
                            "train_return_std": train_return_std,
                            "train_len_mean": train_len_mean,
                        }
                episode_returns.clear()
                episode_lengths.clear()
        except BaseException as exc:
            run_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            failure_record = build_failure_record(exc, epoch=self.current_epoch)
            try:
                write_failure(self.output_dir, failure_record)
            except Exception as write_exc:
                print(f"Warning: failed to write failure metadata: {write_exc!r}")
            raise
        finally:
            self._finalize_run(run_status, failure_record)
            if self.collector is not None:
                self.collector.close()
            self.logger.close()
            self.env.close()

    def _finalize_run(self, status: str, failure_record: dict | None = None) -> None:
        ended_at = utc_now_iso()
        duration_sec = monotonic_time() - self.run_start_monotonic
        summary = {
            "schema_version": 1,
            "run_id": self.run_metadata.get("run_id"),
            "status": status,
            "started_at": self.run_metadata.get("started_at"),
            "ended_at": ended_at,
            "duration_sec": duration_sec,
            "completed_epochs": self.completed_epochs,
            "target_epochs": int(self.cfg.train.epochs),
            "total_env_steps": self.total_env_steps,
            "latest_epoch_record": self.latest_record,
            "best_train_return": self.best_train_return_record,
            "checkpoints": self.checkpoint_paths,
        }
        if self.resume_metadata is not None:
            summary["resume"] = self.resume_metadata
        if failure_record is not None:
            summary["failure"] = {
                "exception_type": failure_record["exception_type"],
                "exception_message": failure_record["exception_message"],
                "epoch": failure_record["epoch"],
                "failure_path": str((self.output_dir / "failure.json").resolve()),
            }
        try:
            summary_path = write_run_summary(self.output_dir, summary)
            self.run_metadata.update(
                {
                    "status": status,
                    "ended_at": ended_at,
                    "duration_sec": duration_sec,
                    "summary_path": str(summary_path.resolve()),
                    "completed_epochs": self.completed_epochs,
                    "total_env_steps": self.total_env_steps,
                }
            )
            if failure_record is not None:
                self.run_metadata["failure"] = summary["failure"]
            write_json(self.run_metadata, self.output_dir / "run_metadata.json")
        except Exception as exc:
            print(f"Warning: failed to finalize run metadata: {exc!r}")

    def save_checkpoint(self, epoch: int) -> None:
        if not self.supports_checkpoints:
            return
        ckpt = {
            "epoch": epoch,
            "total_env_steps": self.total_env_steps,
            "state": self.method.state_dict(),
            "method_name": self.method.name,
            "method_variant": self.method.variant,
            "obs_rms_mean": None if self.obs_rms is None else self.obs_rms.mean,
            "obs_rms_var": None if self.obs_rms is None else self.obs_rms.var,
            "obs_rms_count": None if self.obs_rms is None else self.obs_rms.count,
        }
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(ckpt, checkpoint_path)
        self.checkpoint_paths.append(str(checkpoint_path.resolve()))
