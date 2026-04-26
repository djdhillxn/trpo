from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch

from trpo_repro.methods import make_method
from trpo_repro.rollouts.worker import rollout_worker_loop


class ParallelRolloutCollector:
    """Synchronous multi-process rollout collection.

    This collector is intentionally Linux/Colab-focused and only activates when
    train.num_workers > 1. The default repository path remains the existing
    single-worker collector in Runner.
    """

    def __init__(self, env, cfg) -> None:
        if not sys.platform.startswith("linux"):
            raise RuntimeError(
                "Parallel rollout collection is only supported on Linux/Colab in this repo. "
                "Use train.num_workers=1 on macOS/local machines."
            )

        import multiprocessing as mp

        self.cfg = cfg
        self.num_workers = int(cfg.train.get("num_workers", 1))
        if self.num_workers <= 1:
            raise ValueError("ParallelRolloutCollector requires num_workers > 1.")
        if bool(cfg.train.get("normalize_obs", False)):
            raise ValueError("Parallel rollout collection does not support normalize_obs=true.")

        self.ctx = mp.get_context("fork")
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()

        self.target_steps = int(cfg.train.steps_per_epoch)
        self.max_ep_len = int(cfg.train.get("max_ep_len", 1000))
        self.obs_shape = tuple(env.observation_space.shape)
        if hasattr(env.action_space, "n"):
            self.act_shape = ()
            self.act_dtype = np.int64
        else:
            self.act_shape = tuple(env.action_space.shape)
            self.act_dtype = np.float32

        base = self.target_steps // self.num_workers
        remainder = self.target_steps % self.num_workers
        self.local_targets = [base + (1 if i < remainder else 0) for i in range(self.num_workers)]
        self.capacities = [target + self.max_ep_len for target in self.local_targets]
        self.starts = np.cumsum([0] + self.capacities[:-1]).tolist()
        self.total_capacity = int(sum(self.capacities))

        self.shared = self._allocate_shared_arrays()
        self.request_id = 0
        self.processes = []

        # Validate that the main repo's method abstraction can be instantiated on CPU.
        make_method(env.observation_space, env.action_space, cfg, device=torch.device("cpu"))

        cfg_dict = self._to_builtin(cfg)
        shared_meta = {
            "starts": self.starts,
            "capacities": self.capacities,
            "obs": self.shared["obs"],
            "act": self.shared["act"],
            "ret": self.shared["ret"],
            "weight": self.shared["weight"],
            "logp": self.shared["logp"],
        }
        for worker_id in range(self.num_workers):
            proc = self.ctx.Process(
                target=rollout_worker_loop,
                args=(worker_id, self.task_queue, self.result_queue, cfg_dict, shared_meta),
                daemon=True,
            )
            proc.start()
            self.processes.append(proc)

    @staticmethod
    def _to_builtin(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: ParallelRolloutCollector._to_builtin(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ParallelRolloutCollector._to_builtin(v) for v in value]
        return value

    def _raw_ndarray(self, shape: tuple[int, ...], dtype: np.dtype):
        c_type = np.ctypeslib.as_ctypes_type(dtype)
        raw = self.ctx.RawArray(c_type, int(np.prod(shape)))
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
        return raw, arr

    def _allocate_shared_arrays(self) -> dict[str, Any]:
        _, obs = self._raw_ndarray((self.total_capacity, *self.obs_shape), np.float32)
        act_shape = (self.total_capacity,) if len(self.act_shape) == 0 else (self.total_capacity, *self.act_shape)
        _, act = self._raw_ndarray(act_shape, np.dtype(self.act_dtype))
        _, ret = self._raw_ndarray((self.total_capacity,), np.float32)
        _, weight = self._raw_ndarray((self.total_capacity,), np.float32)
        _, logp = self._raw_ndarray((self.total_capacity,), np.float32)
        return {"obs": obs, "act": act, "ret": ret, "weight": weight, "logp": logp}

    @staticmethod
    def _state_to_cpu(state: Any) -> Any:
        if isinstance(state, torch.Tensor):
            return state.detach().cpu()
        if isinstance(state, dict):
            return {k: ParallelRolloutCollector._state_to_cpu(v) for k, v in state.items()}
        if isinstance(state, list):
            return [ParallelRolloutCollector._state_to_cpu(v) for v in state]
        if isinstance(state, tuple):
            return tuple(ParallelRolloutCollector._state_to_cpu(v) for v in state)
        return state

    def collect(self, method_state: dict[str, Any]) -> dict[str, Any]:
        self.request_id += 1
        request_id = self.request_id
        state_cpu = self._state_to_cpu(method_state)
        for worker_id, local_target in enumerate(self.local_targets):
            self.task_queue.put(
                {
                    "cmd": "collect",
                    "request_id": request_id,
                    "target_steps": int(local_target),
                    "method_state": state_cpu,
                    "worker_id": worker_id,
                }
            )

        results: list[dict[str, Any] | None] = [None] * self.num_workers
        received = 0
        while received < self.num_workers:
            result = self.result_queue.get()
            if int(result.get("request_id", -1)) != request_id:
                continue
            worker_id = int(result["worker_id"])
            results[worker_id] = result
            received += 1

        for result in results:
            if result is not None and result.get("error") is not None:
                raise RuntimeError(f"Parallel rollout worker failed: {result['error']}")

        obs_parts = []
        act_parts = []
        ret_parts = []
        weight_parts = []
        logp_parts = []
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        batch_steps = 0

        for worker_id, result in enumerate(results):
            assert result is not None
            count = int(result["count"])
            start = int(self.starts[worker_id])
            stop = start + count
            if count > 0:
                obs_parts.append(self.shared["obs"][start:stop].copy())
                act_parts.append(self.shared["act"][start:stop].copy())
                ret_parts.append(self.shared["ret"][start:stop].copy())
                weight_parts.append(self.shared["weight"][start:stop].copy())
                logp_parts.append(self.shared["logp"][start:stop].copy())
            batch_steps += int(result["steps_collected"])
            episode_returns.extend(float(x) for x in result["episode_returns"])
            episode_lengths.extend(int(x) for x in result["episode_lengths"])

        if not obs_parts:
            raise RuntimeError("Parallel collector returned an empty batch.")

        return {
            "obs": np.concatenate(obs_parts, axis=0),
            "act": np.concatenate(act_parts, axis=0),
            "ret": np.concatenate(ret_parts, axis=0),
            "weight": np.concatenate(weight_parts, axis=0),
            "adv": np.concatenate(weight_parts, axis=0),
            "logp": np.concatenate(logp_parts, axis=0),
            "batch_steps": int(batch_steps),
            "episodes_in_batch": len(episode_returns),
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
        }

    def close(self) -> None:
        for _ in self.processes:
            self.task_queue.put({"cmd": "close"})
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
        self.processes.clear()
