
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from trpo_repro.algos.conjugate_gradient import conjugate_gradient
from trpo_repro.algos.line_search import backtracking_line_search
from trpo_repro.models.policies import max_kl, mean_kl
from trpo_repro.utils.torch_utils import flat_grad, flat_params, set_flat_params, to_tensor


@dataclass
class TRPOStats:
    policy_loss_before: float
    policy_loss_after: float
    value_loss_before: float
    value_loss_after: float
    entropy: float
    approx_kl: float
    line_search_success: bool
    cg_norm: float


class TRPOAgent:
    def __init__(
        self,
        policy: nn.Module,
        value_fn: nn.Module | None,
        cfg,
        device: torch.device,
        *,
        update_mode: str = "trpo",
        kl_constraint_metric: str = "average",
        fvp_kl_metric: str = "average",
    ) -> None:
        self.policy = policy.to(device)
        self.value_fn = None if value_fn is None else value_fn.to(device)
        self.cfg = cfg
        self.device = device
        self.update_mode = update_mode.lower()
        self.kl_constraint_metric = kl_constraint_metric.lower()
        self.fvp_kl_metric = fvp_kl_metric.lower()
        self.vf_optimizer = None
        if self.value_fn is not None:
            self.vf_optimizer = torch.optim.Adam(
                self.value_fn.parameters(),
                lr=float(cfg.algo.get("vf_lr", 1e-3)),
                weight_decay=float(cfg.algo.get("vf_weight_decay", 0.0)),
            )

    @torch.no_grad()
    def step(self, obs: torch.Tensor, deterministic: bool = False):
        action, logp, _ = self.policy.act(obs, deterministic=deterministic)
        if self.value_fn is None:
            value = torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)
        else:
            value = self.value_fn(obs)
        return action, value, logp

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        if self.value_fn is None:
            return torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)
        return self.value_fn(obs)

    def update(self, batch) -> TRPOStats:
        obs = batch["obs"]
        act = batch["act"]
        ret = batch["ret"]
        weight = batch.get("weight", batch["adv"])
        old_logp = batch["logp"]
        chunk_size = self._chunk_size_for(batch)

        if self.value_fn is not None and chunk_size is not None:
            raise ValueError("Chunked full-batch updates are only supported for paper_mc (no value function).")

        if chunk_size is None:
            return self._update_standard(obs, act, ret, weight, old_logp)
        return self._update_chunked(obs, act, weight, old_logp, chunk_size)

    def _chunk_size_for(self, batch) -> int | None:
        configured = self.cfg.algo.get("full_batch_chunk_size")
        if configured is None:
            return None
        configured = int(configured)
        return configured if configured > 0 else None

    def _update_standard(self, obs, act, ret, weight, old_logp) -> TRPOStats:
        with torch.no_grad():
            old_dist = self.policy.distribution(obs)
            old_surr = self._surrogate(obs, act, weight, old_logp).item()
            entropy = old_dist.entropy().mean().item()
            value_loss_before = float("nan") if self.value_fn is None else self._value_loss(obs, ret).item()
        if self.value_fn is None:
            value_loss_after = float("nan")
        else:
            self._update_value_function(obs, ret)
            value_loss_after = self._value_loss(obs, ret).item()
        surr = self._surrogate(obs, act, weight, old_logp)
        grads = torch.autograd.grad(surr, self.policy.parameters())
        g = flat_grad(grads).detach()

        def fvp(v: torch.Tensor) -> torch.Tensor:
            new_dist = self.policy.distribution(obs)
            kl = self._kl_metric(old_dist, new_dist, metric=self.fvp_kl_metric)
            grad_kl = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = flat_grad(grad_kl)
            kl_v = (flat_grad_kl * v).sum()
            hvp = torch.autograd.grad(kl_v, self.policy.parameters())
            damping = float(self.cfg.algo.get("cg_damping", 1e-1))
            return flat_grad(hvp).detach() + damping * v

        step_dir = conjugate_gradient(
            fvp,
            g,
            nsteps=int(self.cfg.algo.get("cg_iters", 10)),
            residual_tol=float(self.cfg.algo.get("cg_residual_tol", 1e-10)),
        )
        if self.update_mode == "trpo":
            new_params, success, approx_kl = self._apply_trpo_step(obs, act, weight, old_logp, old_dist, old_surr, g, step_dir)
        elif self.update_mode == "npg":
            new_params, success, approx_kl = self._apply_npg_step(obs, old_dist, step_dir)
        else:
            raise ValueError(f"Unsupported update mode: {self.update_mode}")
        set_flat_params(self.policy, new_params)
        with torch.no_grad():
            policy_loss_after = self._surrogate(obs, act, weight, old_logp).item()
        return TRPOStats(old_surr, policy_loss_after, value_loss_before, value_loss_after, entropy, approx_kl, success, step_dir.norm().item())

    def _update_chunked(self, obs, act, weight, old_logp, chunk_size: int) -> TRPOStats:
        old_policy = copy.deepcopy(self.policy).to(self.device)
        old_policy.eval()
        for param in old_policy.parameters():
            param.requires_grad_(False)

        old_surr = self._surrogate_chunked(old_policy, obs, act, weight, old_logp, chunk_size)
        entropy = self._entropy_chunked(old_policy, obs, chunk_size)
        g = self._policy_grad_chunked(obs, act, weight, old_logp, chunk_size)

        def fvp(v: torch.Tensor) -> torch.Tensor:
            return self._fvp_chunked(old_policy, obs, v, chunk_size)

        step_dir = conjugate_gradient(
            fvp,
            g,
            nsteps=int(self.cfg.algo.get("cg_iters", 10)),
            residual_tol=float(self.cfg.algo.get("cg_residual_tol", 1e-10)),
        )

        if self.update_mode == "trpo":
            new_params, success, approx_kl = self._apply_trpo_step_chunked(obs, act, weight, old_logp, old_policy, old_surr, g, step_dir, chunk_size)
        elif self.update_mode == "npg":
            new_params, success, approx_kl = self._apply_npg_step_chunked(obs, old_policy, step_dir, chunk_size)
        else:
            raise ValueError(f"Unsupported update mode: {self.update_mode}")
        set_flat_params(self.policy, new_params)
        policy_loss_after = self._surrogate_chunked(self.policy, obs, act, weight, old_logp, chunk_size)
        return TRPOStats(old_surr, policy_loss_after, float("nan"), float("nan"), entropy, approx_kl, success, step_dir.norm().item())

    def _apply_trpo_step(self, obs, act, weight, old_logp, old_dist, old_surr, gradient, step_dir):
        shs = 0.5 * (step_dir * self._fvp_on_batch(obs, old_dist, step_dir)).sum()
        scale = torch.sqrt(torch.tensor(float(self.cfg.algo.max_kl), device=self.device) / (shs + 1e-8))
        full_step = step_dir * scale
        expected_improve = (gradient * full_step).sum().item()
        old_params = flat_params(self.policy)

        def evaluate(candidate_params: torch.Tensor) -> tuple[float, float]:
            set_flat_params(self.policy, candidate_params)
            with torch.no_grad():
                new_dist = self.policy.distribution(obs)
                new_surr = self._surrogate(obs, act, weight, old_logp).item()
                kl_val = self._kl_metric(old_dist, new_dist, metric=self.kl_constraint_metric).item()
            return new_surr - old_surr, kl_val

        new_params, success, _, approx_kl = backtracking_line_search(
            old_params,
            full_step,
            evaluate,
            float(self.cfg.algo.get("backtrack_coeff", 0.8)),
            int(self.cfg.algo.get("backtrack_iters", 10)),
            expected_improve,
            float(self.cfg.algo.max_kl),
        )
        if not success:
            new_params = old_params
        return new_params, success, approx_kl

    def _apply_trpo_step_chunked(self, obs, act, weight, old_logp, old_policy, old_surr, gradient, step_dir, chunk_size):
        shs = 0.5 * (step_dir * self._fvp_chunked(old_policy, obs, step_dir, chunk_size)).sum()
        scale = torch.sqrt(torch.tensor(float(self.cfg.algo.max_kl), device=self.device) / (shs + 1e-8))
        full_step = step_dir * scale
        expected_improve = (gradient * full_step).sum().item()
        old_params = flat_params(self.policy)

        def evaluate(candidate_params: torch.Tensor) -> tuple[float, float]:
            set_flat_params(self.policy, candidate_params)
            new_surr = self._surrogate_chunked(self.policy, obs, act, weight, old_logp, chunk_size)
            kl_val = self._kl_chunked(old_policy, self.policy, obs, self.kl_constraint_metric, chunk_size)
            return new_surr - old_surr, kl_val

        new_params, success, _, approx_kl = backtracking_line_search(
            old_params,
            full_step,
            evaluate,
            float(self.cfg.algo.get("backtrack_coeff", 0.8)),
            int(self.cfg.algo.get("backtrack_iters", 10)),
            expected_improve,
            float(self.cfg.algo.max_kl),
        )
        if not success:
            new_params = old_params
        return new_params, success, approx_kl

    def _apply_npg_step(self, obs, old_dist, step_dir):
        old_params = flat_params(self.policy)
        step_scale = float(self.cfg.algo.get("npg_stepsize", 0.05))
        new_params = old_params + step_scale * step_dir
        set_flat_params(self.policy, new_params)
        with torch.no_grad():
            new_dist = self.policy.distribution(obs)
            approx_kl = self._kl_metric(old_dist, new_dist, metric=self.kl_constraint_metric).item()
        return new_params, True, approx_kl

    def _apply_npg_step_chunked(self, obs, old_policy, step_dir, chunk_size):
        old_params = flat_params(self.policy)
        step_scale = float(self.cfg.algo.get("npg_stepsize", 0.05))
        new_params = old_params + step_scale * step_dir
        set_flat_params(self.policy, new_params)
        approx_kl = self._kl_chunked(old_policy, self.policy, obs, self.kl_constraint_metric, chunk_size)
        return new_params, True, approx_kl

    def _chunk_obs_to_device(self, obs_chunk):
        if isinstance(obs_chunk, torch.Tensor):
            return obs_chunk.to(self.device, dtype=torch.float32)
        return to_tensor(obs_chunk, self.device, dtype=torch.float32)

    def _iter_chunks(self, obs, *others, chunk_size: int):
        n = len(obs)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_obs = obs[start:end]
            chunk_other = [tensor[start:end] for tensor in others]
            yield end - start, self._chunk_obs_to_device(chunk_obs), chunk_other

    def _surrogate_chunked(self, policy_model, obs, act, weight, old_logp, chunk_size: int) -> float:
        total = 0.0
        n = len(weight)
        with torch.no_grad():
            for _, obs_chunk, (act_chunk, weight_chunk, old_logp_chunk) in self._iter_chunks(obs, act, weight, old_logp, chunk_size=chunk_size):
                dist = policy_model.distribution(obs_chunk)
                logp = policy_model.log_prob_from_dist(dist, act_chunk)
                total += (torch.exp(logp - old_logp_chunk) * weight_chunk).sum().item()
        return total / float(n)

    def _entropy_chunked(self, policy_model, obs, chunk_size: int) -> float:
        total = 0.0
        n = len(obs)
        with torch.no_grad():
            for _, obs_chunk, _ in self._iter_chunks(obs, chunk_size=chunk_size):
                total += policy_model.distribution(obs_chunk).entropy().sum().item()
        return total / float(n)

    def _policy_grad_chunked(self, obs, act, weight, old_logp, chunk_size: int) -> torch.Tensor:
        total_n = float(len(weight))
        total_grad = None
        for _, obs_chunk, (act_chunk, weight_chunk, old_logp_chunk) in self._iter_chunks(obs, act, weight, old_logp, chunk_size=chunk_size):
            dist = self.policy.distribution(obs_chunk)
            logp = self.policy.log_prob_from_dist(dist, act_chunk)
            objective = (torch.exp(logp - old_logp_chunk) * weight_chunk).sum() / total_n
            grads = torch.autograd.grad(objective, self.policy.parameters())
            flat = flat_grad(grads).detach()
            total_grad = flat if total_grad is None else total_grad + flat
        return total_grad if total_grad is not None else torch.zeros_like(flat_params(self.policy))

    def _kl_chunked(self, old_policy, new_policy, obs, metric: str, chunk_size: int) -> float:
        metric = metric.lower()
        total_n = float(len(obs))
        total = 0.0
        max_val = 0.0
        with torch.no_grad():
            for chunk_len, obs_chunk, _ in self._iter_chunks(obs, chunk_size=chunk_size):
                old_dist = old_policy.distribution(obs_chunk)
                new_dist = new_policy.distribution(obs_chunk)
                if metric == "average":
                    total += self._kl_metric(old_dist, new_dist, metric="average").item() * chunk_len
                elif metric == "max":
                    max_val = max(max_val, self._kl_metric(old_dist, new_dist, metric="max").item())
                else:
                    raise ValueError(f"Unsupported KL metric: {metric}")
        return total / total_n if metric == "average" else max_val

    def _fvp_chunked(self, old_policy, obs, v: torch.Tensor, chunk_size: int) -> torch.Tensor:
        if self.fvp_kl_metric != "average":
            raise ValueError("Chunked Fisher-vector products currently support only average KL metric.")
        total_n = float(len(obs))
        total_hvp = None
        for chunk_len, obs_chunk, _ in self._iter_chunks(obs, chunk_size=chunk_size):
            with torch.no_grad():
                old_dist = old_policy.distribution(obs_chunk)
            new_dist = self.policy.distribution(obs_chunk)
            kl = mean_kl(old_dist, new_dist) * (chunk_len / total_n)
            grad_kl = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = flat_grad(grad_kl)
            kl_v = (flat_grad_kl * v).sum()
            hvp = torch.autograd.grad(kl_v, self.policy.parameters())
            flat_hvp = flat_grad(hvp).detach()
            total_hvp = flat_hvp if total_hvp is None else total_hvp + flat_hvp
        if total_hvp is None:
            total_hvp = torch.zeros_like(v)
        damping = float(self.cfg.algo.get("cg_damping", 1e-1))
        return total_hvp + damping * v

    def _fvp_on_batch(self, obs, old_dist, v: torch.Tensor) -> torch.Tensor:
        new_dist = self.policy.distribution(obs)
        kl = self._kl_metric(old_dist, new_dist, metric=self.fvp_kl_metric)
        grad_kl = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = flat_grad(grad_kl)
        kl_v = (flat_grad_kl * v).sum()
        hvp = torch.autograd.grad(kl_v, self.policy.parameters())
        damping = float(self.cfg.algo.get("cg_damping", 1e-1))
        return flat_grad(hvp).detach() + damping * v

    def _surrogate(self, obs, act, weight, old_logp):
        dist = self.policy.distribution(obs)
        logp = self.policy.log_prob_from_dist(dist, act)
        ratio = torch.exp(logp - old_logp)
        return (ratio * weight).mean()

    def _kl_metric(self, old_dist, new_dist, metric: str) -> torch.Tensor:
        metric = metric.lower()
        if metric == "average":
            return mean_kl(old_dist, new_dist)
        if metric == "max":
            return max_kl(old_dist, new_dist)
        raise ValueError(f"Unsupported KL metric: {metric}")

    def _value_loss(self, obs, ret):
        if self.value_fn is None:
            raise RuntimeError("Value loss requested but value function is disabled.")
        pred = self.value_fn(obs)
        return ((pred - ret) ** 2).mean()

    def _update_value_function(self, obs, ret) -> None:
        if self.value_fn is None or self.vf_optimizer is None:
            return
        vf_iters = int(self.cfg.algo.get("vf_iters", 80))
        batch_size = int(self.cfg.algo.get("vf_batch_size", len(obs)))
        n = len(obs)
        for _ in range(vf_iters):
            permutation = torch.randperm(n, device=obs.device)
            for start in range(0, n, batch_size):
                idx = permutation[start : start + batch_size]
                loss = self._value_loss(obs[idx], ret[idx])
                self.vf_optimizer.zero_grad()
                loss.backward()
                self.vf_optimizer.step()
