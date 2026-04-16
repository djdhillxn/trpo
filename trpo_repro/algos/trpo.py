from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from trpo_repro.algos.conjugate_gradient import conjugate_gradient
from trpo_repro.algos.line_search import backtracking_line_search
from trpo_repro.models.policies import max_kl, mean_kl
from trpo_repro.utils.torch_utils import flat_grad, flat_params, set_flat_params


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
    """Shared second-order policy optimizer for TRPO-family methods.

    update_mode:
      - "trpo": trust-region step with backtracking line search
      - "npg": natural-gradient step with a fixed scalar stepsize and no line search

    kl_constraint_metric:
      - "average": average batch KL, matching practical TRPO
      - "max": maximum per-state batch KL, useful for small CartPole-style variants

    fvp_kl_metric controls which KL surrogate defines the local metric used in the
    Fisher-vector product. For the max-KL variant we keep the default as "average"
    because it is smoother and more stable, while the line search enforces the max KL.
    """

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

    def update(self, batch: dict[str, torch.Tensor]) -> TRPOStats:
        obs = batch["obs"]
        act = batch["act"]
        ret = batch["ret"]
        weight = batch.get("weight", batch["adv"])
        old_logp = batch["logp"]

        with torch.no_grad():
            old_dist = self.policy.distribution(obs)
            old_surr = self._surrogate(obs, act, weight, old_logp).item()
            entropy = old_dist.entropy().mean().item()
            if self.value_fn is None:
                value_loss_before = float("nan")
            else:
                value_loss_before = self._value_loss(obs, ret).item()

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
            new_params, success, approx_kl = self._apply_trpo_step(
                obs=obs,
                act=act,
                weight=weight,
                old_logp=old_logp,
                old_dist=old_dist,
                old_surr=old_surr,
                gradient=g,
                step_dir=step_dir,
            )
        elif self.update_mode == "npg":
            new_params, success, approx_kl = self._apply_npg_step(
                obs=obs,
                old_dist=old_dist,
                step_dir=step_dir,
            )
        else:
            raise ValueError(f"Unsupported update mode: {self.update_mode}")

        set_flat_params(self.policy, new_params)

        with torch.no_grad():
            policy_loss_after = self._surrogate(obs, act, weight, old_logp).item()

        return TRPOStats(
            policy_loss_before=old_surr,
            policy_loss_after=policy_loss_after,
            value_loss_before=value_loss_before,
            value_loss_after=value_loss_after,
            entropy=entropy,
            approx_kl=approx_kl,
            line_search_success=success,
            cg_norm=step_dir.norm().item(),
        )

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
            improvement = new_surr - old_surr
            return improvement, kl_val

        new_params, success, _, approx_kl = backtracking_line_search(
            old_params=old_params,
            full_step=full_step,
            evaluate=evaluate,
            backtrack_coeff=float(self.cfg.algo.get("backtrack_coeff", 0.8)),
            max_backtracks=int(self.cfg.algo.get("backtrack_iters", 10)),
            expected_improve_rate=expected_improve,
            max_kl=float(self.cfg.algo.max_kl),
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
