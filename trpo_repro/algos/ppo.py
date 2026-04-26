from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from trpo_repro.models.policies import mean_kl
from trpo_repro.utils.torch_utils import to_tensor


@dataclass
class PPOStats:
    policy_loss_before: float
    policy_loss_after: float
    value_loss_before: float
    value_loss_after: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    kl_coef: float
    value_explained_variance_before: float
    value_explained_variance_after: float


class PPOAgent:
    def __init__(
        self,
        policy: nn.Module,
        value_fn: nn.Module,
        cfg,
        device: torch.device,
        *,
        variant: str = "clip",
    ) -> None:
        self.policy = policy.to(device)
        self.value_fn = value_fn.to(device)
        self.cfg = cfg
        self.device = device
        self.variant = variant.lower()
        if self.variant not in {"clip", "kl_penalty", "kl"}:
            raise ValueError(f"Unsupported PPO variant: {variant}")
        if self.variant == "kl":
            self.variant = "kl_penalty"
        self.base_pi_lr = float(cfg.algo.get("ppo_pi_lr", 3e-4))
        self.base_vf_lr = float(cfg.algo.get("ppo_vf_lr", cfg.algo.get("vf_lr", 1e-3)))
        self.base_clip_ratio = float(cfg.algo.get("ppo_clip_ratio", 0.2))
        self.anneal_lr = bool(cfg.algo.get("ppo_anneal_lr", False))
        self.anneal_clip_ratio = bool(cfg.algo.get("ppo_anneal_clip_ratio", False))
        self.progress_remaining = 1.0
        self.current_clip_ratio = self.base_clip_ratio
        self.pi_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.base_pi_lr)
        self.vf_optimizer = torch.optim.Adam(
            self.value_fn.parameters(),
            lr=self.base_vf_lr,
            weight_decay=float(cfg.algo.get("vf_weight_decay", 0.0)),
        )
        self.kl_coef = float(cfg.algo.get("ppo_kl_coef", 1.0))
        self._apply_schedules()

    def _apply_schedules(self) -> None:
        lr_mult = self.progress_remaining if self.anneal_lr else 1.0
        clip_mult = self.progress_remaining if self.anneal_clip_ratio else 1.0
        for group in self.pi_optimizer.param_groups:
            group["lr"] = self.base_pi_lr * lr_mult
        for group in self.vf_optimizer.param_groups:
            group["lr"] = self.base_vf_lr * lr_mult
        self.current_clip_ratio = self.base_clip_ratio * clip_mult

    def set_training_progress(self, progress_remaining: float) -> None:
        self.progress_remaining = float(np.clip(progress_remaining, 0.0, 1.0))
        self._apply_schedules()

    @torch.no_grad()
    def step(self, obs: torch.Tensor, deterministic: bool = False):
        action, logp, _ = self.policy.act(obs, deterministic=deterministic)
        value = self.value_fn(obs)
        return action, value, logp

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_fn(obs)

    def update(self, batch) -> PPOStats:
        obs = batch["obs"]
        act = batch["act"]
        ret = batch["ret"]
        adv = batch.get("adv", batch.get("weight"))
        old_logp = batch["logp"]
        n = int(batch.get("size", len(ret)))

        old_policy = copy.deepcopy(self.policy).to(self.device)
        old_policy.eval()
        for param in old_policy.parameters():
            param.requires_grad_(False)

        policy_loss_before, _, _, _ = self._policy_metrics(self.policy, old_policy, obs, act, adv, old_logp, n)
        value_loss_before, value_ev_before = self._value_metrics(obs, ret, n)

        self._update_policy(obs, act, adv, old_logp, old_policy, n)
        self._update_value_function(obs, ret, n)

        policy_loss_after, entropy_after, approx_kl_after, clip_fraction_after = self._policy_metrics(
            self.policy, old_policy, obs, act, adv, old_logp, n
        )
        value_loss_after, value_ev_after = self._value_metrics(obs, ret, n)

        if self.variant == "kl_penalty" and bool(self.cfg.algo.get("ppo_adaptive_kl", True)):
            target_kl = float(self.cfg.algo.get("ppo_target_kl", 0.01))
            if approx_kl_after > 1.5 * target_kl:
                self.kl_coef *= 2.0
            elif approx_kl_after < target_kl / 1.5:
                self.kl_coef *= 0.5
            self.kl_coef = float(np.clip(self.kl_coef, 1e-4, 1e4))

        return PPOStats(
            policy_loss_before=policy_loss_before,
            policy_loss_after=policy_loss_after,
            value_loss_before=value_loss_before,
            value_loss_after=value_loss_after,
            entropy=entropy_after,
            approx_kl=approx_kl_after,
            clip_fraction=clip_fraction_after,
            kl_coef=self.kl_coef if self.variant == "kl_penalty" else float("nan"),
            value_explained_variance_before=value_ev_before,
            value_explained_variance_after=value_ev_after,
        )

    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "value_fn": self.value_fn.state_dict(),
            "ppo_kl_coef": self.kl_coef,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("policy") is not None:
            self.policy.load_state_dict(state["policy"])
        if state.get("value_fn") is not None:
            self.value_fn.load_state_dict(state["value_fn"])
        if state.get("ppo_kl_coef") is not None:
            self.kl_coef = float(state["ppo_kl_coef"])
        self._apply_schedules()

    def _update_policy(self, obs, act, adv, old_logp, old_policy, n: int) -> None:
        minibatch_size = int(self.cfg.algo.get("ppo_minibatch_size", 256))
        update_epochs = int(self.cfg.algo.get("ppo_update_epochs", 10))
        clip_ratio = self.current_clip_ratio
        entropy_coef = float(self.cfg.algo.get("ppo_entropy_coef", 0.0))
        target_kl = float(self.cfg.algo.get("ppo_target_kl", 0.01))
        max_grad_norm = self.cfg.algo.get("ppo_max_grad_norm")
        max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        stop_early = False

        for _ in range(update_epochs):
            for idx in self._iter_minibatches(n, minibatch_size, shuffle=True):
                idx_t = self._index_tensor(idx, act.device)
                obs_mb = self._obs_minibatch(obs, idx)
                act_mb = act.index_select(0, idx_t)
                adv_mb = adv.index_select(0, idx_t)
                old_logp_mb = old_logp.index_select(0, idx_t)

                with torch.no_grad():
                    old_dist_mb = old_policy.distribution(obs_mb)

                new_dist_mb = self.policy.distribution(obs_mb)
                logp_mb = self.policy.log_prob_from_dist(new_dist_mb, act_mb)
                ratio = torch.exp(logp_mb - old_logp_mb)
                surrogate = ratio * adv_mb
                if self.variant == "clip":
                    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_mb
                    objective = torch.minimum(surrogate, clipped).mean()
                else:
                    kl_mb = mean_kl(old_dist_mb, new_dist_mb)
                    objective = surrogate.mean() - self.kl_coef * kl_mb
                entropy_mb = new_dist_mb.entropy().mean()
                loss_pi = -(objective + entropy_coef * entropy_mb)

                self.pi_optimizer.zero_grad(set_to_none=True)
                loss_pi.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.pi_optimizer.step()

                if self.variant == "clip":
                    with torch.no_grad():
                        approx_kl_mb = mean_kl(old_dist_mb, self.policy.distribution(obs_mb)).item()
                    if approx_kl_mb > 1.5 * target_kl:
                        stop_early = True
                        break
            if stop_early:
                break

    def _update_value_function(self, obs, ret, n: int) -> None:
        minibatch_size = int(self.cfg.algo.get("ppo_minibatch_size", 256))
        vf_epochs = int(self.cfg.algo.get("ppo_vf_epochs", self.cfg.algo.get("ppo_update_epochs", 10)))
        max_grad_norm = self.cfg.algo.get("ppo_max_grad_norm")
        max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        for _ in range(vf_epochs):
            for idx in self._iter_minibatches(n, minibatch_size, shuffle=True):
                idx_t = self._index_tensor(idx, ret.device)
                obs_mb = self._obs_minibatch(obs, idx)
                ret_mb = ret.index_select(0, idx_t)
                value_pred = self.value_fn(obs_mb)
                value_loss = ((value_pred - ret_mb) ** 2).mean()
                self.vf_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.value_fn.parameters(), max_grad_norm)
                self.vf_optimizer.step()

    @torch.no_grad()
    def _policy_metrics(self, policy, old_policy, obs, act, adv, old_logp, n: int):
        batch_size = int(self.cfg.algo.get("ppo_eval_batch_size", self.cfg.algo.get("ppo_minibatch_size", 256)))
        clip_ratio = self.current_clip_ratio
        total_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip = 0.0
        total_count = 0
        for idx in self._iter_minibatches(n, batch_size, shuffle=False):
            idx_t = self._index_tensor(idx, act.device)
            obs_mb = self._obs_minibatch(obs, idx)
            act_mb = act.index_select(0, idx_t)
            adv_mb = adv.index_select(0, idx_t)
            old_logp_mb = old_logp.index_select(0, idx_t)
            old_dist_mb = old_policy.distribution(obs_mb)
            new_dist_mb = policy.distribution(obs_mb)
            logp_mb = policy.log_prob_from_dist(new_dist_mb, act_mb)
            ratio = torch.exp(logp_mb - old_logp_mb)
            surrogate = ratio * adv_mb
            if self.variant == "clip":
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_mb
                objective = torch.minimum(surrogate, clipped).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean().item()
            else:
                kl_mb = mean_kl(old_dist_mb, new_dist_mb)
                objective = surrogate.mean() - self.kl_coef * kl_mb
                clip_fraction = float("nan")
            loss = -objective.item()
            entropy = new_dist_mb.entropy().mean().item()
            kl_val = mean_kl(old_dist_mb, new_dist_mb).item()
            count = len(idx)
            total_loss += loss * count
            total_entropy += entropy * count
            total_kl += kl_val * count
            if not np.isnan(clip_fraction):
                total_clip += clip_fraction * count
            total_count += count
        clip_stat = total_clip / total_count if self.variant == "clip" and total_count > 0 else float("nan")
        return total_loss / total_count, total_entropy / total_count, total_kl / total_count, clip_stat

    @torch.no_grad()
    def _value_metrics(self, obs, ret, n: int) -> tuple[float, float]:
        batch_size = int(self.cfg.algo.get("ppo_eval_batch_size", self.cfg.algo.get("ppo_minibatch_size", 256)))
        total_loss = 0.0
        total_count = 0
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for idx in self._iter_minibatches(n, batch_size, shuffle=False):
            idx_t = self._index_tensor(idx, ret.device)
            obs_mb = self._obs_minibatch(obs, idx)
            ret_mb = ret.index_select(0, idx_t)
            value_pred = self.value_fn(obs_mb)
            loss = ((value_pred - ret_mb) ** 2).mean().item()
            count = len(idx)
            total_loss += loss * count
            total_count += count
            preds.append(value_pred.detach().cpu())
            targets.append(ret_mb.detach().cpu())
        if total_count == 0:
            return float("nan"), float("nan")
        pred = torch.cat(preds).numpy()
        target = torch.cat(targets).numpy()
        variance = float(np.var(target))
        if variance <= 1e-8:
            explained_variance = float("nan")
        else:
            explained_variance = 1.0 - float(np.var(target - pred) / variance)
        return total_loss / total_count, explained_variance

    @staticmethod
    def _iter_minibatches(n: int, batch_size: int, *, shuffle: bool):
        indices = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            yield indices[start : start + batch_size]

    @staticmethod
    def _index_tensor(index: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(index, device=device, dtype=torch.long)

    def _obs_minibatch(self, obs, index: np.ndarray) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            idx_t = self._index_tensor(index, obs.device)
            batch = obs.index_select(0, idx_t)
            return batch.to(self.device, dtype=torch.float32)
        return to_tensor(obs[index], self.device, dtype=torch.float32)
