from typing import Any

import torch

from trpo_repro.algos.advantages import canonicalize_estimator
from trpo_repro.algos.ppo import PPOAgent
from trpo_repro.methods.base import BaseMethod, MethodUpdateStats
from trpo_repro.models.policies import make_policy
from trpo_repro.models.value_functions import ValueFunction


class PPOMethod(BaseMethod):
    trainable = True
    supports_checkpoints = True
    batch_obs_to_device = False

    def __init__(self, obs_space, act_space, cfg, device: torch.device) -> None:
        super().__init__(cfg=cfg, device=device)
        self.estimator = self._resolve_estimator(cfg)
        if self.estimator not in {"gae", "value_baseline"}:
            raise ValueError(f"PPO supports gae or value_baseline estimators, got: {self.estimator}")
        self._variant = self._resolve_variant(cfg)
        self.policy = make_policy(obs_space, act_space, cfg)
        self.value_fn = ValueFunction(tuple(obs_space.shape), cfg)
        self.agent = PPOAgent(self.policy, self.value_fn, cfg, device, variant=self._variant)

    @staticmethod
    def _resolve_estimator(cfg) -> str:
        explicit = cfg.algo.get("estimator")
        if explicit is not None:
            return canonicalize_estimator(explicit)
        legacy = str(cfg.algo.get("advantage_mode", "gae")).lower()
        return canonicalize_estimator(legacy)

    @staticmethod
    def _resolve_variant(cfg) -> str:
        method_cfg = cfg.get("method", {})
        variant = "clip"
        if isinstance(method_cfg, dict):
            variant = str(method_cfg.get("variant", "clip")).lower()
        aliases = {
            "clipped": "clip",
            "clip": "clip",
            "kl": "kl_penalty",
            "kl_penalty": "kl_penalty",
            "penalty": "kl_penalty",
        }
        return aliases.get(variant, variant)

    @property
    def name(self) -> str:
        return "ppo"

    @property
    def variant(self) -> str:
        return self._variant

    def set_training_progress(self, epoch: int, total_epochs: int) -> None:
        if total_epochs <= 1:
            progress_remaining = 1.0
        else:
            progress_remaining = max(0.0, 1.0 - float(epoch - 1) / float(total_epochs - 1))
        self.agent.set_training_progress(progress_remaining)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        return self.agent.step(obs, deterministic=deterministic)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.agent.value(obs)

    def update(self, batch) -> MethodUpdateStats:
        if batch is None:
            return MethodUpdateStats(did_update=0.0)
        stats = self.agent.update(batch)
        return MethodUpdateStats(
            policy_loss_before=stats.policy_loss_before,
            policy_loss_after=stats.policy_loss_after,
            value_loss_before=stats.value_loss_before,
            value_loss_after=stats.value_loss_after,
            entropy=stats.entropy,
            approx_kl=stats.approx_kl,
            line_search_success=float("nan"),
            cg_norm=float("nan"),
            clip_fraction=stats.clip_fraction,
            kl_coef=stats.kl_coef,
            value_explained_variance_before=stats.value_explained_variance_before,
            value_explained_variance_after=stats.value_explained_variance_after,
            did_update=1.0,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "value_fn": self.value_fn.state_dict(),
            "ppo_kl_coef": self.agent.kl_coef,
            "method_name": self.name,
            "method_variant": self.variant,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("policy") is not None:
            self.policy.load_state_dict(state["policy"])
        if state.get("value_fn") is not None:
            self.value_fn.load_state_dict(state["value_fn"])
        if state.get("ppo_kl_coef") is not None:
            self.agent.kl_coef = float(state["ppo_kl_coef"])
