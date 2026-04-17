from __future__ import annotations

from typing import Any

import torch

from trpo_repro.algos.trpo import TRPOAgent
from trpo_repro.methods.base import BaseMethod, MethodUpdateStats
from trpo_repro.models.policies import make_policy
from trpo_repro.models.value_functions import ValueFunction


class SecondOrderPolicyMethod(BaseMethod):
    trainable = True
    supports_checkpoints = True

    method_name = "trpo"
    update_mode = "trpo"
    kl_constraint_metric = "average"
    fvp_kl_metric = "average"

    def __init__(self, obs_space, act_space, cfg, device: torch.device) -> None:
        super().__init__(cfg=cfg, device=device)
        self.estimator = self._resolve_estimator(cfg)
        self.use_value_function = self.estimator in {"mc_baseline", "gae"}
        self.policy = make_policy(obs_space, act_space, cfg)
        self.value_fn = ValueFunction(tuple(obs_space.shape), cfg) if self.use_value_function else None
        fvp_kl_metric = str(cfg.algo.get("fvp_kl_metric", self.fvp_kl_metric)).lower()
        self.agent = TRPOAgent(
            self.policy,
            self.value_fn,
            cfg,
            device,
            update_mode=self.update_mode,
            kl_constraint_metric=self.kl_constraint_metric,
            fvp_kl_metric=fvp_kl_metric,
        )

    @staticmethod
    def _resolve_estimator(cfg) -> str:
        explicit = cfg.algo.get("estimator")
        if explicit is not None:
            return str(explicit).lower()
        legacy = str(cfg.algo.get("advantage_mode", "mc")).lower()
        mapping = {
            "mc": "mc_baseline",
            "gae": "gae",
        }
        return mapping.get(legacy, legacy)

    @property
    def name(self) -> str:
        return self.method_name

    @property
    def variant(self) -> str:
        return self.estimator

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        return self.agent.step(obs, deterministic=deterministic)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.agent.value(obs)

    def update(self, batch: dict[str, torch.Tensor] | None) -> MethodUpdateStats:
        if batch is None:
            return MethodUpdateStats(did_update=0.0)
        stats = self.agent.update(batch)
        line_search_success = float(int(stats.line_search_success)) if self.update_mode == "trpo" else float("nan")
        return MethodUpdateStats(
            policy_loss_before=stats.policy_loss_before,
            policy_loss_after=stats.policy_loss_after,
            value_loss_before=stats.value_loss_before,
            value_loss_after=stats.value_loss_after,
            entropy=stats.entropy,
            approx_kl=stats.approx_kl,
            line_search_success=line_search_success,
            cg_norm=stats.cg_norm,
            did_update=1.0,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "value_fn": None if self.value_fn is None else self.value_fn.state_dict(),
            "method_name": self.name,
            "method_variant": self.variant,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "policy" in state and state["policy"] is not None:
            self.policy.load_state_dict(state["policy"])
        if self.value_fn is not None and state.get("value_fn") is not None:
            self.value_fn.load_state_dict(state["value_fn"])


class TRPOMethod(SecondOrderPolicyMethod):
    method_name = "trpo"
    update_mode = "trpo"
    kl_constraint_metric = "average"
    fvp_kl_metric = "average"


class NaturalPolicyGradientMethod(SecondOrderPolicyMethod):
    method_name = "natural_pg"
    update_mode = "npg"
    kl_constraint_metric = "average"
    fvp_kl_metric = "average"


class TRPOMaxKLMethod(SecondOrderPolicyMethod):
    method_name = "trpo_max_kl"
    update_mode = "trpo"
    kl_constraint_metric = "max"
    fvp_kl_metric = "average"
