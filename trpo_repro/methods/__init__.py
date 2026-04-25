from __future__ import annotations

import torch

from trpo_repro.methods.ppo_method import PPOMethod
from trpo_repro.methods.random_policy import RandomPolicyMethod
from trpo_repro.methods.trpo_method import NaturalPolicyGradientMethod, TRPOMaxKLMethod, TRPOMethod


_METHODS = {
    "trpo": TRPOMethod,
    "natural_pg": NaturalPolicyGradientMethod,
    "npg": NaturalPolicyGradientMethod,
    "trpo_max_kl": TRPOMaxKLMethod,
    "ppo": PPOMethod,
    "random": RandomPolicyMethod,
}


def resolve_method_name(cfg) -> str:
    method_cfg = cfg.get("method", {})
    if isinstance(method_cfg, dict):
        name = method_cfg.get("name")
    else:
        name = None
    return str(name or "trpo").lower()


def make_method(obs_space, act_space, cfg, device: torch.device):
    name = resolve_method_name(cfg)
    if name not in _METHODS:
        raise ValueError(f"Unsupported method: {name}")
    return _METHODS[name](obs_space=obs_space, act_space=act_space, cfg=cfg, device=device)
