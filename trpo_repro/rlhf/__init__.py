"""RLHF utilities for token-level PPO post-training of language models.

This subpackage intentionally lives beside the original MuJoCo/Atari PPO code.
The classical RL stack is kept intact; these modules reuse the same PPO ideas for
language-model rollouts where actions are generated tokens.
"""

__all__ = [
    "data",
    "formatting",
    "reward_model",
    "lm_policy",
    "rollout",
    "ppo_lm",
    "metrics",
]
