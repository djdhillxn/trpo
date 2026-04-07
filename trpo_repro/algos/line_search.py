from __future__ import annotations

from collections.abc import Callable

import torch

EvalFn = Callable[[torch.Tensor], tuple[float, float]]

def backtracking_line_search(
        old_params: torch.Tensor,
        full_steps: torch.Tensor,
        evaluate: EvalFn,
        backtrack_coeff: float,
        max_backtracks: int, 
        expected_improve_rate: float,
        max_kl: float,
) -> tuple[torch.Tensor, bool, float, float]:
    for step_idx in range(max_backtracks):
        stepfrac = backtrack_coeff**step_idx
        candidate = old_params + stepfrac * full_steps
        surrogate, kl = evaluate(candidate)
        actual_improve = surrogate
        expected_improve = expected_improve_rate * stepfrac
        improvement_ok = actual_improve > 0.0
        ratio_ok = actual_improve >= 0.1 * expected_improve
        kl_ok = kl <= max_kl
        if improvement_ok and ratio_ok and kl_ok: 
            return candidate, True, surrogate, kl
    return old_params, False, 0.0, 0.0