from __future__ import annotations

import torch

def conjugate_gradient(Avp, b: torch.Tensor, nsteps: int, residual_tol: float = 1e-10) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(nsteps):
        Avp_p = Avp(p)
        alpha = rdotr / (torch.dot(p, Avp_p) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Avp_p
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x