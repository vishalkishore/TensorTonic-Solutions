
import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    mask = p > 0
    p = p[mask]
    q = q[mask]

    return float(np.sum(
        p * (np.log(p) - np.log(np.maximum(q, eps)))
    ))
