import numpy as np
import math
import pandas as pd
from typing import Tuple, List, Dict


# ------------------------------------------------------------------#
# Helper functions
# ------------------------------------------------------------------#
def softmax(q: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Temperature-controlled soft-max."""
    z = (q / alpha) - np.max(q / alpha)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


def kl_to_uniform(p: np.ndarray) -> float:
    """KL(p || uniform)."""
    n = p.size
    p_safe = np.clip(p, 1e-15, 1.0)
    return float(np.sum(p_safe * (np.log(p_safe) + math.log(n))))


def divergence_function(
    q_old: np.ndarray, alpha: float, fixed_idx: int
) -> callable:
    """Return D(k) = KL(softmax(q(k)) || uniform) as a scalar fn of k."""
    d = q_old - q_old[fixed_idx]  # centred differences wrt fixed entry

    def D(k: float) -> float:
        q_k = q_old[fixed_idx] + k * d
        p = softmax(q_k, alpha)
        return kl_to_uniform(p)

    return D


def find_scale_k(
    D: callable,
    delta: float,
    eps: float = 1e-6,
    k_low: float = 1e-9,
    k_high: float = 1.0,
    max_iter: int = 100,
) -> float:
    """Monotone bisection search for D(k)=delta."""
    # enlarge k_high until upper bracket found
    while D(k_high) < delta:
        k_high *= 2.0
        if k_high > 1e9:
            raise ValueError("Cannot bracket root; delta may be too large.")
    # bisection
    for _ in range(max_iter):
        k_mid = 0.5 * (k_low + k_high)
        D_mid = D(k_mid)
        if abs(D_mid - delta) <= eps:
            return k_mid
        if D_mid < delta:
            k_low = k_mid
        else:
            k_high = k_mid
    return 0.5 * (k_low + k_high)


def find_q_new(
    q_old: np.ndarray,
    alpha: float,
    delta: float,
    fixed_index: int = 0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float, float]:
    """Compute q_new satisfying constraints for arbitrary fixed index."""
    if not (0 <= fixed_index < q_old.size):
        raise IndexError("fixed_index out of range.")

    D = divergence_function(q_old, alpha, fixed_index)
    k_star = find_scale_k(D, delta, eps=eps)
    q_new = q_old[fixed_index] + k_star * (q_old - q_old[fixed_index])
    achieved_kl = D(k_star)
    return q_new, achieved_kl, k_star


def order_preserved(q_old: np.ndarray, q_new: np.ndarray) -> bool:
    """Check monotone order preservation."""
    for i in range(len(q_old)):
        for j in range(len(q_old)):
            if (q_old[i] - q_old[j]) * (q_new[i] - q_new[j]) < -1e-12:
                return False
    return True


# ------------------------------------------------------------------#
# Demonstration with various fixed indices
# ------------------------------------------------------------------#
alpha = 1.0
test_cases: List[Dict] = [
    {
        "name": "A",
        "q_old": np.array([2.0, 1.5, 0.2, -0.5, -1.0]),
        "delta": 0.01,
        "fixed": 2,
    },
    {
        "name": "B",
        "q_old": np.array([1.0, -0.1, 0.0, -1.0, 2.0]),
        "delta": 0.2,
        "fixed": 4,
    },
    {
        "name": "C",
        "q_old": np.array([-3.0, -1.0, 0.0, 1.0, 3.0]),
        "delta": 0.1,
        "fixed": 3,
    },
]

rows = []
for case in test_cases:
    q_old = case["q_old"]
    delta = case["delta"]
    fixed = case["fixed"]
    q_new, kl_val, k_star = find_q_new(q_old, alpha, delta, fixed_index=fixed)
    rows.append(
        {
            "Case": case["name"],
            "Fixed idx": fixed,
            "n": q_old.size,
            "δ target": delta,
            "KL achieved": kl_val,
            "|Δ|": abs(kl_val - delta),
            "k*": k_star,
            "Order OK": order_preserved(q_old, q_new),
            "q_old": np.round(q_old, 3),
            "q_new": np.round(q_new, 3),
        }
    )

print(rows)


