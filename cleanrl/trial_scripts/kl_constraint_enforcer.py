import numpy as np

np.set_printoptions(precision=3, suppress=True)


def kl_project(q, actions, delta, alpha, max_iter=3, tol=1e-6):
    """
    Project a batch of Q-value rows onto the KL-constraint surface
    by shifting all non-taken logits with the SAME constant c.

    Parameters
    ----------
    q        : (B, A)  float32/64
    actions  : (B,)    int, index of the taken action in each row
    delta    : scalar  desired KL(π || uniform)
    alpha    : scalar  temperature used in softmax
    Returns
    -------
    q_new : (B, A)  with ordering preserved and KL == delta (up to tol)
    """
    B, A = q.shape
    target_H = np.log(A) - delta  # scalar

    # ----- pre-compute per-sample quantities --------------------------------
    idx = np.arange(B)
    qa = q[idx, actions]  # (B,)
    Aa = np.exp(qa / alpha)  # (B,)

    mask = np.ones_like(q, dtype=bool)
    mask[idx, actions] = False
    qb = q[mask].reshape(B, A - 1)  # (B, A−1)
    exp_qb = np.exp(qb / alpha)
    S = exp_qb.sum(1)  # (B,)

    w = exp_qb / S[:, None]  # (B, A−1)
    Hw = -np.sum(w * np.log(w + 1e-12), axis=1)  # (B,)

    # ----- Newton solve for x = p_b ------------------------------------------------
    # start at current probability mass on non-taken actions
    x = S / (Aa + S)  # (B,)

    for _ in range(max_iter):
        # f(x)  = H_bin(x) + x*H_w - target_H
        Hbin = -x * np.log(x + 1e-12) - (1 - x) * np.log(1 - x + 1e-12)
        f = Hbin + x * Hw - target_H  # (B,)
        if np.all(np.abs(f) < tol):
            break
        # f'(x) = ln((1-x)/x) + H_w
        fp = np.log((1 - x) / (x + 1e-12)) + Hw
        x = np.clip(x - f / fp, 1e-8, 1 - 1e-8)  # stay in (0,1)

    # ----- compute constant shift ------------------------------------------
    k = (x / (1 - x)) * (Aa / S)  # (B,)
    c = alpha * np.log(k)  # (B,)

    q_new = q.copy()
    q_new[mask] += np.repeat(c[:, None], A - 1, axis=1).ravel()
    return q_new


# ---- Demo ------------------------------------------------------------------
np.random.seed(42)
B, A = 3, 5  # 3 states, 5 discrete actions
alpha = 0.5
delta = 0.4  # target KL

q_before = np.random.randn(B, A) * 2
actions = np.random.randint(0, A, size=B)

q_after = kl_project(q_before, actions, delta, alpha, max_iter=1000, tol=1e-6)

print("Taken actions:", actions, "\n")
print("Q before adjustment:\n", q_before, "\n")
print("Q after  adjustment:\n", q_after, "\n")


# Verify KL & ordering
def entropy(p): return -np.sum(p * np.log(p + 1e-12))


uniform_kl = []
for i in range(B):
    p_before = np.exp(q_before[i] / alpha)
    p_before /= p_before.sum()
    p_after = np.exp(q_after[i] / alpha)
    p_after /= p_after.sum()
    kl_after = np.sum(p_after * np.log(p_after * (A)))  # KL(π || uniform) = logA - H(π)
    uniform_kl.append(kl_after)
    # assert np.all(np.argsort(-p_before) == np.argsort(-p_after)), "Ordering changed!"

print("KL(π || U) after adjustment for each state:", np.round(uniform_kl, 4))

