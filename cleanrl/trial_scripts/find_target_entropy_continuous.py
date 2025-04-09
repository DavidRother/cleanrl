import math

# These constants come from your actor definition.
LOG_STD_MAX = 2
LOG_STD_MIN = -5

def target_entropy_from_exploitation_probability(p, d, p_min=None):
    """
    Computes the SAC-style target entropy for a continuous Gaussian policy
    in d dimensions, where the Gaussian's log_std is constrained between LOG_STD_MIN and LOG_STD_MAX.
    The exploitation probability p is interpreted as a scalar that, in analogy with the discrete case,
    varies between p_min (fully exploratory) and 1 (fully exploitative).

    Parameters:
      p (float): Exploitation probability (best action mass). Must be in (p_min, 1).
      d (int): Dimensionality of the action space.
      p_min (float, optional): The minimum exploitation probability corresponding
          to full exploration. By default, we set p_min = 1/d.

    Returns:
      float: The SAC target entropy (a negative value), computed as -H,
             where H is the differential entropy of an isotropic Gaussian with effective log_std
             chosen by interpolating between LOG_STD_MAX (exploration) and LOG_STD_MIN (exploitation).

    Raises:
      ValueError: If p is not in the interval (p_min, 1).
    """
    if p_min is None:
        p_min = 1.0 / d  # analogous to uniform allocation in the discrete case
    if p <= p_min or p >= 1.0:
        raise ValueError(f"Exploitation probability must be in the open interval ({p_min}, 1).")

    # Normalize exploitation probability between 0 and 1.
    # When p = p_min, normalized_exploitation = 0 (fully exploratory).
    # When p -> 1, normalized_exploitation -> 1 (fully exploitative).
    normalized_exploitation = (p - p_min) / (1.0 - p_min)

    # Linearly interpolate between LOG_STD_MAX (exploration) and LOG_STD_MIN (exploitation)
    target_log_std = LOG_STD_MAX - normalized_exploitation * (LOG_STD_MAX - LOG_STD_MIN)

    # Compute per-dimension differential entropy of a Gaussian:
    # H_i = 0.5 * [1 + ln(2*pi*exp(2*target_log_std))]
    #     = 0.5 * [1 + ln(2*pi) + 2*target_log_std]
    H_per_dim = 0.5 * (1 + math.log(2 * math.pi) + 2 * target_log_std)

    # Total entropy for d dimensions.
    H_total = d * H_per_dim

    # SAC target entropy is defined as the negative of the entropy.
    return H_total

# Example usage:
if __name__ == "__main__":
    action_dim = 6  # for example, a 3-dimensional action space
    p_exploitation = 0.7  # user-defined exploitation probability
    target_ent = target_entropy_from_exploitation_probability(p_exploitation, action_dim)
    print("For a {}-dimensional Gaussian policy with exploitation probability p = {:.2f}:".format(action_dim, p_exploitation))
    print("SAC target entropy = {:.4f}".format(target_ent))