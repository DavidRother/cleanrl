import math


def target_entropy_from_exploitation_probability_binary(p):
    """
    Computes the SAC-style target entropy for a binary distribution where:
      - Best action: probability p
      - Second action: probability (1-p)

    The entropy is given by:
        H(p) = -[ p * log(p) + (1-p) * log(1-p) ]
    And the SAC target entropy is defined as the negative of this computed entropy.

    Parameters:
        p (float): Exploitation probability for the best action (must be in (0, 1)).

    Returns:
        float: The SAC target entropy (a negative value).

    Raises:
        ValueError: If p is not strictly between 0 and 1.
    """
    if p <= 0 or p >= 1:
        raise ValueError("Exploitation probability p must be in the open interval (0, 1).")

    # Compute the binary entropy
    H = - (p * math.log(p) + (1 - p) * math.log(1 - p))

    # SAC target entropy is the negative of the computed entropy.
    return H


def target_entropy_from_exploitation_probability(p, n):
    """
    Computes the SAC-style target entropy given an exploitation probability p and
    the number of actions n.

    In this setting:
      - Best action: probability p
      - Other actions: probability (1-p) uniformly divided among the n-1 remaining actions

    The entropy of such a distribution is:
        H(p) = -[ p * log(p) + (1-p) * log((1-p)/(n-1)) ]
    And the SAC target entropy is defined as negative H(p).

    Parameters:
      p (float): Exploitation probability for the best action (0 < p < 1).
      n (int): Number of available actions.

    Returns:
      float: SAC target entropy (negative value).

    Raises:
      ValueError: If p is not strictly between 0 and 1.
    """
    if p <= 0 or p >= 1:
        raise ValueError("Exploitation probability p must be in the open interval (0, 1).")

    # Compute the entropy of the distribution.
    entropy = - (p * math.log(p) + (1 - p) * math.log((1 - p) / (n - 1)))

    # Return the SAC-style target entropy (i.e., negative of the computed entropy).
    return entropy


# Example usage:
if __name__ == "__main__":
    n = 5

    # Let's compute the target entropy for a given exploitation probability.
    # For instance, if p is obtained from some process, we can calculate the target entropy.
    p_example = 0.90  # Example exploitation probability
    target_entropy = target_entropy_from_exploitation_probability_binary(p_example)

    print("For n = {} actions and exploitation probability p = {:.4f}:".format(n, p_example))
    print("SAC target entropy = {:.4f}".format(target_entropy))
