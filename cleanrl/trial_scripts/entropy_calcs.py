import math


def compute_target_entropy(n, best_action_prob):
    """
    Compute the target entropy for a discrete distribution with n actions,
    where the best action has probability `best_action_prob` and the rest of the probability
    is uniformly distributed among the other actions.

    Parameters:
        n (int): Number of actions.
        best_action_prob (float): Probability assigned to the best action.

    Returns:
        float: The entropy in nats.
    """
    # Probability for each of the other actions.
    other_prob = (1 - best_action_prob) / (n - 1)

    # Compute entropy: H = - sum_i p_i * log(p_i)
    entropy = - (best_action_prob * math.log(best_action_prob) + (n - 1) * other_prob * math.log(other_prob))
    return entropy


if __name__ == "__main__":
    n = 4
    best_action_prob = 0.7
    entropy_nats = compute_target_entropy(n, best_action_prob)
    sac_usual = -math.log(1/n)

    # In SAC, the target entropy is typically set to the negative of the desired entropy value.
    target_entropy = -entropy_nats

    print(
        "For a discrete distribution with {} actions and a best action probability of {}:".format(n, best_action_prob))
    print("Calculated entropy (nats): {:.5f}".format(entropy_nats))
    print("Target entropy (for SAC, negative value): {:.5f}".format(target_entropy))
    print(f"Target entropy Usual: {sac_usual}")
