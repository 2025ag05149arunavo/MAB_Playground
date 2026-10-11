import numpy as np

def Greedy_simulation(arm_means, num_arms, total_steps, rng=None, **kwargs):
    """
    Simulates the epsilon-greedy algorithm over given time horizons.

    Parameters:
    -----------
    algorithm : BanditAlgorithm
        The bandit algorithm instance to store the results.
    num_arms : int
        The length of the arm_means array.
    total_steps : int
        The number of steps to simulate.
    epsilon : float
        A parameter for the choice of epsilon

    Returns:
    --------
    dict
        A dictionary where keys are time horizons and values are tuples of (total_reward, suboptimal_arms_count, total_regret, zeros_count, ones_count).
    """

    # Get the epsilon value from kwargs
    epsilon = kwargs.get('epsilon')

    # Initialize Q-values and counts for each arm
    Q = np.zeros(num_arms)
    N = np.zeros(num_arms)

    # Initialize arrays to store results
    suboptimal_arms = np.zeros(total_steps, dtype=int)
    regret = np.zeros(total_steps)
    zeros_counts = np.zeros(total_steps, dtype=int)
    ones_counts = np.zeros(total_steps, dtype=int)
    total_rewards = np.zeros(total_steps)

    # Initialize counters
    total_reward = 0
    zeros_count = 0
    ones_count = 0
    rewards = np.zeros(num_arms)
    counts = np.zeros(num_arms)
    suboptimal_arms_count = 0
    total_regret = 0

    # Simulation loop
    for t in range(1, total_steps + 1):
        if rng.random() < epsilon:
            # Exploration: choose a random arm
            arm = rng.integers(num_arms)
        else:
            # Exploitation: choose the best arm based on Q-values
            arm = np.argmax(Q)

        # Simulate pulling the chosen arm
        reward = rng.binomial(1, arm_means[arm])
        counts[arm] += 1
        rewards[arm] += reward
        total_reward += reward
        total_rewards[t - 1] = total_reward

        # Update Q-value incrementally
        N[arm] += 1
        Q[arm] += (reward - Q[arm]) / N[arm]
        
        # Update counts of zeros and ones
        if reward == 0:
            zeros_count += 1
        else:
            ones_count += 1

        zeros_counts[t - 1] = zeros_count
        ones_counts[t - 1] = ones_count

        # Check if the chosen arm is suboptimal
        if arm != np.argmax(arm_means):
            suboptimal_arms_count += 1
            total_regret += np.max(arm_means) - arm_means[arm]

        # Record regret and suboptimal arm count
        regret[t - 1] = total_regret
        suboptimal_arms[t - 1] = suboptimal_arms_count

    # Return the results as a dictionary
    return {
        "total_rewards": total_rewards,
        "suboptimal_arms": suboptimal_arms,
        "regret": regret,
        "zeros_counts": zeros_counts,
        "ones_counts": ones_counts
    }
