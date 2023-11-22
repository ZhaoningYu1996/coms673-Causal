import numpy as np
import math

def simulate_arm_pull(arm, arms_info):
    mean, std_dev = arms_info[arm]
    reward = np.random.normal(mean, std_dev)
    return reward

def update_empirical_estimates(arm, reward, empirical_estimates, arm_counts):
    arm_counts[arm] += 1
    old_estimate = empirical_estimates[arm]
    new_estimate = old_estimate + (reward - old_estimate) / arm_counts[arm]
    empirical_estimates[arm] = new_estimate

def calculate_ucb(empirical_estimates, arm_counts, t):
    ucb_estimates = {}
    for arm in empirical_estimates:
        ucb_estimates[arm] = empirical_estimates[arm] + math.sqrt(2 * math.log(t) / arm_counts[arm])
    return ucb_estimates

def crm_alg(arms, T, arms_info):
    N = len(arms)
    t = 2 * N + 2
    beta = 1
    arm_counts = {arm: 1 for arm in arms}
    empirical_estimates = {arm: 0 for arm in arms} 

    for t in range(2 * N + 2, T + 1):
        ucb_estimates = calculate_ucb(empirical_estimates, arm_counts, t)

        if arm_counts["a0"] < beta**2 * math.log(t):
            arm_pulled = "a0"
        else:
            arm_pulled = max(ucb_estimates, key=ucb_estimates.get)

        reward = simulate_arm_pull(arm_pulled, arms_info)
        update_empirical_estimates(arm_pulled, reward, empirical_estimates, arm_counts)

        if empirical_estimates["a0"] < max(empirical_estimates.values()):
            beta = min(2 * math.sqrt(2) / (empirical_estimates["best"] - empirical_estimates["a0"]), math.sqrt(math.log(t)))

# Example 
arms = ["a0", "a1"] 
T = 1000  # Total number of rounds
arms_info = {"a0": (mean1, std_dev1), "a1": (mean2, std_dev2)} 
crm_alg(arms, T, arms_info)
