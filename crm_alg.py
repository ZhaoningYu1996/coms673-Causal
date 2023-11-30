import numpy as np
import math
from tqdm import tqdm

def P_X1():
    return np.random.choice([0, 1], p=[0.5, 0.5])

def P_X2_X3_given_X1(X1):
    return np.random.choice([0, 1], p=[0.25 + 0.5 * X1, 0.75 - 0.5 * X1])

def P_Y_given_X2_X3(X2, X3):
    return 1 if X2 == X3 else 0

def simulate_intervention(do_X2=None, do_X3=None):
    X1 = P_X1()
    X2 = do_X2 if do_X2 is not None else P_X2_X3_given_X1(X1)
    X3 = do_X3 if do_X3 is not None else P_X2_X3_given_X1(X1)
    Y = P_Y_given_X2_X3(X2, X3)
    return Y

def compute_mu_hat_0(t, observations):
    return sum(1 for Y, a_s in observations if Y == 1 and a_s == 'a0') / len(observations) if observations else 0

def compute_mu_hat_i_x(t, arm, interventions):
    return sum(interventions[arm]) / len(interventions[arm]) if interventions[arm] else 0

def compute_mu_bar_a(t, mu_hat, N_a_t, C_i_x_t=0):
    # return mu_hat + np.sqrt(2 * np.log(t) / (N_a_t + C_i_x_t))
    if N_a_t + C_i_x_t <= 0:
        return mu_hat  # or handle the case as you see fit
    return mu_hat + np.sqrt(2 * np.log(t) / (N_a_t + C_i_x_t))

def CRM_ALG(T):
    # Initialize variables
    N = 2  # Number of arms (X2 and X3)
    beta = 1
    
    for _ in range(1):  # 30 independent runs
        observations = []  # Store (Y, a_s)
        interventions = {'X2': [], 'X3': []}  # Store outcomes for interventions
        arm_pulls = {'a0': 0, 'X2': 0, 'X3': 0}  # Number of times each arm is pulled
        cumulative_regrets = []
        regret = 0
        for t in tqdm(range(1, T + 1)):
            # Decide which arm to pull
            if arm_pulls['a0'] < beta**2 * np.log(t):
                chosen_arm = 'a0'
            else:
                # Compute UCB for each arm and choose the highest
                ucb_X2 = compute_mu_bar_a(t, compute_mu_hat_i_x(t, 'X2', interventions), arm_pulls['X2'])
                ucb_X3 = compute_mu_bar_a(t, compute_mu_hat_i_x(t, 'X3', interventions), arm_pulls['X3'])
                chosen_arm = 'X2' if ucb_X2 > ucb_X3 else 'X3'

            # Simulate intervention or observation based on chosen arm
            Y = simulate_intervention(do_X2=1 if chosen_arm == 'X2' else None,
                                      do_X3=1 if chosen_arm == 'X3' else None)

            # Update observations and interventions
            observations.append((Y, chosen_arm))
            if chosen_arm != 'a0':
                interventions[chosen_arm].append(Y)

            # Update arm pull counts
            arm_pulls[chosen_arm] += 1

            # Update beta if necessary
            mu_0 = compute_mu_hat_0(t, observations)
            mu_2 = compute_mu_hat_i_x(t, "X2", interventions)
            mu_3 = compute_mu_hat_i_x(t, "X3", interventions)
            mu_star = max(mu_2, mu_3)
            if mu_0 < mu_star:
                beta = min(2 * np.sqrt(2 / (mu_star - mu_0)), np.sqrt(np.log(t)))
            
            best_arm_outcome = 5/8
            regret += best_arm_outcome - Y

        # Calculate cumulative regret
        cumulative_regrets.append(regret)
    return np.mean(cumulative_regrets)

T = 1000  # example time range
average_cumulative_regret = CRM_ALG(T)
print(f"Average Cumulative Regret over {T} time steps: {average_cumulative_regret}")
