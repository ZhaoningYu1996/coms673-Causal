import numpy as np
import math
from tqdm import tqdm
import random
import torch

def P_X1():
    return np.random.choice([0, 1], p=[0.5, 0.5])

def P_X2_X3_given_X1(X1):
    return np.random.choice([0, 1], p=[0.75 - 0.5 * X1, 0.25 + 0.5 * X1])

def P_Y_given_X2_X3(X2, X3):
    return 1 if X2 == X3 else 0

def simulate_intervention(do_X1=None, do_X2=None, do_X3=None):
    X1 = do_X1 if do_X1 is not None else P_X1()
    X2 = do_X2 if do_X2 is not None else P_X2_X3_given_X1(X1)
    X3 = do_X3 if do_X3 is not None else P_X2_X3_given_X1(X1)
    Y = P_Y_given_X2_X3(X2, X3)
    # if do_X1 is None and do_X2 is None and do_X3 is None:
    #     regret = 1/2
    # elif do_X1 is not None:
    #     regret = 5/8
    # elif do_X2 is not None or do_X3 is not None:
    #     regret = 1/4
    regret = 0
        
    return X1, X2, X3, Y, regret

def compute_mu_hat_0(observations, arm_time_stamp):
    if not observations:
        return 0
    count = 0
    # print(arm_time_stamp)
    # print(observations)
    for i in arm_time_stamp["a0"]:
        if observations["Y"][i-1] == 1:
            count += 1
    return count/len(arm_time_stamp["a0"]), len(arm_time_stamp["a0"])
    # return sum(1 for Y, a_s in observations if Y == 1 and a_s == 'a0') / len(observations) if observations else 0

def compute_mu_hat_i_x(arm, x, observations, arm_time_stamp):
    N = len(arm_time_stamp[arm])
    S = arm_time_stamp["a0"]
    S_even = [S[i]-1 for i in range(len(S)) if i % 2 == 1]  # Even indices
    S_odd = [S[i]-1 for i in range(len(S)) if i % 2 == 0] 

    if arm in ["X1_0", "X1_1"]:
        C = 0
        count = 0
        for i in arm_time_stamp[arm]:
            if observations["Y"][i-1] == 1:
                count += 1
        return count / N, N, C
    if arm in ["X2_0", "X2_1"]:
        S_z_0 = []
        S_z_1 = []
        for i in S_odd:
            if observations["X2"][i] == x and observations["X1"][i] == 0:
                S_z_0.append(i)
            if observations["X2"][i] == x and observations["X1"][i] == 1:
                S_z_1.append(i)

        if len(S_z_0) > len(S_z_1):
            C = len(S_z_1)
            S_z_0 = random.sample(S_z_0, C)
        else:
            C = len(S_z_0)
            S_z_1 = random.sample(S_z_1, C)

        if C == 0:
            count = 0
            for i in arm_time_stamp[arm]:
                if observations["Y"][i-1] == 1:
                    count += 1
            return count / N, N, C
        
        num = int(len(S_even)/C)

        # if C == 1:
        #     p_t_c_0 = 0
        #     p_t_c_1 = 0
        #     S_t_c_length = num
        #     sub_S_even = S_even
        #     for i in sub_S_even:
        #         if observations["X1"][i] == 0:
        #             p_t_c_0 += 1
        #         if observations["X1"][i] == 1:
        #             p_t_c_1 += 1
        #     Y_i_x = observations["Y"][S_z_0[0]]*p_t_c_0 + observations["Y"][S_z_1[0]]*p_t_c_1
        # else:
        Y_i_x = 0
        for i in range(C-1):
            p_t_c_0 = 0
            p_t_c_1 = 0
            sub_S_even = S_even[i*num:(i+1)*num]
            for j in sub_S_even:
                if observations["X1"][j] == 0:
                    p_t_c_0 += 1
                if observations["X1"][j] == 1:
                    p_t_c_1 += 1
            Y_c_i_x = observations["Y"][S_z_0[i]]*p_t_c_0/num + observations["Y"][S_z_1[i]]*p_t_c_1/num
            Y_i_x += Y_c_i_x
        
        p_t_c_0 = 0
        p_t_c_1 = 0
        sub_S_even = S_even[(C-1)*num:]
        for j in sub_S_even:
            if observations["X1"][j] == 0:
                p_t_c_0 += 1
            if observations["X1"][j] == 1:
                p_t_c_1 += 1
        Y_c_i_x = observations["Y"][S_z_0[-1]]*p_t_c_0/(len(S_even)-(C-1)*num) + observations["Y"][S_z_1[-1]]*p_t_c_1/(len(S_even)-(C-1)*num)
        Y_i_x += Y_c_i_x

        count = 0
        for i in arm_time_stamp[arm]:
            if observations["Y"][i-1] == 1:
                count += 1
    
        return (count + Y_i_x) / (N + C), N, C
    
    if arm in ["X3_0", "X3_1"]:
        S_z_0 = []
        S_z_1 = []
        for i in S_odd:
            if observations["X3"][i] == x and observations["X1"][i] == 0:
                S_z_0.append(i)
            if observations["X3"][i] == x and observations["X1"][i] == 1:
                S_z_1.append(i)

        if len(S_z_0) > len(S_z_1):
            C = len(S_z_1)
            S_z_0 = random.sample(S_z_0, C)
        else:
            C = len(S_z_0)
            S_z_1 = random.sample(S_z_1, C)
        
        if C == 0:
            count = 0
            for i in arm_time_stamp[arm]:
                if observations["Y"][i-1] == 1:
                    count += 1
            return count / N, N, 0
        
        num = int(len(S_even)/C)
        
        Y_i_x = 0
        for i in range(C-1):
            p_t_c_0 = 0
            p_t_c_1 = 0
            S_t_c_length = num
            sub_S_even = S_even[i*num:(i+1)*num]
            for j in sub_S_even:
                if observations["X1"][j] == 0:
                    p_t_c_0 += 1
                if observations["X1"][j] == 1:
                    p_t_c_1 += 1
            Y_c_i_x = observations["Y"][S_z_0[i]]*p_t_c_0/num + observations["Y"][S_z_1[i]]*p_t_c_1/num
            Y_i_x += Y_c_i_x
        p_t_c_0 = 0
        p_t_c_1 = 0
        S_t_c_length = num
        sub_S_even = S_even[(C-1)*num:]
        for j in sub_S_even:
            if observations["X1"][j] == 0:
                p_t_c_0 += 1
            if observations["X1"][j] == 1:
                p_t_c_1 += 1
        Y_c_i_x = observations["Y"][S_z_0[-1]]*p_t_c_0/(len(S_even)-(C-1)*num) + observations["Y"][S_z_1[-1]]*p_t_c_1/(len(S_even)-(C-1)*num)
        Y_i_x += Y_c_i_x
        
        count = 0
        for i in arm_time_stamp[arm]:
            if observations["Y"][i-1] == 1:
                count += 1
        
        return (count + Y_i_x) / (N + C), N, C

def compute_mu_bar_a(t, mu_hat, N_a_t, C_i_x_t):
    # return mu_hat + np.sqrt(2 * np.log(t) / (N_a_t + C_i_x_t))
    if N_a_t + C_i_x_t <= 0:
        return mu_hat  # or handle the case as you see fit
    return mu_hat + np.sqrt(2 * np.log(t) / (N_a_t + C_i_x_t))

def initialize(observations, nodes, arm_time_stamp, ):
    
    for node in nodes:
        if node == "X1":
            observations["pulled_arm"].append("X1_0")
            arm_time_stamp["X1_0"].append(2*nodes.index(node)+1)
            X1, X2, X3, Y, reward = simulate_intervention(do_X1=0)
            observations["X1"].append(X1)
            observations["X2"].append(X2)
            observations["X3"].append(X3)
            observations["Y"].append(Y)
            observations["pulled_arm"].append("X1_1")
            arm_time_stamp["X1_1"].append(2*nodes.index(node)+2)
            X1, X2, X3, Y, reward = simulate_intervention(do_X1=1)
            observations["X1"].append(X1)
            observations["X2"].append(X2)
            observations["X3"].append(X3)
            observations["Y"].append(Y)
        if node == "X2":
            observations["pulled_arm"].append("X2_0")
            arm_time_stamp["X2_0"].append(2*nodes.index(node)+1)
            X1, X2, X3, Y, reward = simulate_intervention(do_X2=0)
            observations["X1"].append(X1)
            observations["X2"].append(X2)
            observations["X3"].append(X3)
            observations["Y"].append(Y)
            observations["pulled_arm"].append("X2_1")
            arm_time_stamp["X2_1"].append(2*nodes.index(node)+2)
            X1, X2, X3, Y, reward = simulate_intervention(do_X2=1)
            observations["X1"].append(X1)
            observations["X2"].append(X2)
            observations["X3"].append(X3)
            observations["Y"].append(Y)
        if node == "X3":
            observations["pulled_arm"].append("X3_0")
            arm_time_stamp["X3_0"].append(2*nodes.index(node)+1)
            X1, X2, X3, Y, reward = simulate_intervention(do_X3=0)
            observations["X1"].append(X1)
            observations["X2"].append(X2)
            observations["X3"].append(X3)
            observations["Y"].append(Y)
            observations["pulled_arm"].append("X3_1")
            arm_time_stamp["X3_1"].append(2*nodes.index(node)+2)
            X1, X2, X3, Y, reward = simulate_intervention(do_X3=1)
            observations["X1"].append(X1)
            observations["X2"].append(X2)
            observations["X3"].append(X3)
            observations["Y"].append(Y)
        if node == "a0":
            observations["pulled_arm"].append("a0")
            arm_time_stamp[node].append(2*nodes.index(node)+1)
            X1, X2, X3, Y, reward = simulate_intervention()
            observations["X1"].append(X1)
            observations["X2"].append(X2)
            observations["X3"].append(X3)
            observations["Y"].append(Y)
    
    return observations, arm_time_stamp
            
# def CRM_ALG(T, nodes):
#     # Initialize variables
    
#     all_regret_list = []
#     for _ in range(2):  # 30 independent runs
#         # N = 2  # Number of arms (X2 and X3)
#         beta = 1
#         observations = {x: [] for x in nodes}  # Store observations 0 or 1
#         observations["pulled_arm"] = []
#         observations["Y"] = []
#         # arm_time_stamp = {"a0": [], "X1": [], "X2": [], "X3": []}
#         arm_time_stamp = {"a0": [], "X1_0": [], "X1_1": [], "X2_0": [], "X2_1": [], "X3_0": [], "X3_1": []}
        
#         observations, arm_time_stamp = initialize(observations, nodes, arm_time_stamp) # Pull each arm once
#         # print(stop)
        
#         # interventions = {'X2': [], 'X3': []}  # Store outcomes for interventions
#         arm_pulls = {'a0': 0, 'X1_0': 0, 'X1_1': 0, 'X2_0': 0, 'X2_1': 0, 'X3_0': 0, 'X3_1': 0}  # Number of times each arm is pulled
        
#         cumulative_regrets = []
#         regret = 0
#         regret_list = []
#         for t in tqdm(range(8, 8+T)):
#             # Decide which arm to pull
#             if arm_pulls['a0'] < beta**2 * np.log(t):
#                 chosen_arm = 'a0'
#                 observations["pulled_arm"].append(chosen_arm)
#                 arm_time_stamp[chosen_arm].append(t)
#                 X1, X2, X3, Y, reward = simulate_intervention()
#                 observations["X1"].append(X1)
#                 observations["X2"].append(X2)
#                 observations["X3"].append(X3)
#                 observations["Y"].append(Y)
#                 mu_hat_a0, N_a0 = compute_mu_hat_0(observations, arm_time_stamp)
#                 ucb_a0 = compute_mu_bar_a(t, mu_hat_a0, N_a0, 0)

#             else:
#                 # Compute UCB for each arm and choose the highest
#                 mu_hat_1_0, N_1_0, C_1_0 = compute_mu_hat_i_x('X1_0', 0, observations, arm_time_stamp)
#                 mu_hat_1_1, N_1_1, C_1_1 = compute_mu_hat_i_x('X1_1', 1, observations, arm_time_stamp)
#                 mu_hat_2_0, N_2_0, C_2_0 = compute_mu_hat_i_x('X2_0', 0, observations, arm_time_stamp)
#                 mu_hat_2_1, N_2_1, C_2_1 = compute_mu_hat_i_x('X2_1', 1, observations, arm_time_stamp)
#                 mu_hat_3_0, N_3_0, C_3_0 = compute_mu_hat_i_x('X3_0', 0, observations, arm_time_stamp)
#                 mu_hat_3_1, N_3_1, C_3_1 = compute_mu_hat_i_x('X3_1', 1, observations, arm_time_stamp)
#                 mu_hat_a0, N_a0 = compute_mu_hat_0(observations, arm_time_stamp)
#                 ucb_X1_0 = compute_mu_bar_a(t, mu_hat_1_0, N_1_0, C_1_0)
#                 ucb_X1_1 = compute_mu_bar_a(t, mu_hat_1_1, N_1_1, C_1_1)
#                 ucb_X2_0 = compute_mu_bar_a(t, mu_hat_2_0, N_2_0, C_2_0)
#                 ucb_X2_1 = compute_mu_bar_a(t, mu_hat_2_1, N_2_1, C_2_1)
#                 ucb_X3_0 = compute_mu_bar_a(t, mu_hat_3_0, N_3_0, C_3_0)
#                 ucb_X3_1 = compute_mu_bar_a(t, mu_hat_3_1, N_3_1, C_3_1)
#                 ucb_a0 = compute_mu_bar_a(t, mu_hat_a0, N_a0, 0)
#                 ucb_list = [ucb_X1_0, ucb_X1_1, ucb_X2_0, ucb_X2_1, ucb_X3_0, ucb_X3_1, ucb_a0]

#                 if ucb_list.index(max(ucb_list)) == 0:
#                     chosen_arm = "X1_0"
#                     observations["pulled_arm"].append(chosen_arm)
#                     arm_time_stamp[chosen_arm].append(t)
#                     X1, X2, X3, Y, reward = simulate_intervention(do_X1=0)
#                     observations["X1"].append(X1)
#                     observations["X2"].append(X2)
#                     observations["X3"].append(X3)
#                     observations["Y"].append(Y)
#                 elif ucb_list.index(max(ucb_list)) == 1:
#                     chosen_arm = "X1_1"
#                     observations["pulled_arm"].append(chosen_arm)
#                     arm_time_stamp[chosen_arm].append(t)
#                     X1, X2, X3, Y, reward = simulate_intervention(do_X1=1)
#                     observations["X1"].append(X1)
#                     observations["X2"].append(X2)
#                     observations["X3"].append(X3)
#                     observations["Y"].append(Y)
#                 elif ucb_list.index(max(ucb_list)) == 2:
#                     chosen_arm = "X2_0"
#                     observations["pulled_arm"].append(chosen_arm)
#                     arm_time_stamp[chosen_arm].append(t)
#                     X1, X2, X3, Y, reward = simulate_intervention(do_X2=0)
#                     observations["X1"].append(X1)
#                     observations["X2"].append(X2)
#                     observations["X3"].append(X3)
#                     observations["Y"].append(Y)
#                 elif ucb_list.index(max(ucb_list)) == 3:
#                     chosen_arm = "X2_1"
#                     observations["pulled_arm"].append(chosen_arm)
#                     arm_time_stamp[chosen_arm].append(t)
#                     X1, X2, X3, Y, reward = simulate_intervention(do_X2=1)
#                     observations["X1"].append(X1)
#                     observations["X2"].append(X2)
#                     observations["X3"].append(X3)
#                     observations["Y"].append(Y)
#                 elif ucb_list.index(max(ucb_list)) == 4:
#                     chosen_arm = "X3_0"
#                     observations["pulled_arm"].append(chosen_arm)
#                     arm_time_stamp[chosen_arm].append(t)
#                     X1, X2, X3, Y, reward = simulate_intervention(do_X3=0)
#                     observations["X1"].append(X1)
#                     observations["X2"].append(X2)
#                     observations["X3"].append(X3)
#                     observations["Y"].append(Y)
#                 elif ucb_list.index(max(ucb_list)) == 5:
#                     chosen_arm = "X3_1"
#                     observations["pulled_arm"].append(chosen_arm)
#                     arm_time_stamp[chosen_arm].append(t)
#                     X1, X2, X3, Y, reward = simulate_intervention(do_X3=1)
#                     observations["X1"].append(X1)
#                     observations["X2"].append(X2)
#                     observations["X3"].append(X3)
#                     observations["Y"].append(Y)
#                 elif ucb_list.index(max(ucb_list)) == 6:
#                     chosen_arm = "a0"
#                     observations["pulled_arm"].append(chosen_arm)
#                     arm_time_stamp[chosen_arm].append(t)
#                     X1, X2, X3, Y, reward = simulate_intervention()
#                     observations["X1"].append(X1)
#                     observations["X2"].append(X2)
#                     observations["X3"].append(X3)
#                     observations["Y"].append(Y)
            
#             # Update arm pull counts
#             arm_pulls[chosen_arm] += 1

#             # Update beta if necessary
#             mu_hat_1_0, N_1_0, C_1_0 = compute_mu_hat_i_x('X1_0', 0, observations, arm_time_stamp)
#             mu_hat_1_1, N_1_1, C_1_1 = compute_mu_hat_i_x('X1_1', 1, observations, arm_time_stamp)
#             mu_hat_2_0, N_2_0, C_2_0 = compute_mu_hat_i_x('X2_0', 0, observations, arm_time_stamp)
#             mu_hat_2_1, N_2_1, C_2_1 = compute_mu_hat_i_x('X2_1', 1, observations, arm_time_stamp)
#             mu_hat_3_0, N_3_0, C_3_0 = compute_mu_hat_i_x('X3_0', 0, observations, arm_time_stamp)
#             mu_hat_3_1, N_3_1, C_3_1 = compute_mu_hat_i_x('X3_1', 1, observations, arm_time_stamp)
#             mu_hat_a0, N_a0 = compute_mu_hat_0(observations, arm_time_stamp)
#             mu_hat_list = [mu_hat_1_0, mu_hat_1_1, mu_hat_2_0, mu_hat_2_1, mu_hat_3_0, mu_hat_3_1, mu_hat_a0]
#             # print(mu_hat_list)
#             # print(stop)
#             if mu_hat_a0 < max(mu_hat_list):
#                 beta = min(2 * np.sqrt(2 / (max(mu_hat_list) - mu_hat_a0)), np.sqrt(np.log(t)))
            
#             best_arm_outcome = 5/8
#             regret += best_arm_outcome - Y
#             regret_list.append(regret)

#         # Calculate cumulative regret
#         # cumulative_regrets.append(regret)
#         all_regret_list.append(regret_list)
    
#     all_regret = np.array(all_regret_list)
#     print(all_regret.shape)
#     # return np.mean(cumulative_regrets)
#     return np.mean(all_regret, axis=0)

def CRM_ALG(T, nodes):
    # Initialize variables
    
    all_regret_list = []
    for _ in range(30):  # 30 independent runs
        # N = 2  # Number of arms (X2 and X3)
        beta = 1
        observations = {x: [] for x in nodes}  # Store observations 0 or 1
        observations["pulled_arm"] = []
        observations["Y"] = []
        # arm_time_stamp = {"a0": [], "X1": [], "X2": [], "X3": []}
        arm_time_stamp = {"a0": [], "X1_0": [], "X1_1": [], "X2_0": [], "X2_1": [], "X3_0": [], "X3_1": []}
        
        observations, arm_time_stamp = initialize(observations, nodes, arm_time_stamp) # Pull each arm once
        # print(stop)
        
        # interventions = {'X2': [], 'X3': []}  # Store outcomes for interventions
        # arm_pulls = {'a0': 0, 'X1_0': 0, 'X1_1': 0, 'X2_0': 0, 'X2_1': 0, 'X3_0': 0, 'X3_1': 0}  # Number of times each arm is pulled
        
        regret = 0
        regret_list = []
        mu_hat_list = []
        ucb_list = []
        for t in tqdm(range(8, T)):
            # Decide which arm to pull
            if len(arm_time_stamp["a0"]) < beta**2 * np.log(t):
                chosen_arm = 'a0'
                observations["pulled_arm"].append(chosen_arm)
                arm_time_stamp[chosen_arm].append(t)
                X1, X2, X3, Y, reward = simulate_intervention()
                observations["X1"].append(X1)
                observations["X2"].append(X2)
                observations["X3"].append(X3)
                observations["Y"].append(Y)
                #mu_hat_a0, N_a0 = compute_mu_hat_0(observations, arm_time_stamp)
                #ucb_a0 = compute_mu_bar_a(t, mu_hat_a0, N_a0, 0)

            else:
                if ucb_list.index(max(ucb_list)) == 0:
                    chosen_arm = "X1_0"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y, reward = simulate_intervention(do_X1=0)
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
                elif ucb_list.index(max(ucb_list)) == 1:
                    chosen_arm = "X1_1"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y, reward = simulate_intervention(do_X1=1)
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
                elif ucb_list.index(max(ucb_list)) == 2:
                    chosen_arm = "X2_0"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y, reward = simulate_intervention(do_X2=0)
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
                elif ucb_list.index(max(ucb_list)) == 3:
                    chosen_arm = "X2_1"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y, reward = simulate_intervention(do_X2=1)
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
                elif ucb_list.index(max(ucb_list)) == 4:
                    chosen_arm = "X3_0"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y, reward = simulate_intervention(do_X3=0)
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
                elif ucb_list.index(max(ucb_list)) == 5:
                    chosen_arm = "X3_1"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y, reward = simulate_intervention(do_X3=1)
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
                elif ucb_list.index(max(ucb_list)) == 6:
                    chosen_arm = "a0"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y, reward = simulate_intervention()
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
            
            # Update arm pull counts
            # arm_pulls[chosen_arm] += 1
            
            mu_hat_1_0, N_1_0, C_1_0 = compute_mu_hat_i_x('X1_0', 0, observations, arm_time_stamp)
            mu_hat_1_1, N_1_1, C_1_1 = compute_mu_hat_i_x('X1_1', 1, observations, arm_time_stamp)
            mu_hat_2_0, N_2_0, C_2_0 = compute_mu_hat_i_x('X2_0', 0, observations, arm_time_stamp)
            mu_hat_2_1, N_2_1, C_2_1 = compute_mu_hat_i_x('X2_1', 1, observations, arm_time_stamp)
            mu_hat_3_0, N_3_0, C_3_0 = compute_mu_hat_i_x('X3_0', 0, observations, arm_time_stamp)
            mu_hat_3_1, N_3_1, C_3_1 = compute_mu_hat_i_x('X3_1', 1, observations, arm_time_stamp)
            mu_hat_a0, N_a0 = compute_mu_hat_0(observations, arm_time_stamp)
            mu_hat_list = [mu_hat_1_0, mu_hat_1_1, mu_hat_2_0, mu_hat_2_1, mu_hat_3_0, mu_hat_3_1, mu_hat_a0]
            ucb_X1_0 = compute_mu_bar_a(t, mu_hat_1_0, N_1_0, C_1_0)
            ucb_X1_1 = compute_mu_bar_a(t, mu_hat_1_1, N_1_1, C_1_1)
            ucb_X2_0 = compute_mu_bar_a(t, mu_hat_2_0, N_2_0, C_2_0)
            ucb_X2_1 = compute_mu_bar_a(t, mu_hat_2_1, N_2_1, C_2_1)
            ucb_X3_0 = compute_mu_bar_a(t, mu_hat_3_0, N_3_0, C_3_0)
            ucb_X3_1 = compute_mu_bar_a(t, mu_hat_3_1, N_3_1, C_3_1)
            ucb_a0 = compute_mu_bar_a(t, mu_hat_a0, N_a0, 0)
            ucb_list = [ucb_X1_0, ucb_X1_1, ucb_X2_0, ucb_X2_1, ucb_X3_0, ucb_X3_1, ucb_a0]

            # print(mu_hat_list)
            # print(stop)
            if mu_hat_a0 < max(mu_hat_list):
                beta = min(2 * np.sqrt(2) / (max(mu_hat_list) - mu_hat_a0), np.sqrt(np.log(t)))
            
            best_arm_outcome = 5/8
            regret += best_arm_outcome - Y
            regret_list.append(regret)
        
        # print(arm_pulls)
        # print(mu_hat_list)

        # Calculate cumulative regret
        # cumulative_regrets.append(regret)
        print(regret_list[-1])
        all_regret_list.append(regret_list)
    
    all_regret = np.array(all_regret_list)
    print(all_regret.shape)
    # print(observations)
    # return np.mean(cumulative_regrets)
    return np.mean(all_regret, axis=0)

T = 50000  # example time range
nodes = ["X1", "X2", "X3", "a0"]
cumulative_regret = CRM_ALG(T, nodes)
torch.save(cumulative_regret, "crm_5000.pt")
print(f"Average Cumulative Regret over {T} time steps: {cumulative_regret[-1]}")
import matplotlib.pyplot as plt
x_values = list(range(len(cumulative_regret)))
plt.plot(x_values, cumulative_regret)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('crm_50000.png')
plt.show()
