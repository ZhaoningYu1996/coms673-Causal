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
    return X1, X2, X3, Y

def compute_mu_hat_i_x(theta, arm, x, observations, arm_time_stamp):
    N = len(arm_time_stamp[arm])
    S = arm_time_stamp["a0"]
    S_even = [S[i]-1 for i in range(len(S)) if i % 2 == 1]  # Even indices
    S_odd = [S[i]-1 for i in range(len(S)) if i % 2 == 0] 

    if arm in ["X1_0", "X1_1"]:
        return theta
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
            return theta
        
        num = int(len(S_even)/C)

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
    
        return ((N+1) * theta + Y_i_x) / (N + C + 1)
    
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
            return theta
        
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
        
        # return ((N+1) * theta + Y_i_x) / (N + C + 1)
        return (theta + Y_i_x / C)/2

def initialize(observations, intervention, arm_time_stamp, ):
    # X1 = 0
    observations["pulled_arm"].append("X1_0")
    arm_time_stamp["X1_0"].append(1)
    X1, X2, X3, Y = simulate_intervention(do_X1=0)
    observations["X1"].append(X1)
    observations["X2"].append(X2)
    observations["X3"].append(X3)
    observations["Y"].append(Y)
    
    # X1 = 1
    observations["pulled_arm"].append("X1_1")
    arm_time_stamp["X1_1"].append(2)
    X1, X2, X3, Y = simulate_intervention(do_X1=1)
    observations["X1"].append(X1)
    observations["X2"].append(X2)
    observations["X3"].append(X3)
    observations["Y"].append(Y)

    # X2 = 0
    observations["pulled_arm"].append("X2_0")
    arm_time_stamp["X2_0"].append(3)
    X1, X2, X3, Y, reward = simulate_intervention(do_X2=0)
    observations["X1"].append(X1)
    observations["X2"].append(X2)
    observations["X3"].append(X3)
    observations["Y"].append(Y)
    
    # X2 = 1
    observations["pulled_arm"].append("X2_1")
    arm_time_stamp["X2_1"].append(4)
    X1, X2, X3, Y = simulate_intervention(do_X2=1)
    observations["X1"].append(X1)
    observations["X2"].append(X2)
    observations["X3"].append(X3)
    observations["Y"].append(Y)

    # X3 = 0
    observations["pulled_arm"].append("X3_0")
    arm_time_stamp["X3_0"].append(5)
    X1, X2, X3, Y, reward = simulate_intervention(do_X3=0)
    observations["X1"].append(X1)
    observations["X2"].append(X2)
    observations["X3"].append(X3)
    observations["Y"].append(Y)
    
    # X3 = 1
    observations["pulled_arm"].append("X3_1")
    arm_time_stamp["X3_1"].append(6)
    X1, X2, X3, Y, reward = simulate_intervention(do_X3=1)
    observations["X1"].append(X1)
    observations["X2"].append(X2)
    observations["X3"].append(X3)
    observations["Y"].append(Y)

    # a0
    observations["pulled_arm"].append("a0")
    arm_time_stamp["a0"].append(7)
    X1, X2, X3, Y, reward = simulate_intervention()
    observations["X1"].append(X1)
    observations["X2"].append(X2)
    observations["X3"].append(X3)
    observations["Y"].append(Y)
    
    return observations, arm_time_stamp
            
def causal_ts(T, T0):
    
    all_regret_list = []
    for _ in range(30):
    
        S_list = {"X1_0": 1, "X1_1": 1, "X2_0": 1, "X2_1": 1, "X3_0": 1, "X3_1": 1, "a0": 1}
        F_list = {"X1_0": 1, "X1_1": 1, "X2_0": 1, "X2_1": 1, "X3_0": 1, "X3_1": 1, "a0": 1}
        
        observations = {"X1": [], "X2": [], "X3": [], "a0": [], "pulled_arm": [], "Y": []}
        
        arm_time_stamp = {"a0": [], "X1_0": [], "X1_1": [], "X2_0": [], "X2_1": [], "X3_0": [], "X3_1": []}
        
        regret_list = []
        
        regret = 0
        
        	
        # rng = np.random.default_rng()
        
        for t in tqdm(range(T)):
            if t <= T0:
                chosen_arm = 'a0'
                observations["pulled_arm"].append(chosen_arm)
                arm_time_stamp[chosen_arm].append(t)
                X1, X2, X3, Y = simulate_intervention()
                observations["X1"].append(X1)
                observations["X2"].append(X2)
                observations["X3"].append(X3)
                observations["Y"].append(Y)
                if Y == 1:
                    S_list["a0"] += 1
                else:
                    F_list["a0"] += 1
            else:
                # print(stop)
                # theta_X1_0 = rng.beta(S_list["X1_0"], F_list["X1_0"])
                # theta_X1_1 = rng.beta(S_list["X1_1"], F_list["X1_1"])
                # theta_X2_0 = rng.beta(S_list["X2_0"], F_list["X2_0"])
                # theta_X2_1 = rng.beta(S_list["X2_1"], F_list["X2_1"])
                # theta_X3_0 = rng.beta(S_list["X3_0"], F_list["X3_0"])
                # theta_X3_1 = rng.beta(S_list["X3_1"], F_list["X3_1"])
                # theta_a0 = rng.beta(S_list["a0"], F_list["a0"])
                theta_X1_0 = np.random.beta(S_list["X1_0"], F_list["X1_0"])
                theta_X1_1 = np.random.beta(S_list["X1_1"], F_list["X1_1"])
                theta_X2_0 = np.random.beta(S_list["X2_0"], F_list["X2_0"])
                theta_X2_1 = np.random.beta(S_list["X2_1"], F_list["X2_1"])
                theta_X3_0 = np.random.beta(S_list["X3_0"], F_list["X3_0"])
                theta_X3_1 = np.random.beta(S_list["X3_1"], F_list["X3_1"])
                theta_a0 = np.random.beta(S_list["a0"], F_list["a0"])
                mu_hat_1_0 = compute_mu_hat_i_x(theta_X1_0, 'X1_0', 0, observations, arm_time_stamp)
                mu_hat_1_1 = compute_mu_hat_i_x(theta_X1_1, 'X1_1', 1, observations, arm_time_stamp)
                mu_hat_2_0 = compute_mu_hat_i_x(theta_X2_0, 'X2_0', 0, observations, arm_time_stamp)
                mu_hat_2_1 = compute_mu_hat_i_x(theta_X2_1, 'X2_1', 1, observations, arm_time_stamp)
                mu_hat_3_0 = compute_mu_hat_i_x(theta_X3_0, 'X3_0', 0, observations, arm_time_stamp)
                mu_hat_3_1 = compute_mu_hat_i_x(theta_X3_1, 'X3_1', 1, observations, arm_time_stamp)
                mu_hat_a0 = theta_a0
                ts_list = [mu_hat_1_0, mu_hat_1_1, mu_hat_2_0, mu_hat_2_1, mu_hat_3_0, mu_hat_3_1, mu_hat_a0]
                
                # print(ts_list)
                
                if ts_list.index(max(ts_list)) == 0:
                    chosen_arm = "X1_0"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y = simulate_intervention(do_X1=0)
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
                    if Y == 1:
                        S_list["X1_0"] += 1
                    else:
                        F_list["X1_0"] += 1
                elif ts_list.index(max(ts_list)) == 1:
                    chosen_arm = "X1_1"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y = simulate_intervention(do_X1=1)
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
                    if Y == 1:
                        S_list["X1_1"] += 1
                    else:
                        F_list["X1_1"] += 1
                elif ts_list.index(max(ts_list)) == 2:
                    chosen_arm = "X2_0"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y = simulate_intervention(do_X2=0)
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
                    if Y == 1:
                        S_list["X2_0"] += 1
                    else:
                        F_list["X2_0"] += 1
                elif ts_list.index(max(ts_list)) == 3:
                    chosen_arm = "X2_1"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y = simulate_intervention(do_X2=1)
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
                    if Y == 1:
                        S_list["X2_1"] += 1
                    else:
                        F_list["X2_1"] += 1
                elif ts_list.index(max(ts_list)) == 4:
                    chosen_arm = "X3_0"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y = simulate_intervention(do_X3=0)
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
                    if Y == 1:
                        S_list["X3_0"] += 1
                    else:
                        F_list["X3_0"] += 1
                elif ts_list.index(max(ts_list)) == 5:
                    chosen_arm = "X3_1"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y = simulate_intervention(do_X3=1)
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
                    if Y == 1:
                        S_list["X3_1"] += 1
                    else:
                        F_list["X3_1"] += 1
                elif ts_list.index(max(ts_list)) == 6:
                    chosen_arm = "a0"
                    observations["pulled_arm"].append(chosen_arm)
                    arm_time_stamp[chosen_arm].append(t)
                    X1, X2, X3, Y = simulate_intervention()
                    observations["X1"].append(X1)
                    observations["X2"].append(X2)
                    observations["X3"].append(X3)
                    observations["Y"].append(Y)
                    if Y == 1:
                        S_list["a0"] += 1
                    else:
                        F_list["a0"] += 1
            # print("-------->")
            # print(S_list)
            # print(F_list)
            # if t > T0:
            #     print(ts_list)
            best_arm_outcome = 5/8
            regret += best_arm_outcome - Y
            # print("--->")
            # print(Y)
            regret_list.append(regret)
        all_regret_list.append(regret_list)
    
    all_regret = np.array(all_regret_list)

    return np.mean(all_regret, axis=0)

T = 10000  # example time range
T0 = 200
nodes = ["X1", "X2", "X3", "a0"]
cumulative_regret = causal_ts(T, T0)
torch.save(cumulative_regret, "ts_10000_200.pt")
print(f"Average Cumulative Regret over {T} time steps: {cumulative_regret[-1]}")
import matplotlib.pyplot as plt
x_values = list(range(len(cumulative_regret)))
plt.plot(x_values, cumulative_regret)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('TS_10000_200.png')
plt.show()
