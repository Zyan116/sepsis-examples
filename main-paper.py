"""This file contains code necessary for replicating all results in the article "Methods of Off-policy Policy Evaluation Under Unobserved Confounding",
where most of codes for constructing MDPs, producing behaviour and evaluation policies, and counterfactual evaluation 
in the second example are borrowed from https://github.com/clinicalml/gumbel-max-scm/.
"""
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import block_diag
import itertools
from sepsisSimDiabetes.State import State
from sepsisSimDiabetes.DataGenerator import DataGenerator
import generator_mdp as DGEN
import conf_est as CEST
import std_est as SEST
import loss_minimization as LB
from utils import plot_design_sensitivity
from utils import MatrixMDP
from utils import format_dgen_samps
from utils import eval_on_policy
from utils import BatchSampler
from utils import plot_trajectory


NUM_OBS_STATES = 720
NUM_HID_STATES = 2  # Binary value of diabetes
NUM_PROJ_OBS_STATES = int(720 / 5)  # Marginalizing over glucose
NUM_FULL_STATES = int(NUM_OBS_STATES * NUM_HID_STATES)
NUM_ACTIONS_TOTAL = 8


#### The First Example

### Loading data
with open('Data/value_function.pkl', 'rb') as f:
    value_function = pickle.load(f)
with open('Data/t0_policy.pkl', 'rb') as f:
    t0_policy = pickle.load(f).transpose([0, 2, 1])
with open('Data/optimal_policy_st.pkl', 'rb') as f:
    optimal_policy_st = pickle.load(f).transpose([1, 0])
with open('Data/tx_tr.pkl', 'rb') as f:
    tx, tr = pickle.load(f)
with open('Data/mixed_policy.pkl', 'rb') as f:
    mixed_policy = pickle.load(f).transpose([1, 0])

### Evaluation policies
with_antibiotics = (t0_policy[0, :, :], optimal_policy_st)
without_antibiotics = (t0_policy[1, :, :], optimal_policy_st)
optimal_policy = (optimal_policy_st, optimal_policy_st)

### Standard OPE methods under unobserved confounding
## PDIS and IS estimates
config = {'Gamma': 2.0, 'num_itrs': 5000, 'max_horizon': 5, 'discount': 0.99,'confounding_threshold': 0.75
          , 'nS': NUM_FULL_STATES + 1, 'nA': NUM_ACTIONS_TOTAL, 'p_diabetes': 0.2, 'bootstrap': False}
dgen = DGEN.data_generator(transitions=(tx, tr), 
            policies=(mixed_policy, t0_policy), 
            value_fn=value_function, config=config)
trajectories, returns, _, _ = dgen.simulate_pd(config['num_itrs'], use_tqdm=True)

IS = CEST.conf_est(trajectories=trajectories, returns=returns.sum(axis = -1), k=1, config=config)
is_wo = IS.compute(without_antibiotics)
is_w = IS.compute(with_antibiotics)
is_op = IS.compute(optimal_policy)
PDIS = CEST.conf_est(trajectories=trajectories, returns=returns, k=1, config=config)
pdis_wo = PDIS.compute_pd(without_antibiotics)
pdis_w = PDIS.compute_pd(with_antibiotics)
pdis_op = PDIS.compute_pd(optimal_policy)

print('IS and PDIS estimates of performance of WO are identical:', is_wo == pdis_wo)
print('IS and PDIS estimates of performance of W are identical:', is_w == pdis_w)
print('IS and PDIS estimates of performance of OP are identical:', is_op == pdis_op)

## Performance of standard OPE method (PDIS)
config = {'Gamma': 2.0, 'num_itrs': 5000, 'max_horizon': 5, 'discount': 0.99,'confounding_threshold': 0.75
          , 'nS': NUM_FULL_STATES + 1, 'nA': NUM_ACTIONS_TOTAL, 'p_diabetes': 0.2, 'n_bootstrap': 100, 'bootstrap': True}
rep = 200
est_mat = np.zeros((rep, 3 * 3))
dgen = DGEN.data_generator(transitions=(tx, tr), 
            policies=(mixed_policy, t0_policy), 
            value_fn=value_function, config=config)
for i in range(rep):  
    trajectories, returns, iter_ids, iter_ius = dgen.simulate_pd(config['num_itrs'], use_tqdm=True)
    
    # confounded
    pdis = CEST.conf_est(trajectories=trajectories, returns=returns, k=1, config=config)
    pdis_w = pdis.compute_pd(with_antibiotics)
    pdis_wo = pdis.compute_pd(without_antibiotics)
    pdis_op = pdis.compute_pd(optimal_policy)
    
    est_mat[i, 0] = pdis_wo.mean()
    est_mat[i, 1] = pdis_w.mean()
    est_mat[i, 2] = pdis_op.mean()
    
    # unconofunded
    spdis = SEST.std_est(trajectories=trajectories, returns=returns, config=config
                         , iter_ids = iter_ids, iter_ius = iter_ius, k = 1)
    spdis_w = spdis.compute_pd(with_antibiotics)
    spdis_wo = spdis.compute_pd(without_antibiotics)
    spdis_op = spdis.compute_pd(optimal_policy)
    
    est_mat[i, 3] = spdis_wo.mean()
    est_mat[i, 4] = spdis_w.mean()
    est_mat[i, 5] = spdis_op.mean()
    
    # true aveage reward
    dgen_w = DGEN.data_generator(transitions=(tx, tr), 
            policies=(optimal_policy_st, t0_policy[0,...]), 
            value_fn=value_function, config=config)
    _, ret_w = dgen_w.simulate_target(config['num_itrs'], use_tqdm=True)
    
    dgen_wo = DGEN.data_generator(transitions=(tx, tr), 
                policies=(optimal_policy_st, t0_policy[1,...]), 
                value_fn=value_function, config=config)
    _, ret_wo = dgen_wo.simulate_target(config['num_itrs'], use_tqdm=True)
    
    dgen_op = DGEN.data_generator(transitions=(tx, tr), 
                policies=(optimal_policy_st, optimal_policy_st), 
                value_fn=value_function, config=config)
    _, ret_op = dgen_op.simulate_target(config['num_itrs'], use_tqdm=True)
    
    est_mat[i, 6] = np.mean(ret_wo.sum(axis = -1))
    est_mat[i, 7] = np.mean(ret_w.sum(axis = -1))
    est_mat[i, 8] = np.mean(ret_op.sum(axis = -1))
    
# mean and 95% CI
def cal_ci(sample, ci = 0.05): # function calculating CI
    n = len(sample)
    sample = sorted(sample)
    quantile = int(n * ci)
    return sample[quantile], sample[-quantile]
    
# table 1
table1 = pd.DataFrame(columns = ['OPE method', 'Policy', 'Mean', 'CI'])
methods = ['conf', 'unconf', 'true']
pols = ['WO', 'W', 'OP']
for i in range(len(methods)):
    for j in range(len(pols)):
        table1 = table1.append({'OPE method': methods[i], 'Policy': pols[j]
                                , 'Mean': est_mat[:,3*i+j].mean(), 'CI': cal_ci(est_mat[:,3*i+j])}, ignore_index=True)    
 
## Performance of standard OPE method (PDIS) as H increases
Ts = [5, 7, 10, 15, 20]
rep = 50
mat_pdis_hor = np.zeros((len(Ts), 3*rep))
mat_spdis_hor = np.zeros((len(Ts), 3*rep))
mat_true_hor = np.zeros((len(Ts), 3*rep))
for i in range(rep):
    for it in range(len(Ts)):
        config = {'Gamma': 2.0, 'num_itrs': 4000, 'max_horizon': Ts[it], 'discount': 0.99
                  ,'confounding_threshold': 0.75, 'nS': NUM_FULL_STATES + 1, 'nA': NUM_ACTIONS_TOTAL
                  , 'p_diabetes': 0.2, 'n_bootstrap': 100, 'bootstrap': True}
        dgen = DGEN.data_generator(transitions=(tx, tr), 
                    policies=(mixed_policy, t0_policy), 
                    value_fn=value_function, config=config)
        trajectories, returns, iter_ids, iter_ius = dgen.simulate_pd(config['num_itrs'], use_tqdm=True)
        
        # confounded
        pdis = CEST.conf_est(trajectories=trajectories, returns=returns, k=1, config=config)
        pdis_w = pdis.compute_pd(with_antibiotics)
        pdis_wo = pdis.compute_pd(without_antibiotics)
        pdis_op = pdis.compute_pd(optimal_policy)
        
        mat_pdis_hor[it, i] = pdis_wo.mean()
        mat_pdis_hor[it, i+rep] = pdis_w.mean()
        mat_pdis_hor[it, i+2*rep] = pdis_op.mean()
        
        # unconfounded
        spdis = SEST.std_est(trajectories=trajectories, returns=returns, config=config
                             , iter_ids = iter_ids, iter_ius = iter_ius, k = 1)
        spdis_w = spdis.compute_pd(with_antibiotics)
        spdis_wo = spdis.compute_pd(without_antibiotics)
        spdis_op = spdis.compute_pd(optimal_policy)
        
        mat_spdis_hor[it, i] = spdis_wo.mean()
        mat_spdis_hor[it, i+rep] = spdis_w.mean()
        mat_spdis_hor[it, i+2*rep] = spdis_op.mean()
        
        # true average reward
        dgen_w = DGEN.data_generator(transitions=(tx, tr), 
            policies=(optimal_policy_st, t0_policy[0,...]), 
            value_fn=value_function, config=config)
        _, ret_w = dgen_w.simulate_target(config['num_itrs'], use_tqdm=True)
        
        dgen_wo = DGEN.data_generator(transitions=(tx, tr), 
                    policies=(optimal_policy_st, t0_policy[1,...]), 
                    value_fn=value_function, config=config)
        _, ret_wo = dgen_wo.simulate_target(config['num_itrs'], use_tqdm=True)
        
        dgen_op = DGEN.data_generator(transitions=(tx, tr), 
                    policies=(optimal_policy_st, optimal_policy_st), 
                    value_fn=value_function, config=config)
        _, ret_op = dgen_op.simulate_target(config['num_itrs'], use_tqdm=True)
        
        mat_true_hor[it, i] = np.mean(ret_wo.sum(axis = -1))
        mat_true_hor[it, i+rep] = np.mean(ret_w.sum(axis = -1))
        mat_true_hor[it, i+2*rep] = np.mean(ret_op.sum(axis = -1))
        
# summary (contains information in table 2)
sum_hor = pd.DataFrame(columns = ['OPE method', 'Policy', 'T', 'mean', 'var'])
for it in range(len(Ts)):
    for i in range(3):
        vals = mat_spdis_hor[it, (i*rep):((i+1)*rep)]
        sum_hor = sum_hor.append({'OPE method': 'unconf', 'Policy': pols[i], 'T':Ts[it]
                                  , 'mean':vals.mean(), 'var':vals.var()}, ignore_index=True)
for it in range(len(Ts)):
    for i in range(3):
        vals = mat_pdis_hor[it, (i*rep):((i+1)*rep)]
        sum_hor = sum_hor.append({'OPE method': 'conf', 'Policy': pols[i], 'T':Ts[it]
                                  , 'mean':vals.mean(), 'var':vals.var()}, ignore_index=True)
for it in range(len(Ts)):
    for i in range(3):
        vals = mat_true_hor[it, (i*rep):((i+1)*rep)]
        sum_hor = sum_hor.append({'OPE method': 'true', 'Policy': pols[i], 'T':Ts[it]
                                  , 'mean':vals.mean(), 'var':vals.var()}, ignore_index=True)
        
# figure 2
plt.plot(Ts, sum_hor[(sum_hor['OPE method'] == 'true') & (sum_hor['Policy'] == 'WO')]['mean']
         , label = 'WO - true', color = 'green', marker = '*', linestyle = '', markersize = 10.0)
plt.plot(Ts, sum_hor[(sum_hor['OPE method'] == 'unconf') & (sum_hor['Policy'] == 'WO')]['mean']
         , label = 'WO - unconf',color = 'green', marker = '+', linestyle = '--')
plt.plot(Ts, sum_hor[(sum_hor['OPE method'] == 'conf') & (sum_hor['Policy'] == 'WO')]['mean']
         , label = 'WO - conf',color = 'green', marker = 's', linestyle = '-.')
plt.plot(Ts, sum_hor[(sum_hor['OPE method'] == 'true') & (sum_hor['Policy'] == 'WO')]['mean']
         , label = 'W - true', color = 'blue', marker = '*', linestyle = '', markersize = 10.0)
plt.plot(Ts, sum_hor[(sum_hor['OPE method'] == 'unconf') & (sum_hor['Policy'] == 'W')]['mean']
         , label = 'W - unconf',color = 'blue', marker = '+', linestyle = '--')
plt.plot(Ts, sum_hor[(sum_hor['OPE method'] == 'conf') & (sum_hor['Policy'] == 'W')]['mean']
         , label = 'W - conf',color = 'blue', marker = 's', linestyle = '-.')
plt.plot(Ts, sum_hor[(sum_hor['OPE method'] == 'true') & (sum_hor['Policy'] == 'WO')]['mean']
         , label = 'OP - true', color = 'red', marker = '*', linestyle = '', markersize = 10.0)
plt.plot(Ts, sum_hor[(sum_hor['OPE method'] == 'unconf') & (sum_hor['Policy'] == 'OP')]['mean']
         , label = 'OP - unconf',color = 'red', marker = '+', linestyle = '--')
plt.plot(Ts, sum_hor[(sum_hor['OPE method'] == 'conf') & (sum_hor['Policy'] == 'OP')]['mean']
         , label = 'OP - conf',color = 'red', marker = 's', linestyle = '-.')
plt.title('Estimated reward of evaluation policies using PDIS as H increases')
plt.xlabel('H')
plt.ylabel('Avergae reward')
plt.ylim(-0.03, 0.17)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

## Performance of standard OPE method (PDIS) as Gamma increases
Gammas = [2, 4, 8, 12]
rep = 50
mat_pdis_gam = np.zeros((len(Gammas), 3*rep))
for i in range(rep):
    for ig in range(len(Gammas)):
        config = {'Gamma': Gammas[ig], 'num_itrs': 4000, 'max_horizon': 5, 'discount': 0.99
                  ,'confounding_threshold': 0.75, 'nS': NUM_FULL_STATES + 1, 'nA': NUM_ACTIONS_TOTAL
                  , 'p_diabetes': 0.2, 'n_bootstrap': 100, 'bootstrap': True}
        dgen = DGEN.data_generator(transitions=(tx, tr), 
                    policies=(mixed_policy, t0_policy), 
                    value_fn=value_function, config=config)
        trajectories, returns, iter_ids, iter_ius = dgen.simulate_pd(config['num_itrs'], use_tqdm=True)
        
        # confounded
        pdis = CEST.conf_est(trajectories=trajectories, returns=returns, k=1, config=config)
        pdis_w = pdis.compute_pd(with_antibiotics)
        pdis_wo = pdis.compute_pd(without_antibiotics)
        pdis_op = pdis.compute_pd(optimal_policy)
        
        mat_pdis_gam[ig, i] = pdis_wo.mean()
        mat_pdis_gam[ig, i+rep] = pdis_w.mean()
        mat_pdis_gam[ig, i+2*rep] = pdis_op.mean()
        
# summary (contains information in table 3)
sum_gam = pd.DataFrame(columns = ['OPE method', 'Policy', 'Gamma', 'mean', 'var'])
for ig in range(len(Gammas)):
    for i in range(3):
        vals = mat_pdis_gam[ig, (i*rep):((i+1)*rep)]
        sum_gam = sum_gam.append({'OPE method': 'conf', 'Policy': pols[i], 'T':Gammas[ig]
                                  , 'mean':vals.mean(), 'var':vals.var()}, ignore_index=True)

# figure 3
plt.plot(Gammas, table1[(table1['OPE method'] == 'true') & (table1['Policy'] == 'WO')]['Mean'] * np.ones(len(Gammas))
         , label = 'WO - true', color = 'green', marker = '*', linestyle = '--', markersize = 10.0)
plt.plot(Gammas, sum_gam[(sum_gam['OPE method'] == 'conf') & (sum_gam['Policy'] == 'WO')]['mean']
         , label = 'WO - conf', color = 'green', marker = 's', linestyle = '-.')
plt.plot(Gammas, table1[(table1['OPE method'] == 'true') & (table1['Policy'] == 'W')]['Mean'] * np.ones(len(Gammas))
         , label = 'W - true', color = 'blue', marker = '*', linestyle = '--', markersize = 10.0)
plt.plot(Gammas, sum_gam[(sum_gam['OPE method'] == 'conf') & (sum_gam['Policy'] == 'W')]['mean']
         , label = 'W - conf', color = 'blue', marker = 's', linestyle = '-.')
plt.plot(Gammas, table1[(table1['OPE method'] == 'true') & (table1['Policy'] == 'WO')]['Mean'] * np.ones(len(Gammas))
         , label = 'OP - true', color = 'red', marker = '*', linestyle = '--', markersize = 10.0)
plt.plot(Gammas, sum_gam[(sum_gam['OPE method'] == 'conf') & (sum_gam['Policy'] == 'OP')]['mean']
         , label = 'OP - conf', color = 'red', marker = 's', linestyle = '-.')
plt.title('Estimated reward of evaluation policies using PDIS as $\Gamma$ increases')
plt.xlabel('$\Gamma$')
plt.ylabel('Avergae reward')
plt.ylim(-0.03, 0.30)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

### Bounds of performance of an evaluation policy
## Bounds of performance of target policies with two different approaches (naive and LM-based)
config = {'Gamma': 2.0, 'num_itrs': 5000, 'max_horizon': 5, 'discount': 0.99,'confounding_threshold': 0.75
          , 'nS': NUM_FULL_STATES + 1, 'nA': NUM_ACTIONS_TOTAL, 'p_diabetes': 0.2, 'n_bootstrap': 100, 'bootstrap': True}
rep = 100
lb_w = np.zeros((rep, 4)) # four columns are: LM-based lower bounds, LM-based upper bounds
                          # , naive lower bounds, naive upper bounds
lb_wo = np.zeros((rep, 4))
lb_op = np.zeros((rep, 4))
dgen = DGEN.data_generator(transitions=(tx, tr), 
            policies=(mixed_policy, t0_policy), 
            value_fn=value_function, config=config)

config_lb = {'Gamma': 2.0, 'lr':5 * 1e-2, 'epoch':80, 'nS': NUM_FULL_STATES + 1
             , 'nA': NUM_ACTIONS_TOTAL, 'bootstrap': True, 'n_bootstrap': 25}
for i in range(rep):
    trajectories, returns, _, _ = dgen.simulate(config['num_itrs'], use_tqdm=True)
    
    lb_data = {'samps': trajectories, 'returns': returns}

    w_lb = LB.loss_minimization(config=config_lb, data=lb_data, 
                                       evaluation_policies=with_antibiotics, scope='with_anti_biotics')
    wo_lb = LB.loss_minimization(config=config_lb, data=lb_data, 
                                       evaluation_policies=without_antibiotics, scope='without_anti_biotics')
    op_lb = LB.loss_minimization(config=config_lb, data=lb_data, 
                                       evaluation_policies=optimal_policy, scope='optimal')
    
    w_lb_ours, w_lb_naive, _ = w_lb.run(use_tqdm=True)
    wo_lb_ours, wo_lb_naive, _ = wo_lb.run(use_tqdm=True)
    op_lb_ours, op_lb_naive, _ = op_lb.run(use_tqdm=True)
    
    lb_w[i, 0], lb_w[i, 2] = w_lb_ours.mean(), w_lb_naive.mean()
    lb_wo[i, 0], lb_wo[i, 2] = wo_lb_ours.mean(), wo_lb_naive.mean()
    lb_op[i, 0], lb_op[i, 2] = op_lb_ours.mean(), op_lb_naive.mean()
    
    w_lb_ours, w_lb_naive, _ = w_lb.run(use_tqdm=True, upper_bound=True)
    wo_lb_ours, wo_lb_naive, _ = wo_lb.run(use_tqdm=True, upper_bound=True)
    op_lb_ours, op_lb_naive, _ = op_lb.run(use_tqdm=True, upper_bound=True)
    
    lb_w[i, 1], lb_w[i, 3] = w_lb_ours.mean(), w_lb_naive.mean()
    lb_wo[i, 1], lb_wo[i, 3] = wo_lb_ours.mean(), wo_lb_naive.mean()
    lb_op[i, 1], lb_op[i, 3] = op_lb_ours.mean(), op_lb_naive.mean()
    
# figure 4
df_lm_low = pd.DataFrame(columns = ['Policy', 'Bounds', 'Outcome'])
for i in range(rep):
    df_lm_low = df_lm_low.append({'Policy': 'WO', 'Bounds': 'LM-based', 'Outcome': lb_wo[i, 0]}, ignore_index=True)
    df_lm_low = df_lm_low.append({'Policy': 'WO', 'Bounds': 'Naive', 'Outcome': lb_wo[i, 2]}, ignore_index=True)
    df_lm_low = df_lm_low.append({'Policy': 'W', 'Bounds': 'LM-based', 'Outcome': lb_w[i, 0]}, ignore_index=True)
    df_lm_low = df_lm_low.append({'Policy': 'W', 'Bounds': 'Naive', 'Outcome': lb_w[i, 2]}, ignore_index=True)
    df_lm_low = df_lm_low.append({'Policy': 'OP', 'Bounds': 'LM-based', 'Outcome': lb_op[i, 0]}, ignore_index=True)
    df_lm_low = df_lm_low.append({'Policy': 'OP', 'Bounds': 'Naive', 'Outcome': lb_op[i, 2]}, ignore_index=True)
    
df_lm_up = pd.DataFrame(columns = ['Policy', 'Bounds', 'Outcome'])
for i in range(rep):
    df_lm_up = df_lm_up.append({'Policy': 'WO', 'Bounds': 'LM-based', 'Outcome': lb_wo[i, 1]}, ignore_index=True)
    df_lm_up = df_lm_up.append({'Policy': 'WO', 'Bounds': 'Naive', 'Outcome': lb_wo[i, 3]}, ignore_index=True)
    df_lm_up = df_lm_up.append({'Policy': 'W', 'Bounds': 'LM-based', 'Outcome': lb_w[i, 1]}, ignore_index=True)
    df_lm_up = df_lm_up.append({'Policy': 'W', 'Bounds': 'Naive', 'Outcome': lb_w[i, 3]}, ignore_index=True)
    df_lm_up = df_lm_up.append({'Policy': 'OP', 'Bounds': 'LM-based', 'Outcome': lb_op[i, 1]}, ignore_index=True)
    df_lm_up = df_lm_up.append({'Policy': 'OP', 'Bounds': 'Naive', 'Outcome': lb_op[i, 3]}, ignore_index=True)

pal = sns.color_palette("Set2")
fig, ax = plt.subplots()
sns.boxplot(x = 'Policy', y = 'Outcome', hue = 'Bounds', data = df_lm_low, palette=pal, fliersize=0)
sns.boxplot(x = 'Policy', y = 'Outcome', hue = 'Bounds', data = df_lm_up, palette=pal, fliersize=0)
plt.axhline(y = table1[(table1['OPE method'] == 'true') & (table1['Policy'] == 'WO')]['Mean']
            , xmin=0.06, xmax=0.26, color = pal[3], linestyle = '--', label = 'true value')
plt.axhline(y = table1[(table1['OPE method'] == 'true') & (table1['Policy'] == 'W')]['Mean']
            , xmin=0.4, xmax=0.6, color = pal[3], linestyle = '--')
plt.axhline(y = table1[(table1['OPE method'] == 'true') & (table1['Policy'] == 'OP')]['Mean']
            , xmin=0.75, xmax=0.95, color = pal[3], linestyle = '--')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles[0:3], labels[0:3],
               loc='upper left',
               fontsize='large',
               handletextpad=0.5)


## Minimum size of samples for LM_based bounds to be effective
sizes = np.array([100, 200, 300, 400, 500]).astype(int)
lb_w_size = np.zeros((len(sizes), 2)) 
lb_wo_size = np.zeros((len(sizes), 2))

config = {'Gamma': 2.0, 'num_itrs': 500, 'max_horizon': 5, 'discount': 0.99,'confounding_threshold': 0.75
          , 'nS': NUM_FULL_STATES + 1, 'nA': NUM_ACTIONS_TOTAL, 'p_diabetes': 0.2, 'n_bootstrap': 100, 'bootstrap': True}
dgen = DGEN.data_generator(transitions=(tx, tr), 
            policies=(mixed_policy, t0_policy), 
            value_fn=value_function, config=config)
trajectories, returns, _, _ = dgen.simulate(config['num_itrs'], use_tqdm=True)

for i in range(len(sizes)):
    n = sizes[i]
    idxs = np.random.choice(np.arange(5000), size = n, replace = False)
    traj, ret = trajectories[idxs, :], returns[idxs]
    ## caculating bounds
    lm_config = {'Gamma': 2.0, 'lr':5 * 1e-2, 'epoch':80, 'nS': NUM_FULL_STATES + 1
                 , 'nA': NUM_ACTIONS_TOTAL, 'bootstrap': True, 'n_bootstrap': 25}
    lb_data = {'samps': traj, 'returns': ret}
    
    w_lb = LB.loss_minimization(config=lm_config, data=lb_data, 
                                       evaluation_policies=with_antibiotics, scope='with_anti_biotics')
    wo_lb = LB.loss_minimization(config=lm_config, data=lb_data, 
                                       evaluation_policies=without_antibiotics, scope='without_anti_biotics')
    
    w_lb_ours, _, _ = w_lb.run(use_tqdm=True)
    wo_lb_ours, _, _ = wo_lb.run(use_tqdm=True)

    
    lb_w_size[i, 0] = w_lb_ours.mean()
    lb_wo_size[i, 0] = wo_lb_ours.mean()
    
    w_lb_ours, _, _ = w_lb.run(use_tqdm=True, upper_bound=True)
    wo_lb_ours, _, _ = wo_lb.run(use_tqdm=True, upper_bound=True)
    
    lb_w_size[i, 1] = w_lb_ours.mean()
    lb_wo_size[i, 1] = wo_lb_ours.mean()
    
# figure 5
plt.plot(sizes, lb_w_size[:, 0], color = 'blue', label = 'W')
plt.plot(sizes, lb_w_size[:, 1], color = 'blue')
plt.plot(sizes, lb_wo_size[:, 0], color = 'green', label = 'WO')
plt.plot(sizes, lb_wo_size[:, 1], color = 'green')
plt.xlabel('Sample Size')
plt.ylabel('Outcome')
plt.title('Performance of LM-based bounds with different sample sizes')
plt.legend()

## Design sensitivity analysis
Gamma_Star = 2.0 # repeat the same analysis for Gamma_Star = 3.0, 4.0, 5.0, 6.0, 7.0
config = {'Gamma': Gamma_Star, 'num_itrs': 5000, 'max_horizon': 5, 'discount': 0.99
          ,'confounding_threshold': 0.75, 'nS': NUM_FULL_STATES + 1, 'nA': NUM_ACTIONS_TOTAL
          , 'p_diabetes': 0.2, 'n_bootstrap': 100, 'bootstrap': True}
dgen = DGEN.data_generator(transitions=(tx, tr), 
            policies=(mixed_policy, t0_policy), 
            value_fn=value_function, config=config)
trajectories, returns, _, _ = dgen.simulate(config['num_itrs'], use_tqdm=True)

loss_minimization_config = {'Gamma': 1.0, 'lr':5 * 1e-2, 'epoch':150, 
        'nS': NUM_FULL_STATES + 1, 'nA': NUM_ACTIONS_TOTAL, 
        'bootstrap': True, 'n_bootstrap': 50}
lb_data = {'samps': trajectories, 'returns': returns}

GAMMAs = np.arange(1.0, 7.0, 0.5)
with_results2 = np.zeros((4, len(GAMMAs), loss_minimization_config['n_bootstrap']))
without_results2 = np.zeros((4, len(GAMMAs), loss_minimization_config['n_bootstrap']))

with_anti_lb = LB.loss_minimization(config=loss_minimization_config, data=lb_data
                                    , evaluation_policies=with_antibiotics, scope='with_anti_biotics')
without_anti_lb = LB.loss_minimization(config=loss_minimization_config, data=lb_data
                                       , evaluation_policies=without_antibiotics, scope='without_anti_biotics')
i = 0
for Gamma in tqdm(GAMMAs):
    with_anti_lb.update_gamma(Gamma)
    without_anti_lb.update_gamma(Gamma)
    # Lowerbounds
    with_results2[0, i, :], with_results2[1, i, :], _ = with_anti_lb.run()
    without_results2[0, i, :], without_results2[1, i, :], _ = without_anti_lb.run()

    # Upperbounds
    with_results2[2, i, :], with_results2[3, i, :], _ = with_anti_lb.run(upper_bound=True)
    without_results2[2, i, :], without_results2[3, i, :], _ = without_anti_lb.run(upper_bound=True)
    i += 1
    
# figure 6
naive_results = {'GAMMAs': GAMMAs, 'pol1_lower':with_results2[1, :, :].mean(axis=-1)
                 ,'pol2_lower':without_results2[1, :, :].mean(axis=-1),'pol1_upper':with_results2[3, :, :].mean(axis=-1)
                 ,'pol2_upper':without_results2[3, :, :].mean(axis=-1)
                 ,'pol1_label': 'with antibiotics', 'pol2_label': 'without antibiotics','cross': 1.39}

plt.figure(figsize=(7,5))
plot_design_sensitivity(naive_results, title='Design Sensitivity, Naive', yticks=[-0.1, 0, 0.1, 0.2, 0.3, 0.4]
                        ,  xlim=[0.95, 2.5], ylim=[-0.01, 0.18], fontsize = 15)
plt.xticks([1.0, 1.5, 2.0, 2.5, 3.0])

ours_results = {'GAMMAs': GAMMAs, 'pol1_lower':with_results2[0, :, :].mean(axis=-1)
                ,'pol2_lower':without_results2[0, :, :].mean(axis=-1),'pol1_upper':with_results2[2, :, :].mean(axis=-1)
                ,'pol2_upper':without_results2[2, :, :].mean(axis=-1),'pol1_label': 'with antibiotics'
                , 'pol2_label': 'without antibiotics','cross': 2.45}

plt.figure(figsize=(7,5))
plot_design_sensitivity(ours_results, title='Design Sensitivity, LM-based', yticks=[-0.1, 0, 0.1, 0.2, 0.3, 0.4]
                        ,  xlim=[0.95, 2.5], ylim=[-0.01, 0.18], fontsize = 15)
plt.xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

# code for figure 7
Gamma_Star = 6.0
config = {'Gamma': Gamma_Star, 'num_itrs': 5000, 'max_horizon': 5, 'discount': 0.99
          ,'confounding_threshold': 0.75, 'nS': NUM_FULL_STATES + 1, 'nA': NUM_ACTIONS_TOTAL
          , 'p_diabetes': 0.2, 'n_bootstrap': 100, 'bootstrap': True}
dgen = DGEN.data_generator(transitions=(tx, tr), policies=(mixed_policy, t0_policy), value_fn=value_function, config=config)
trajectories, returns, _, _ = dgen.simulate(config['num_itrs'], use_tqdm=True)

loss_minimization_config = {'Gamma': 1.0, 'lr':5 * 1e-2, 'epoch':150
                            , 'nS': NUM_FULL_STATES + 1, 'nA': NUM_ACTIONS_TOTAL
                            , 'bootstrap': True, 'n_bootstrap': 50}
lb_data = {'samps': trajectories, 'returns': returns}

GAMMAs = np.arange(1.0, 9.0, 0.5)
with_results6 = np.zeros((2, len(GAMMAs), loss_minimization_config['n_bootstrap']))
without_results6 = np.zeros((2, len(GAMMAs), loss_minimization_config['n_bootstrap']))

with_anti_lb = LB.loss_minimization(config=loss_minimization_config, data=lb_data
                                    , evaluation_policies=with_antibiotics, scope='with_anti_biotics')
without_anti_lb = LB.loss_minimization(config=loss_minimization_config, data=lb_data
                                       , evaluation_policies=without_antibiotics, scope='without_anti_biotics')
i = 0
for Gamma in tqdm(GAMMAs):
    with_anti_lb.update_gamma(Gamma)
    without_anti_lb.update_gamma(Gamma)
    # Lowerbounds
    with_results6[0, i, :], _, _ = with_anti_lb.run()
    without_results6[0, i, :], _, _ = without_anti_lb.run()

    # Upperbounds
    with_results6[1, i, :], _, _ = with_anti_lb.run(upper_bound=True)
    without_results6[1, i, :], _, _ = without_anti_lb.run(upper_bound=True)
    i += 1
    
ours_results = {'GAMMAs': GAMMAs, 'pol1_lower':with_results6[0, :, :].mean(axis=-1)
                ,'pol2_lower':without_results6[0, :, :].mean(axis=-1),'pol1_upper':with_results6[1, :, :].mean(axis=-1)
                ,'pol2_upper':without_results6[1, :, :].mean(axis=-1),'pol1_label': 'with antibiotics'
                , 'pol2_label': 'without antibiotics','cross': 6.45}

plt.figure(figsize=(7,5))
plot_design_sensitivity(ours_results, title='Design Sensitivity, LM-based'
                        , yticks=[-0.1, 0, 0.1, 0.2, 0.3, 0.4],  xlim=[0.95, 8.5], ylim=[-0.1, 0.4], fontsize = 15)
plt.xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


#### The Second Example

## MDPs costruction and data generation
# get the transition and reward matrix from file
with open("Data/diab_txr_mats-replication.pkl", "rb") as f:
    mdict = pickle.load(f)

tx_mat = mdict["tx_mat"]
r_mat = mdict["r_mat"]
p_mixture = np.array([1 - 0.2, 0.2]) # probabilities of being non-diabetic and diabetic

tx_mat_full = np.zeros((NUM_ACTIONS_TOTAL, NUM_FULL_STATES, NUM_FULL_STATES))
r_mat_full = np.zeros((NUM_ACTIONS_TOTAL, NUM_FULL_STATES, NUM_FULL_STATES))

for a in range(NUM_ACTIONS_TOTAL):
    tx_mat_full[a, ...] = block_diag(tx_mat[0, a, ...], tx_mat[1, a,...])
    r_mat_full[a, ...] = block_diag(r_mat[0, a, ...], r_mat[1, a, ...])
    
# obtain behaviour policy by policy iteration on full MDP    
fullMDP = MatrixMDP(tx_mat_full, r_mat_full)
fullPol = fullMDP.policyIteration(discount=0.99, eval_type=1)

physPolSoft = np.copy(fullPol)
physPolSoft[physPolSoft == 1] = 1 - 0.05        # add randoness
physPolSoft[physPolSoft == 0] = 0.05 / (NUM_ACTIONS_TOTAL - 1)

# projection from full state space to reduced state space
n_states_abs = NUM_OBS_STATES + 2
discStateIdx = n_states_abs - 1
deadStateIdx = n_states_abs - 2
n_proj_states = NUM_PROJ_OBS_STATES + 2
proj_matrix = np.zeros((n_states_abs, n_proj_states))
for i in range(n_states_abs - 2):
    this_state = State(state_idx = i, idx_type='obs', 
                       diabetic_idx = 1)  
    j = this_state.get_state_idx('proj_obs')
    proj_matrix[i, j] = 1

# add the projection to death and discharge
proj_matrix[deadStateIdx, -2] = 1
proj_matrix[discStateIdx, -1] = 1
proj_matrix = proj_matrix.astype(int)
proj_lookup = proj_matrix.argmax(axis=-1)

# generate trajectories with full state space
dgen = DataGenerator()
states, actions, rewards, diab, emp_tx_totals, emp_r_totals = dgen.simulate(1000, 20, policy=physPolSoft
                                                                            , policy_idx_type='full', p_diabetes=0.2, use_tqdm=False) 
obs_samps = format_dgen_samps(states, actions, rewards, diab, 20, 1000)

# absorbing states
death_states = (emp_r_totals.sum(axis=0).sum(axis=0) < 0)
disch_states = (emp_r_totals.sum(axis=0).sum(axis=0) > 0)
death_states = np.concatenate([death_states, np.array([True, False])])
disch_states = np.concatenate([disch_states, np.array([False, True])])

# construct transition matrix for reduced state space
est_tx_abs = np.zeros((NUM_ACTIONS_TOTAL, n_states_abs, n_states_abs))
est_tx_abs[:, :-2, :-2] = emp_tx_totals
est_tx_abs[:, death_states, deadStateIdx] = 1
est_tx_abs[:, disch_states, discStateIdx] = 1

proj_tx = np.zeros((NUM_ACTIONS_TOTAL, n_proj_states, n_proj_states))
proj_tx_mat = np.zeros_like(proj_tx)
for a in range(NUM_ACTIONS_TOTAL):
    proj_tx[a] = proj_matrix.T.dot(est_tx_abs[a]).dot(proj_matrix)
nonzero_idx = proj_tx.sum(axis=-1) != 0    # normalization
proj_tx_mat[nonzero_idx] = proj_tx[nonzero_idx]
proj_tx_mat[nonzero_idx] /= proj_tx_mat[nonzero_idx].sum(axis=-1, keepdims=True)    

# construct reward matrix for reduced state space
proj_r_mat = np.zeros((NUM_ACTIONS_TOTAL, n_proj_states, n_proj_states))
proj_r_mat[..., -2] = -1
proj_r_mat[..., -1] = 1

proj_r_mat[..., -2, -2] = 0 # No reward once in aborbing state
proj_r_mat[..., -1, -1] = 0

# construct distribution on initial states
initial_state_arr = states[:, 0, 0]
initial_state_counts = np.zeros((n_states_abs,1))
for i in range(initial_state_arr.shape[0]):
    initial_state_counts[initial_state_arr[i]] += 1

proj_state_counts = proj_matrix.T.dot(initial_state_counts).T   # Project initial state counts to new states
proj_p_initial_state = proj_state_counts / proj_state_counts.sum()

zero_sa_pairs = proj_tx_mat.sum(axis=-1) == 0   # Because some SA pairs are never observed, assume they cause instant death
proj_tx_mat[zero_sa_pairs, -2] = 1  # Always insta-death if you take a never-taken action

# MDP with reduced state space
projMDP = MatrixMDP(proj_tx_mat, proj_r_mat, p_initial_state=proj_p_initial_state)

# obtain evaluation policy by policy iteration on MDP with reduced state space
try:
    RlPol = projMDP.policyIteration(discount=0.99)
except:
    assert np.allclose(proj_tx_mat.sum(axis=-1), 1)
    RlPol = projMDP.policyIteration(discount=0.99, skip_check=True)
    
# construct observed trajectories with reduced state space
def projection_func(obs_state_idx):
    if obs_state_idx == -1:
        return -1
    else:
        return proj_lookup[obs_state_idx]
proj_f = np.vectorize(projection_func)
states_proj = proj_f(states)
obs_samps_proj = format_dgen_samps(states_proj, actions, rewards, diab, 20, 1000)

# true average reward of evaluation policy
states_rl, actions_rl, rewards_rl, diab_rl, _, _ = dgen.simulate(1000, 20, policy=RlPol[:-2, :]
                                                                 , policy_idx_type='proj_obs', p_diabetes=0.2, use_tqdm=False) 

obs_samps_rlpol = format_dgen_samps(states_rl, actions_rl, rewards_rl, diab_rl, 20, 1000)

this_true_rl_reward = eval_on_policy(obs_samps_rlpol, discount=1, bootstrap=True, n_bootstrap=100)  

# observed reward from the samples given
this_obs_reward = eval_on_policy(obs_samps_proj, discount=1, bootstrap=True, n_bootstrap=100)

## Counterfactual evaluation
# generate counterfactual trajectories with different number of cf trajectories per observation
BSampler = BatchSampler(mdp=projMDP)
cf_trajectories = {}
cf_rewards = []
n_cf = [5, 10, 15, 20, 25, 30]

for n in n_cf:
    this_cf_opt_samps_proj = BSampler.cf_trajectory(
        obs_samps_proj, 
        cf_policy=RlPol, 
        n_cf_samps=n, use_tqdm=True, tqdm_desc='CF OPE: n =  ' + str(n))

    this_cf_opt_samps_proj_reshaped = \
    this_cf_opt_samps_proj.reshape(this_cf_opt_samps_proj.shape[0] * this_cf_opt_samps_proj.shape[1]
                                   ,this_cf_opt_samps_proj.shape[2], this_cf_opt_samps_proj.shape[3])

    this_offpol_opt_reward_cf = eval_on_policy(this_cf_opt_samps_proj_reshaped, discount=1, bootstrap=True, n_bootstrap=100)
    
    cf_trajectories[str(n) + ' cf samps'] = np.copy(this_cf_opt_samps_proj)
    cf_rewards.append(this_offpol_opt_reward_cf)
    
cf_counts = np.zeros((1000, 3, len(n_cf)))
cf_votes_prob = np.zeros((1000, len(n_cf)))
for idx_n in range(len(n_cf)):
    n = n_cf[idx_n]    
    this_cf_opt_samps_proj = cf_trajectories[str(n) + ' cf samps']

    obs_vs_cf_reward = pd.DataFrame(np.zeros((1000, 2)),columns = ['Obs Reward', 'CF Reward'])

    for obs_idx in range(1000):
        # Get the MAP counterfactual reward
        cf_r_all = this_cf_opt_samps_proj[obs_idx][..., 6].sum(axis=-1)
        (rew, idx, cts) = np.unique(cf_r_all, return_index=True, return_counts=True)
        if len(idx) == 3:
            cf_counts[obs_idx,:,idx_n] = cts
        else:
            for i in range(len(rew)):
                r = int(rew[i])
                cf_counts[obs_idx, r+1, idx_n] = cts[i]
        cf_votes_prob[obs_idx, idx_n] = np.max(cts) / n
        cf_idx = idx[np.argmax(cts)]
        cf_r = cf_r_all[cf_idx]
        assert cf_r in [-1, 0, 1]    

        # Get the observed reward
        obs_r = obs_samps[obs_idx][..., 6].sum()
        assert obs_r in [-1, 0, 1]
        obs_vs_cf_reward.iloc[obs_idx] = np.array([obs_r, cf_r])
        
    result_mat = np.zeros((3, 3))

    cf_idxs = [obs_vs_cf_reward['CF Reward'] == i for i in [-1, 0, 1]]
    obs_idxs = [obs_vs_cf_reward['Obs Reward'] == i for i in [-1, 0, 1]]
 
    for i, j in itertools.product(range(3), range(3)):
        result_mat[i, j] = np.logical_and(obs_idxs[i], cf_idxs[j]).sum()

    result_mat /= result_mat.sum()
    
    # Decompsition results are almost the same no matter how many 
    # cf trajectories are generated per observation.
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 24})
    ax = sns.heatmap(result_mat, annot=True, 
                     annot_kws={'fontsize': 48},
                     fmt='.0%', cbar=False, cmap='Blues')
    ax.set_xlabel("\nCounterfactual Outcome\n" + str(n))
    ax.set_ylabel("Observed Outcome\n")
    ax.set_xticklabels(['Died', 'No Chg.', 'Disch.'])
    ax.set_yticklabels(['Died', 'No Chg.', 'Disch.'])
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_verticalalignment('center')     
    
    # 94.8% of observed individuals have highest vote not lower than 90%.
    print('Proportion of individuals with highest vote not lower than 90% when '+ str(n) + ' cf trajectories per observation:'
          , np.mean(cf_votes_prob[:, i] >= 0.9))
   
# We choose 10 as the number of cf trajectories for the rest analysis.
    
# figure 8
plt.rcParams.update({'font.size': 20})
reward_df = pd.DataFrame({'Obs': this_obs_reward, 
                          'CF': cf_rewards[1],
                          'True': this_true_rl_reward})

plt.figure(figsize=(10,5))
sns.boxplot(data=reward_df, whis=[2.5, 97.5])
plt.ylabel("Average Reward")

# figure 9
this_cf_opt_samps_proj = cf_trajectories['10 cf samps']
obs_vs_cf_reward = pd.DataFrame(np.zeros((1000, 2)),columns = ['Obs Reward', 'CF Reward'])

for obs_idx in range(1000):
    # Get the MAP counterfactual reward
    cf_r_all = this_cf_opt_samps_proj[obs_idx][..., 6].sum(axis=-1)
    (_, idx, cts) = np.unique(cf_r_all, return_index=True, return_counts=True)
    cf_idx = idx[np.argmax(cts)]
    cf_r = cf_r_all[cf_idx]
    assert cf_r in [-1, 0, 1]    

    # Get the observed reward
    obs_r = obs_samps[obs_idx][..., 6].sum()
    assert obs_r in [-1, 0, 1]
    obs_vs_cf_reward.iloc[obs_idx] = np.array([obs_r, cf_r])

result_mat = np.zeros((3, 3))

cf_idxs = [obs_vs_cf_reward['CF Reward'] == i for i in [-1, 0, 1]]
obs_idxs = [obs_vs_cf_reward['Obs Reward'] == i for i in [-1, 0, 1]]

for i, j in itertools.product(range(3), range(3)):
    result_mat[i, j] = np.logical_and(obs_idxs[i], cf_idxs[j]).sum()

result_mat /= result_mat.sum()

plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.size': 24})
ax = sns.heatmap(result_mat, annot=True, 
                 annot_kws={'fontsize': 48},
                 fmt='.0%', cbar=False, cmap='Blues')
ax.set_xlabel("\nCounterfactual Outcome\n")
ax.set_ylabel("Observed Outcome\n")
ax.set_xticklabels(['Died', 'No Chg.', 'Disch.'])
ax.set_yticklabels(['Died', 'No Chg.', 'Disch.'])
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_verticalalignment('center')
    
# analysis of pregression of observed individuals of interest
would_have_lived_idx = np.logical_and(cf_idxs[2], obs_idxs[0])
print("There are {} patients who the model believes would have been discharged".format(would_have_lived_idx.sum()))

# select individuals of interest
selected_obs_trajectories = obs_samps[would_have_lived_idx]
selected_cf_trajectories = this_cf_opt_samps_proj[would_have_lived_idx]
num_selected_traj = selected_obs_trajectories.shape[0]

# Get lengths as the time until action = -1
obs_lengths = np.ones(num_selected_traj)*20
for traj_idx in range(num_selected_traj):
    for time_idx in range(20):
        if selected_obs_trajectories[traj_idx, time_idx, 6] != 0:
            obs_lengths[traj_idx] = time_idx
            break
obs_lengths = obs_lengths.astype(int)

# This gives all of the trajectories with a reasonable length, for further inspection
np.where(obs_lengths > 10)

# figure 10
plt.rcParams.update({'font.size': 14})
# RL policy attempts no treatment from the start, mistake due to glucose
# 8, 17, 27, 30, 37, 49, 63, 73, 83, 89, 100

# RL policy thinks it could have stabilized, then discharged, but glucose would still have been off
# 2, 5, 7, 13, 19, 20, 25, 32, 44, 56, 68, 69, 74, 80, 84, 91

# RL policy thinks it could have stabilized, then discharged;  Might be correct, glucose was not abnormal
# 12, 22, 98, 

# Death cases in cfs
# 2, 7, 12, 20, 37, 44, 56, 68, 69, 74, 80, 82, 98  

traj_idx = 91 # the one used in figure 10
fig, axes = plot_trajectory(
        selected_obs_trajectories, 
        pt_idx=traj_idx, 
        cf=True, 
        cf_samps=selected_cf_trajectories, 
        cf_proj=True
        )
