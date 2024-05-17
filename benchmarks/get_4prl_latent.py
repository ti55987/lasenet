import random
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize

# get_4prl_latent derives latent variables: chosen Q values given
# observable actions and rewards + recovered parameters from MLE/MAP.
def get_4prl_latent(actions, rewards, parameters):
    alpha, neg_alpha, beta, _ = parameters
    beta = beta * 10
    # two-armed bandit task
    num_actions = 2
    # equal value first
    q_values = np.array([1 / num_actions] * num_actions)
    lr_list = [neg_alpha, alpha]

    rpe_history, q_values_history = [], []
    for a, r in zip(actions, rewards):
        q_values_history.append(q_values[a])

        rpe = r - q_values[a]
        q_values[a] += lr_list[r] * rpe  # update q value
        rpe_history.append(rpe)
        # conunter-factual updating
        unchosen_rpe = (1 - r) - q_values[1 - a]
        q_values[1 - a] += lr_list[r] * unchosen_rpe

    return rpe_history, q_values_history

def prl4_neg_log_likelihood(actions, rewards, parameters):
  alpha, neg_alpha, beta, stickiness = parameters

  beta = beta*10 # why do it here?
  num_actions = 2

  lr_list = [neg_alpha, alpha]
  q_values = np.array([1/num_actions]*num_actions) # equal value first

  llh = 0
  prev_a = -1
  for a, r in zip(actions, rewards):
    Q = q_values.copy()
    if prev_a != -1:
       Q[prev_a] = Q[prev_a]+stickiness

    llh += np.log(scipy.special.softmax(beta * Q)[a])

    rpe = r - q_values[a]
    q_values[a] += lr_list[r]*rpe # update q value

    unchosen_rpe = (1-r) - q_values[1-a]
    q_values[1-a] += lr_list[r]*unchosen_rpe # update q value
    prev_a = a

  return -llh

def fit_4prl_mle(actions, rewards, model_specs):
    est_params = []
    for aid in range(actions.shape[0]):
        init_params = [random.uniform(l, h) for l, h in model_specs['bounds']]
        func = lambda x, *args: prl4_neg_log_likelihood(actions[aid], rewards[aid], x)
        res = minimize(
            func,
            init_params,
            bounds = model_specs['bounds'],
            method='L-BFGS-B',
            options={'maxiter': 30})

        est_params.append(res.x)
        print(f'evaluating {aid}...')

    print(len(est_params))
    est_params = np.array(est_params)
    recovered_param = {}
    for idx, pn in enumerate(model_specs['param_names']):
        recovered_param[pn] = est_params[:, idx]

    return pd.DataFrame(recovered_param)   