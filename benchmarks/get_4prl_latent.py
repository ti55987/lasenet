import numpy as np

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
