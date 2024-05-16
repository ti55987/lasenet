import numpy as np
import pandas as pd

import ssm

DEFAULT_THREE_STATE_TX = np.array(
    [[[0.98, 0.01, 0.01], [0.04, 0.94, 0.02], [0.04, 0.01, 0.96]]]
)
DEFAULT_FOUR_STATE_TX = np.array(
    [
        [
            [0.97, 0.01, 0.01, 0.01],
            [0.04, 0.92, 0.02, 0.02],
            [0.04, 0.01, 0.94, 0.01],
            [0.04, 0.01, 0.01, 0.94],
        ]
    ]
)

def simulate_glmhmm(
    num_agent, num_trials_per_agent, num_states, trans_mat=DEFAULT_THREE_STATE_TX, weights_sigma=1,
):
    obs_dim = 1  # number of observed dimensions
    num_categories = 2  # number of categories for output
    input_dim = 4  # input dimensions

    # Make a 2-input driven GLM-HMM
    true_glmhmm = ssm.HMM(
        num_states,
        obs_dim,
        input_dim,
        observations="input_driven_obs",
        observation_kwargs=dict(C=num_categories),
        transitions="standard",
    )

    if num_states == 4:
        trans_mat = DEFAULT_FOUR_STATE_TX
    true_glmhmm.transitions.params = np.log(trans_mat)

    inpts = np.ones((num_agent, num_trials_per_agent, input_dim))  # initialize inpts array
    stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]
    # generate random sequence of stimuli
    inpts[:, :, 0] = data = np.random.choice(
        stim_vals,
        (num_agent, num_trials_per_agent),
        p=[0.15, 0.1, 0.1, 0.06, 0.06, 0.06, 0.06, 0.06, 0.1, 0.1, 0.15],
    )
    # generate bais = 0.5
    inpts[:, :, 1] = np.full((num_agent, num_trials_per_agent), 0.5)
    # (num_sess, num_trials_per_sess, input_dim) (20, 100, 2)
    inpts = list(inpts)  # convert inpts to correct format

    glm_weights = []
    true_latents, actions = [], []
    previous_actions, ar_wsls = [], []
    normal = True
    for a in range(num_agent):
        gen_weights = [
            [get_engaged_state_weights(normal, weights_sigma)],
            [get_biased_left_weights(normal, weights_sigma)],
            [get_biased_right_weights(normal, weights_sigma)],
        ]

        if num_states == 4:
            gen_weights.append([get_win_stay_weights(normal, weights_sigma)])

        gen_weights = np.array(gen_weights)
        curr_state = np.array([np.random.choice(np.arange(num_states))])
        previous_action = np.random.choice([-1, 1])
        wsls = np.random.choice([-1, 1])
        obs = previous_action  # 0 if previous_action == -1 else previous_action
        obs = np.array([obs]).reshape(1, -1)

        true_glmhmm.observations.params = gen_weights
        for t in range(num_trials_per_agent):
            curr_stim = inpts[a][t][0]
            # design matrix: stimulus, bias, previous action, w.s.l.s
            t_input = np.array([curr_stim, 0.5, previous_action, wsls]).reshape(1, -1)
            true_z, true_y = true_glmhmm.sample(
                1, prefix=(curr_state, obs), input=t_input
            )
            previous_actions.append(previous_action)
            ar_wsls.append(wsls)
            true_latents.append(true_z)
            actions.append(true_y.ravel())
            # Update input
            obs = true_y
            curr_state = np.array(true_z)
            previous_action = 1 if true_y.ravel()[0] == 0 else -1
            is_right = previous_action > 0
            rewarded = np.sign(curr_stim) == np.sign(previous_action)
            # wsls: are in {-1, 1}.  1 corresponds to
            # previous choice = right and success OR previous choice = left and
            # failure; -1 corresponds to
            # previous choice = left and success OR previous choice = right and failure
            wsls = (
                1 if (is_right and rewarded) or (not is_right and not rewarded) else -1
            )

        glm_weights.append(gen_weights)

    simulated_data = {
        "agentid": np.repeat(np.arange(num_agent), num_trials_per_agent),
        "trials": np.tile(np.arange(num_trials_per_agent), num_agent),
        "which_state": np.concatenate(true_latents),
        "actions": 1 - np.concatenate(actions),
        "stim": np.concatenate(inpts)[:, 0],
        "previous_action": previous_actions,
        "wsls": ar_wsls,
    }

    return pd.DataFrame(simulated_data)


# Generate four GLM weights in engaged state
def get_engaged_state_weights(normal: bool = False, sigma: int = 2):
    if normal:
        return [
            np.random.normal(6, sigma),
            np.random.normal(0, sigma),
            np.random.normal(0.1, sigma),
            np.random.normal(0, sigma),
        ]
    return [
        np.random.uniform(4.5, 12),
        np.random.uniform(-2, 1),
        np.random.uniform(-0.2, 1),
        np.random.uniform(-0.1, 0.1),
    ]


# Generate four GLM weights in left biased state
def get_biased_left_weights(normal: bool = False, sigma: int = 2):
    if normal:
        return [
            np.random.normal(0.5, sigma),
            np.random.normal(-3, sigma),
            np.random.normal(0.1, sigma),
            np.random.normal(0, sigma),
        ]

    return [
        np.random.uniform(0, 3),
        np.random.uniform(-3.2, -0.2),
        np.random.uniform(-1.5, 1.5),
        np.random.uniform(-0.5, 0.5),
    ]


# Generate four GLM weights in right biased state
def get_biased_right_weights(normal: bool = False, sigma: int = 2):
    if normal:
        return [
            np.random.normal(0.5, sigma),
            np.random.normal(3, sigma),
            np.random.normal(0.1, sigma),
            np.random.normal(0, sigma),
        ]

    return [
        np.random.uniform(-0.1, 2.7),
        np.random.uniform(0.2, 3.5),
        np.random.uniform(-1.5, 1.5),
        np.random.uniform(-0.5, 0.5),
    ]


# Generate four GLM weights in win-stay state
def get_win_stay_weights(normal: bool = False, sigma: int = 2):
    if normal:
        return [
            np.random.normal(2, sigma),
            np.random.normal(0, sigma),
            np.random.normal(0.5, sigma),
            np.random.normal(1, sigma),
        ]

    return [
        np.random.uniform(-0.1, 4),
        np.random.uniform(-0.1, 0.1),
        np.random.uniform(-0.1, 1.5),
        np.random.uniform(-0.25, 2),
    ]
