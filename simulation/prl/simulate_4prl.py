import numpy as np
import pandas as pd
import random
import copy

from scipy import special


def simulate_4prl(parameters, num_trials, pval, minswitch, num_action, agent_id):
    """
    #### Inputs ####
      parameters : model parameter values (list)
      num_trials : number of trials we want to simulate for an agent (int)
      pval : probability of the correct action being rewarded (float)
      minswitch : minimum number of trials required for the correct actions to reverse (int)
      num_action : total number of possible actions (int)
      agent_id : the sequential ID label for the agent that is being simulated (int)

    #### Outputs ####

      data : a pandas data frame containing true parameter values, agent actions/rewards, agent id, trials etc.
    """
    softmax_beta = parameters[0]  # softmax beta
    lr = parameters[1]  # learning rate
    lr_neg = parameters[2]  # negative learning rate
    learningrates = [lr_neg, lr]

    stickiness = parameters[3]  # stickiness parameter

    actionQvalues = np.array([1 / num_action] * num_action)  # initialize action values
    CurrentlyCorrect = random.choice(
        [0, 1]
    )  # initialize the action that is more likely to be rewarded at first
    currLength = minswitch + random.randint(
        0, 5
    )  # the number of trials required for the correct action to switch
    currCum = 0  # initialize cumulative reward

    allactions = []  # initialize list that will store all actions
    allrewards = []  # initialize list that will store all rewards
    allcorrectchoices = []  # initialize list that will store all trial correct actions
    isswitch = [
        0
    ] * num_trials  # initialize the list that will store an index of switch trials (1 if switch, 0 otherwise)
    alltrials = []  # initialize the list that will store the list of trials
    alliscorrectaction = (
        []
    )  # initialize the list that will store whether the agent selected the currently correct action or not (different from the reward)
    rpe_history = []
    unchosen_rpe_history = []

    for i in range(num_trials):

        W = copy.copy(actionQvalues)

        if i > 0:
            W[action] = W[action] + stickiness
        # print(f'W value: a0: {W[0]}, a1: {W[1]}')
        sftmx_p = special.softmax(
            softmax_beta * W
        )  # generate the action probability using the softmax
        action = np.random.choice(
            num_action, p=sftmx_p
        )  # select the action using the probability
        correct = (
            action == CurrentlyCorrect
        )  # is the selected action the action that is currently rewarding
        correct = int(correct)

        if (
            np.random.uniform(0, 1, 1)[0] < pval
        ):  # generate the random value between 0 and 1, if it's smaller than set pvalue the reward is +1, otherwise 0
            r = correct
        else:
            r = 1 - correct

        RPE = r - actionQvalues[action]
        unchosenaction = 1 - action  # action that's not selected
        RPEunchosen = (1 - r) - actionQvalues[
            unchosenaction
        ]  # RPE for the unselected action

        actionQvalues[action] += (
            learningrates[r] * RPE
        )  # update the action values based on the outcome

        actionQvalues[unchosenaction] += learningrates[r] * RPEunchosen
        # print(f'updated q value: a0: {actionQvalues[0]}, a1: {actionQvalues[1]}')
        currCum = currCum + r  # update cumulative reward
        if (r == 1) and (
            currCum >= currLength
        ):  # check for the counter of the trials required to switch correct actions
            CurrentlyCorrect = 1 - CurrentlyCorrect
            currLength = minswitch + random.randint(0, 5)
            currCum = 0
            if i < num_trials - 1:
                isswitch[i + 1] = 1

        # store all trial variables
        allactions.append(action)
        allrewards.append(r)
        allcorrectchoices.append(CurrentlyCorrect)
        alltrials.append(i)
        alliscorrectaction.append(correct)
        rpe_history.append(RPE)
        unchosen_rpe_history.append(RPEunchosen)

    return pd.DataFrame(
        {
            "agentid": [agent_id] * len(allactions),
            "actions": allactions,
            "correct_actions": allcorrectchoices,
            "rewards": allrewards,
            "isswitch": isswitch,
            "iscorrectaction": alliscorrectaction,
            "trials": alltrials,
            "rpe_history": rpe_history,
            "unchosen_rpe_history": unchosen_rpe_history,
            "alpha": [lr] * len(allactions),
            "beta": [softmax_beta] * len(allrewards),
            "neg_alpha": [lr_neg] * len(allrewards),
            "stickiness": [stickiness] * len(allrewards),
        }
    )
