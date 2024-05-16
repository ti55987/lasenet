import numpy as np
from numpy.random import shuffle

import random
import pandas as pd
import copy
from scipy import special


def simulate_hrl(parameters, numtrials, pval, pswitch, numbandits, agentid):

    softmaxbeta = parameters[0]  # softmax beta
    learningrate = parameters[1]  # learning rate
    stickiness = parameters[2]
    epsilon = parameters[3]

    Q = 1 / numbandits * np.ones([1, numbandits])[0]  # initialize action values
    iter = 0
    cb = 1

    a = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 0, 0]])
    a = np.repeat(a, numtrials / len(a), axis=0)
    shuffle(a)

    # allactions = [] #initialize list that will store all actions
    allrewards = []  # initialize list that will store all rewards
    allcorrectcues = []
    alltrials = []
    alliscorrectcue = []
    alliters = []
    allindexofselcolor = []
    allchosenside = []
    isswitch = [0] * numtrials
    allstims0, allstims1, allstims2 = [], [], []
    all_qv = {i: [] for i in range(numbandits)}
    chosen_q, rpe_history = [], []
    for i in range(numtrials):
        # record q values
        for nb in range(numbandits):
            all_qv[nb].append(Q[nb])

        # three stimuli, each pointing to left/right (where left == 1; right == 0)
        stim = a[i]
        # Q value, adding stickiness to the index of previously selected action if not the first iteration
        W = copy.copy(Q)

        if i > 0:
            W[b] = W[b] + stickiness

        # select the action using softmax
        sftmx_p = special.softmax(softmaxbeta * W)
        # generate the action probability using the softmax
        b = np.random.choice(
            numbandits, p=sftmx_p
        )  # generate the action using the probability

        # s= side the selected stimulus is pointing to
        # if there's noise add a possible slip
        if np.random.uniform(0, 1, 1)[0] < epsilon:
            s = 1 - stim[b]
        else:
            s = stim[b]

        # check if the side of the selected cue is the same as the side of the correct cue
        cor = int(s == stim[cb])

        # reward with p probability
        r = int(np.random.uniform(0, 1, 1)[0] < pval[cor])

        # update the q value of the selected cue
        b_rpe = r - Q[b]
        chosen_q.append(Q[b])

        Q[b] += learningrate * (b_rpe)

        # update the q value of other cues (counterfactual learning)
        others = [x for x in list(np.arange(numbandits)) if x != b]
        Q[np.array(others)] += learningrate * ((1 - r) - Q[np.array(others)])

        alltrials.append(i)
        allcorrectcues.append(
            cb
        )  # store the action that was correct on the current trial
        alliters.append(iter)
        allindexofselcolor.append(b)
        allchosenside.append(s)
        alliscorrectcue.append(cor)
        allrewards.append(r)  # store
        allstims0.append(stim[0])
        allstims1.append(stim[1])
        allstims2.append(stim[2])
        rpe_history.append(b_rpe)

        # after n trials (10 in this case?) switch the correct stimulus
        if (iter > 10) and (np.random.uniform(0, 1, 1)[0] < pswitch):
            iter = 1
            bs = np.array([x for x in list(np.arange(numbandits)) if x != cb])
            cb = bs[random.choice([0, 1])]
            if i < numtrials - 1:
                isswitch[i + 1] = 1
        else:
            iter += 1

    data = {
        "agentid": [agentid] * len(alltrials),
        "correctcue": allcorrectcues,
        "rewards": allrewards,
        "isswitch": isswitch,
        "iscorrectcue": alliscorrectcue,
        "trials": alltrials,
        "rpe_history": rpe_history,
        "chosen_qv": chosen_q,
        "chosenside": allchosenside,
        "chosencue": allindexofselcolor,
        "correctruleiteration": alliters,
        "alpha": [learningrate] * len(alltrials),
        "stickiness": [stickiness] * len(alltrials),
        "allstims0": allstims0,
        "allstims1": allstims1,
        "allstims2": allstims2,
        "beta": [softmaxbeta] * len(alltrials),
    }
    for nb in range(numbandits):
        data[f"qv{nb}"] = all_qv[nb]

    return pd.DataFrame(data)
