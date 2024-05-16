import numpy as np
import pandas as pd

from enum import Enum
import tensorflow as tf
from tensorflow import one_hot
from tensorflow.python.keras.layers import Concatenate

from utils.constants import GLM_HMM_SIMULUS_VALUES


class CognitiveModel(Enum):
    PRL4 = 1
    PRL_foraging_dynamic = 2
    HRL2 = 3
    GLM_HMM = 4

def get_latent_labels(data, latent_key):
  num_agents = len(data['agentid'].unique())
  num_trial = len(data['trials'].unique())
  return data[latent_key].to_numpy().astype(np.float32).reshape((num_agents, num_trial))

def get_labels_by_model(data, model: CognitiveModel):
    n_agent = len(data['agentid'].unique())
    n_trial = len(data['trials'].unique())
    if model == CognitiveModel.PRL4:
        # continous label
        qv = np.array(data['rewards'] - data['rpe_history'])
        return qv.astype(np.float32).reshape((n_agent, n_trial))
    elif model == CognitiveModel.PRL_foraging_dynamic:
        # discrete label
        state_labels = get_latent_labels(data, ['latent_att'])
        n_st = len(data.latent_att.unique())
        normalized_st_labels = tf.keras.utils.to_categorical(state_labels, num_classes=n_st)
        # continous label
        winning_q_labels = get_latent_labels(data, ['chosen_qv'])
        return winning_q_labels, normalized_st_labels
    elif model == CognitiveModel.HRL2:
        # discrete label
        chosen_cue_labels = get_latent_labels(data, ['chosencue'])
        n_cue = len(data.chosencue.unique())
        normalized_cue_labels = tf.keras.utils.to_categorical(chosen_cue_labels, num_classes=n_cue)
        # continous label
        #train_labels = _get_qv_labels(data)
        qv = np.array(data['rewards'] - data['rpe_history'])
        winning_q_labels = qv.astype(np.float32).reshape((n_agent, n_trial))
        return winning_q_labels, normalized_cue_labels
    elif model == CognitiveModel.GLM_HMM:
        state_labels = get_latent_labels(data, 'which_state')
        n_st = len(data.which_state.unique())
        return  tf.keras.utils.to_categorical(state_labels, num_classes=n_st)

def get_feature_list_by_model(model: CognitiveModel):
    if model == CognitiveModel.PRL4:
        return ["actions", "rewards"]
    elif model == CognitiveModel.PRL_foraging_dynamic:
        return ["actions", "rewards"]
    elif model == CognitiveModel.HRL2:
        return ["chosenside", "rewards", "allstims0", "allstims1", "allstims2"]
    elif model == CognitiveModel.GLM_HMM:
        return ["stim", "actions", "wsls"]

def get_onehot_features(data, input_list):
    n_agent = len(data["agentid"].unique())
    n_trial = len(data["trials"].unique())
    features = []
    for key in input_list:
        input = data[key].to_numpy()
        unique_input = GLM_HMM_SIMULUS_VALUES if key == "stim" else np.unique(input)

        cat_map = {item: i for i, item in enumerate(unique_input)}
        input_cat = [cat_map[s] for s in input]
        input_cat = np.array(input_cat).astype(np.int32).reshape((n_agent, n_trial))
        features.append(one_hot(input_cat, len(unique_input)))

    return Concatenate(axis=2)(features)
