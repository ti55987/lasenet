import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Bidirectional,
    GRU,
)

import evidential_deep_learning as edl

tf.keras.utils.set_random_seed(11)
tf.config.experimental.enable_op_determinism()

RNN = GRU

def create_evi_model(
    feature_dim: int,
    output_dim: int,
    units: int=70,
    dropout: float=0.2,
    dropout1: float=0.2,
    dropout2: float=0.1):

    activation_func = 'relu'

    encoder_inputs = layers.Input(shape=(None, feature_dim))
    encoder = Bidirectional(RNN(units, return_state=True, return_sequences=True))
    encoder_outputs, _, _ = encoder(encoder_inputs)
    encoder_outputs = Dropout(dropout)(encoder_outputs)

    # Dense layers
    outputs = Dense(int(units/2), activation=activation_func)(encoder_outputs)
    outputs = Dropout(dropout1)(outputs)
    outputs = Dense(int(units/4), activation=activation_func)(outputs)
    outputs = Dropout(dropout2)(outputs)
    outputs = edl.layers.DenseNormalGamma(output_dim)(outputs)

    # @keras.saving.register_keras_serializable(package="evi", name="EvidentialRegressionLoss")
    # def EvidentialRegressionLoss(true, pred):
    #   exp_true = tf.expand_dims(true, -1)
    #   return edl.losses.EvidentialRegression(exp_true, pred, coeff=coeff_reg)

    # optimizer = Adam(learning_rate=learning_rate)
    # model.compile(loss=EvidentialRegressionLoss, optimizer=optimizer)
    return keras.Model(encoder_inputs, outputs)


@keras.saving.register_keras_serializable(package="evi", name="EvidentialRegressionLoss")
def EvidentialRegressionLoss(true, pred):
    exp_true = tf.expand_dims(true, -1)
    return edl.losses.EvidentialRegression(exp_true, pred, coeff=1e-4)