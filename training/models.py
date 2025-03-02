import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Bidirectional,
    GRU,
)

tf.keras.utils.set_random_seed(11)
tf.config.experimental.enable_op_determinism()

RNN = GRU

def create_model(
    feature_dim: int,
    discrete_output_dim: int,
    continuous_output_dim: int,
    units: int = 70,
    dropout: float = 0.2,
    dropout1: float = 0.2,
    dropout2: float = 0.1,
):
    activation_func = "relu"
    # Bidirectional
    encoder_inputs = layers.Input(shape=(None, feature_dim))
    encoder = Bidirectional(RNN(units, return_state=True, return_sequences=True))
    encoder_outputs, _, _ = encoder(encoder_inputs)
    encoder_outputs = Dropout(dropout)(encoder_outputs)
    
    d_outputs_1 = Dense(int(units / 2), activation=activation_func)(encoder_outputs)
    d_outputs_1 = Dropout(dropout1)(d_outputs_1)
    d_outputs_1 = Dense(int(units / 4), activation=activation_func)(d_outputs_1)
    d_outputs_1 = Dropout(dropout2)(d_outputs_1)

    outputs = []
    if discrete_output_dim > 0:
        k_outputs = Dense(discrete_output_dim, activation="softmax", name="discrete_latent")(d_outputs_1)
        outputs.append(k_outputs)

    if continuous_output_dim > 0:
        linear_outputs = Dense(continuous_output_dim, activation="linear", name="continuous_latent")(
            d_outputs_1
        )
        outputs.append(linear_outputs)

    return keras.Model(inputs=encoder_inputs, outputs=outputs)
