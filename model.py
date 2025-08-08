import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length,model_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.model_dim = model_dim

        # Création des positions et de l'angle pour le sinus/cosinus
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, model_dim, 2) * -(np.log(10000.0) / model_dim))

        # Matrice d'encodage de position
        pe = np.zeros((sequence_length, model_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.positional_encoding = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "model_dim": self.model_dim
        })
        return config

# Enregistrer la couche pour que `load_model` puisse la retrouver
tf.keras.utils.get_custom_objects()["PositionalEncoding"] = PositionalEncoding

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, ff_dim, dropout_rate=0.2, trainable=True, **kwargs):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=model_dim// num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(model_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# S'assurer que le modèle est bien enregistré dans `get_custom_objects`
tf.keras.utils.get_custom_objects()["TransformerBlock"] = TransformerBlock
