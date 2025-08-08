# -*- coding: utf-8 -*-
"""
Optimisation du modèle Transformer pour la prédiction des prix des ETF
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import os
if os.path.exists("transformer_model.h5"):
    os.remove("transformer_model.h5")
    
# Charger les données historiques
data = pd.read_csv("C:/Users/2SH/Downloads/archive/Data/ETFs/ydiv.us.txt", sep=",")
data.dropna(inplace=True)  # Suppression des valeurs manquantes
prices = data['Close'].values.reshape(-1, 1)

# Normalisation des prix
scaler = MinMaxScaler(feature_range=(0, 1))
prices_normalized = scaler.fit_transform(prices)
print(prices_normalized[:10])
print(f"Prix min avant normalisation : {np.min(prices)}")
print(f"Prix max avant normalisation : {np.max(prices)}")

print(f"Prix min après normalisation : {np.min(prices_normalized)}")
print(f"Prix max après normalisation : {np.max(prices_normalized)}")

# Création des séquences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(prices_normalized, seq_length)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, model_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self._positional_encoding(sequence_length, model_dim)

    def _positional_encoding(self, sequence_length, model_dim):
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, model_dim, 2) * -(np.log(10000.0) / model_dim))
        pos_encoding = np.zeros((sequence_length, model_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)  # Ajout de la dimension batch

    def call(self, inputs):
        return inputs + self.pos_encoding
    
# Définition du bloc Transformer
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=model_dim // num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(model_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Construction du modèle Transformer
def build_transformer_model(input_shape, model_dim, num_heads, ff_dim, num_layers, dropout_rate=0.2):
    inputs = Input(shape=input_shape)
    x = inputs
    x = PositionalEncoding(input_shape[0], model_dim)(inputs)  # Ajout de l'encodage de position
    for _ in range(num_layers):
        x = TransformerBlock(model_dim, num_heads, ff_dim, dropout_rate)(x)
    x = Dense(model_dim, activation="relu")(x)
    x = Dense(1)(x)
    return Model(inputs, x)

# Hyperparamètres
input_shape = (seq_length, 1)
model_dim = 128
num_heads = 4
ff_dim = 256 
num_layers = 6
dropout_rate = 0.2

model = build_transformer_model(input_shape, model_dim, num_heads, ff_dim, num_layers, dropout_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="mse")

model.summary()

# Conversion des données en tenseurs
X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

# Entraînement avec EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(
    X_tensor, y_tensor,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Sauvegarde du modèle
model.save("transformer_model.h5")

# Génération de nouvelles données
def generate_sequence(model, initial_sequence, num_steps):
    generated_sequence = list(initial_sequence.flatten())  # Convertir en liste de scalaires
    
    for _ in range(num_steps):
        input_seq = np.array(generated_sequence[-seq_length:]).reshape(1, seq_length, 1)
        next_value = model.predict(input_seq, verbose=0).flatten()[0]  # Extraire proprement un scalaire
        generated_sequence.append(next_value)  # Ajouter comme float
    
    return np.array(generated_sequence).reshape(-1, 1)  # Retour en format colonne


initial_sequence = X[0]
synthetic_data = generate_sequence(model, initial_sequence, num_steps=30)
synthetic_data_denormalized = scaler.inverse_transform(synthetic_data)

print(synthetic_data_denormalized[:10])  # Afficher les 10 premières valeurs prédictes

predictions = model.predict(X_tensor)
print("Shape des prédictions :", predictions.shape)  # Pour vérifier la dimension
predictions = predictions[:, -1, 0]  # Prendre la dernière valeur de chaque séquence

y_true = y.squeeze()  # S'assurer que y_true a la bonne forme
print("Shape de y_true :", y_true.shape)  # Vérification
mae = mean_absolute_error(y_true, predictions)
print(f"Erreur absolue moyenne (MAE) : {mae:.4f}")

print("Moyenne des données réelles :", np.mean(prices))
print("Moyenne des données synthétiques :", np.mean(synthetic_data_denormalized))

print("Dernière valeur réelle :", prices[-1])
print("Première valeur prédite :", synthetic_data_denormalized[0])


# Affichage des résultats
plt.figure(figsize=(10, 5))
plt.plot(prices, label='Données Réelles')
plt.plot(range(len(prices), len(prices) + len(synthetic_data_denormalized)), synthetic_data_denormalized, label='Données Synthétiques')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Perte d\'entraînement')
plt.plot(history.history['val_loss'], label='Perte de validation')
plt.xlabel('Épochs')
plt.ylabel('Perte (MSE)')
plt.title('Courbes de Perte')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(prices, label="Données réelles", color="blue")
plt.plot(
    range(len(prices), len(prices) + len(synthetic_data_denormalized)), 
    synthetic_data_denormalized, 
    label="Données synthétiques (prédictions futures)", 
    color="green", linestyle="dashed"
)
plt.xlabel("Temps")
plt.ylabel("Prix")
plt.title("Prédictions futures du modèle")
plt.legend()
plt.show()

