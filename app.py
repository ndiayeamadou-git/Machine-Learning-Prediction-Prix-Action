from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import io
import base64
from keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from model import TransformerBlock, PositionalEncoding  # Importer la couche personnalisée

app = Flask(__name__)

# Enregistrer la couche personnalisée pour éviter les erreurs de chargement
tf.keras.utils.get_custom_objects()["TransformerBlock"] = TransformerBlock

# Charger le modèle entraîné
model = load_model("transformer_model.h5", custom_objects={"TransformerBlock": TransformerBlock, "PositionalEncoding": PositionalEncoding,  "mse": MeanSquaredError()})

# Charger les données pour normalisation
df = pd.read_csv("C:/Users/2SH/Downloads/archive/Data/ETFs/ydiv.us.txt", sep=",")
prices = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(prices)

def generate_sequence(model, initial_sequence, num_steps, scaler, seq_length):
    generated_sequence = list(initial_sequence)

    while len(generated_sequence) < seq_length:
        generated_sequence.insert(0, 0)

    predictions = []
    for i in range(num_steps):
        input_seq = np.array(generated_sequence[-seq_length:]).reshape(1, seq_length, 1)
        predicted_value = model.predict(input_seq, verbose=0)[0, 0]
        generated_sequence.append(predicted_value)
        predictions.append(predicted_value)

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("sequence", [])
    seq_length = 30  # Longueur de séquence utilisée à l'entraînement
    num_steps = 30  # Nombre de pas de prédiction

    if len(data) < seq_length:
        return jsonify({"error": f"La séquence d'entrée doit contenir au moins {seq_length} valeurs."}), 400

    initial_sequence = scaler.transform(np.array(data).reshape(-1, 1))
    predictions = generate_sequence(model, initial_sequence, num_steps, scaler, seq_length)

    # Générer un graphique des prédictions
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Prédictions')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({
        "predictions": [round(p, 3) for p in predictions.flatten().tolist()],
        "plot_url": plot_url
    })

if __name__ == "__main__":
    app.run(debug=True)
