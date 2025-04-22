from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import re
import pandas as pd
import spacy
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- Initialize ----------------
app = Flask(__name__)
nlp = spacy.blank("hi")  # Hindi blank model

# ---------------- Load models and assets ----------------
cnn_model = load_model("cnn_model_sh.keras")
lstm_model = load_model("lstm_model_sh.keras")
rnn_model = load_model("rnn_model_sh.keras")
meta_model = load_model("meta_model_sh.keras")

with open("tokenizer_sh.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder_sh.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("safe_stopwords_hi.json", "r", encoding="utf-8") as f:
    hindi_stopwords = set(json.load(f))

# ---------------- Preprocessing ----------------
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[a-zA-Z]', '', text)  # remove English letters
    text = re.sub(r'[^\u0900-\u097F0-9\s]', '', text)  # keep only Hindi chars and numbers
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp(text)
    tokens = [token.text for token in doc if token.text not in hindi_stopwords and token.text.strip() != ""]
    return " ".join(tokens)

# ---------------- Prediction Function ----------------
def predict_category(headline):
    headline = preprocess_text(headline)
    sequence = tokenizer.texts_to_sequences([headline])
    padded_sequence = pad_sequences(sequence, maxlen=200)

    cnn_pred = cnn_model.predict(padded_sequence)
    lstm_pred = lstm_model.predict(padded_sequence)
    rnn_pred = rnn_model.predict(padded_sequence)

    stacked_pred = np.concatenate([cnn_pred, lstm_pred, rnn_pred], axis=1)
    final_pred = meta_model.predict(stacked_pred)

    predicted_class = label_encoder.inverse_transform([np.argmax(final_pred)])
    confidence = float(np.max(final_pred))

    return predicted_class[0], round(confidence, 3)

# ---------------- Flask Route ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "headline" not in data:
            return jsonify({"error": "Missing 'headline' in request"}), 400

        headline = data["headline"]
        category, confidence = predict_category(headline)

        return jsonify({
            "category": category,
            "confidence": f"{confidence * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
