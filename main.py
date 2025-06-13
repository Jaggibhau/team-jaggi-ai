
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model/decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Team Jaggi AI Predictor is Live!"

@app.route("/predict", methods=["GET"])
def predict():
    history = request.args.get("history")
    if not history:
        return jsonify({"error": "Missing 'history' parameter"}), 400
    try:
        mapping = {"BIG": 1, "SMALL": 0}
        last3 = [mapping[item.strip().upper()] for item in history.split(",")[-3:]]
        if len(last3) != 3:
            return jsonify({"error": "Need exactly 3 values"}), 400
        prediction = model.predict([last3])[0]
        confidence = max(model.predict_proba([last3])[0])
        label = "BIG" if prediction == 1 else "SMALL"
        return jsonify({
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
