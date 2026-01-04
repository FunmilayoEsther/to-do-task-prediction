from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and preprocessor
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Ordinal mapping
ordinal_map = {"low": 1, "medium": 2, "high": 3}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "message":"API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        # Apply ordinal mapping
        for col in ["priority", "dead_urgency", "user_busy_level"]:
            df[col] = df[col].map(ordinal_map)

        X = df.drop(columns=["task_id"], errors="ignore")
        X_processed = preprocessor.transform(X)

        y_prob = model.predict_proba(X_processed)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        return jsonify({
            "predictions": y_pred.tolist(),
            "probabilities": y_prob.round(3).tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
