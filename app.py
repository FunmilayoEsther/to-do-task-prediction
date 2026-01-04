from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# 1. Load model and preprocessor
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# 2. API endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Expecting a list of dicts
        df = pd.DataFrame(data)

        # Map ordinal columns if needed
        ordinal_map = {"low": 1, "medium": 2, "high": 3}
        for col in ["priority", "dead_urgency", "user_busy_level"]:
            if col in df.columns:
                df[col] = df[col].map(ordinal_map)

        # Drop task_id if exists
        X = df.drop(columns=["task_id"], errors="ignore")

        # Transform and predict
        X_processed = preprocessor.transform(X)
        y_prob = model.predict_proba(X_processed)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        return jsonify({
            "predictions": y_pred.tolist(),
            "probabilities": y_prob.round(3).tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
