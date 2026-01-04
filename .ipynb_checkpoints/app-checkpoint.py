from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# 1. Load model and preprocessor
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# 2. Predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        # Map ordinal features to numeric
        ordinal_map = {"low": 1, "medium": 2, "high": 3}
        for col in ["priority", "dead_urgency", "user_busy_level"]:
            if col in df.columns:
                df[col] = df[col].map(ordinal_map)

        # Preprocess
        X = df.copy()
        if "task_id" in X.columns:
            X = X.drop("task_id", axis=1)
        X_processed = preprocessor.transform(X)

        # Predict
        y_pred = model.predict(X_processed)
        y_prob = model.predict_proba(X_processed)[:, 1]

        # Return results
        return jsonify({
            "predictions": y_pred.tolist(),
            "probabilities": y_prob.round(3).tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
