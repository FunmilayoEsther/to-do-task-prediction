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

@app.route("/")
def home():
    return {"message": "Task Completion Prediction API is running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert JSON to DataFrame
    df = pd.DataFrame(data)

    # Map ordinal columns
    for col in ["priority", "dead_urgency", "user_busy_level"]:
        df[col] = df[col].map(ordinal_map)

    # Drop task_id if present
    if "task_id" in df.columns:
        df = df.drop(columns=["task_id"])

    # Preprocess
    X_processed = preprocessor.transform(df)

    # Predict
    preds = model.predict(X_processed)
    probs = model.predict_proba(X_processed)[:, 1]

    # Return results
    return jsonify({
        "predictions": preds.tolist(),
        "probabilities": probs.tolist()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
