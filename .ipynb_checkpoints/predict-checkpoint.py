import pandas as pd
import pickle

# 1. Load model and preprocessor
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# 2. Load new data
# Example new tasks
new_data = pd.DataFrame([
    {
        "task_id": 1,
        "priority": "medium",
        "dead_urgency": "high",
        "user_busy_level": "low",
        "category": "work",
        "task_length_minutes": 45,
        "due_in_days": 2,
        "reminders_set": 1,
        "past_completion_rate": 0.8,
        "is_weekend": 0,
        "day_of_week": 2
    },
    {
        "task_id": 2,
        "priority": "high",
        "dead_urgency": "medium",
        "user_busy_level": "medium",
        "category": "personal",
        "task_length_minutes": 30,
        "due_in_days": 1,
        "reminders_set": 2,
        "past_completion_rate": 0.5,
        "is_weekend": 1,
        "day_of_week": 6
    },
    {
        "task_id": 3,
        "priority": "low",
        "dead_urgency": "low",
        "user_busy_level": "high",
        "category": "study",
        "task_length_minutes": 120,
        "due_in_days": 7,
        "reminders_set": 0,
        "past_completion_rate": 0.9,
        "is_weekend": 0,
        "day_of_week": 0
    }
])

# Map ordinal columns
ordinal_map = {"low": 1, "medium": 2, "high": 3}
for col in ["priority", "dead_urgency", "user_busy_level"]:
    new_data[col] = new_data[col].map(ordinal_map)

# 3. Preprocess and predict
X_new = new_data.drop(["task_id"], axis=1)
X_new_processed = preprocessor.transform(X_new)

y_pred = model.predict(X_new_processed)
y_prob = model.predict_proba(X_new_processed)[:, 1]

# 4. Display results
new_data["pred_completed"] = y_pred
new_data["probability"] = y_prob

print(new_data)
