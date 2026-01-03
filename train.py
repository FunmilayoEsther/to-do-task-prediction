import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
df = pd.read_csv("data/tasks.csv")

# 2. Preprocessing
# Map ordinal columns
ordinal_map = {"low": 1, "medium": 2, "high": 3}
for col in ["priority", "dead_urgency", "user_busy_level"]:
    df[col] = df[col].map(ordinal_map)

# Split features and target
X = df.drop(["completed", "task_id"], axis=1)
y = df["completed"]

# Identify categorical and numerical columns
cat_cols = ["category"]
num_cols = ["priority", "dead_urgency", "user_busy_level",
            "task_length_minutes", "due_in_days",
            "reminders_set", "past_completion_rate", "is_weekend"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

# Fit preprocessing on entire dataset
X_processed = preprocessor.fit_transform(X)

# 3. Train final model
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_processed, y)

# 4. Save model and preprocessor
with open("logreg_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("Training complete. Model and preprocessor saved.")
