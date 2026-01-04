import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import pickle

# Load your dataset
df = pd.read_csv("data/tasks.csv")

# Preprocessing
categorical_cols = ['category']
ordinal_cols = ['priority', 'dead_urgency', 'user_busy_level']
numeric_cols = ['task_length_minutes', 'due_in_days', 'reminders_set', 'past_completion_rate', 'is_weekend', 'day_of_week']

# Example ordinal mapping
ordinal_map = {"low":1, "medium":2, "high":3}
for col in ordinal_cols:
    df[col] = df[col].map(ordinal_map)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

X = df[ordinal_cols + numeric_cols + categorical_cols]
y = df['completed']

X_processed = preprocessor.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_processed, y)

# Save model and preprocessor
with open("logreg_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)
