# Task Completion Prediction – ML Capstone Project

## Project Overview
This project builds a binary classification machine learning model to predicts whether a user will complete a task in a to-do list application. 
The goal is to use task and user behavior features (priority, urgency, reminders, task length, and historical completion rate) to estimate task completion likelihood and serve predictions via a REST API.

This project demonstrates an end-to-end machine learning workflow:
- Data preparation and exploratory analysis
- Model training and selection
- Model serialization
- API-based inference
- Containerization and cloud deployment

### Use Cases
- Identifying tasks users are likely to abandon
- Enable smart reminders and prioritization
- Support personalization in productivity applications


---

## Dataset
The dataset contains 2,000 synthetic data with the following columns:

- task_length_minutes
- priority
- category
- due_in_days
- reminders_set
- user_busy_level
- past_completion_rate
- is_weekend
- dead_urgency
- day_of_week
- completed (target variable)


### Features

| Column | Description |
|------|------------|
| priority | Task priority: low / medium / high |
| dead_urgency | Urgency of deadline: low / medium / high |
| user_busy_level | User’s workload level: low / medium / high |
| category | Task category: work / study / health / chores / personal |
| task_length_minutes | Estimated time required to complete the task |
| due_in_days | Number of days until the deadline |
| reminders_set | Number of reminders set for the task |
| past_completion_rate | Historical task completion rate of the user |
| is_weekend | 1 if task is scheduled on a weekend, else 0 |
| day_of_week | Day of week encoded as integer (0 = Monday) |
| completed | Target variable (1 = completed, 0 = not completed) |

The dataset is available in the `data/tasks.csv` file.  
If not present, the notebook explains how to generate it automatically ( you can run the dataset-generation cell in `data.ipynb`).


## Contents of This Repository

| File | Description |
|------|-------------|
| `notebook.ipynb` | Data preparation, EDA, feature analysis, model comparison, model selection, tuning |
| `train.py` | Script for training and saving the the artifacts |
| `predict.py` | Web service for making predictions |
| `Dockerfile` | Containerization for deployment |
| `requirements.txt` | Python dependencies |
| `data/tasks.csv` | Dataset |
| `deployment/cloud_deploy.md` | Deployment instructions |


---

## Exploratory Data Analysis (EDA)
EDA was performed in the notebook and includes:
- Feature distribution analysis
- Missing value checks
- Target variable balance analysis
- Feature importance inspection

This analysis informed feature selection and model choice.

---


## Model Training
Multiple models were evaluated, including:
- Logistic Regression
- Tree-based models
- Random forest

Hyperparameter tuning was performed, and the best model was selected based on performance metrics.

To train and save the final model:

```bash
python train.py


## Running the Project

### 1. Create environment & install dependencies
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

### 2. Start the Flask API
Start the Flask API: python predict.py

### 3. Health check
curl http://127.0.0.1:5001/health
#### Response
{
  "status": "ok",
  "message": "API is running"
}


### 4. Run the prediction service
curl -X POST http://127.0.0.1:5001/predict \
-H "Content-Type: application/json" \
-d '[{
  "priority":"high",
  "dead_urgency":"medium",
  "user_busy_level":"low",
  "category":"study",
  "task_length_minutes":60,
  "due_in_days":3,
  "reminders_set":1,
  "past_completion_rate":0.7,
  "is_weekend":0,
  "day_of_week":1
}]'


### 4. Docker Containerization
#### Build the Docker image
docker build -t task-completion-api .

#### Run the Docker container
docker run -p 5001:5001 task-completion-api

## Deployment
### Health check (live)
curl https://to-do-task-prediction.onrender.com/health
#### Response 
{"message":"API is running","status":"ok"}

