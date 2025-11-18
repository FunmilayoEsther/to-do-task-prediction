# Task Completion Prediction – ML Capstone Project

## Project Overview
This project predicts whether a user will complete a task in a to-do list application. The goal is to use machine learning to understand task characteristics (priority, deadline, reminders, task length, etc.) and determine the likelihood of completion.

This can help improve productivity apps by:
- Identifying tasks users are likely to abandon
- Sending smart reminders; help build smarter productivity apps
- Improving personalization


## Dataset
The dataset contains 2,000 synthetic data with the following columns:

- task_length_minutes
- priority
- category
- due_in_days
- reminders_set
- past_completion_rate
- is_weekend
- completed (target variable)


| Column | Description |
|--------|-------------|
| task_length_minutes | Estimated time to finish task |
| priority | low / medium / high |
| category | work / study / health / chores / personal |
| due_in_days | Days until deadline |
| reminders_set | 1 = reminder enabled |
| past_completion_rate | User historical completion rate |
| is_weekend | 1 = task scheduled on weekend |
| completed | Target variable |

The dataset is available in the `data/tasks.csv` file.  
If not present, the notebook explains how to generate it automatically ( you can run the dataset-generation cell in `data.ipynb`).


## Contents of This Repository

| File | Description |
|------|-------------|
| `notebook.ipynb` | Data preparation, EDA, model selection, tuning |
| `train.py` | Script for training and saving the final model |
| `predict.py` | Web service for making predictions |
| `Dockerfile` | Containerization for deployment |
| `requirements.txt` | Python dependencies |
| `data/tasks.csv` | Dataset |
| `deployment/cloud_deploy.md` | Deployment instructions |

---

## Running the Project

### 1. Create environment & install dependencies

### 2. Train the model

### 3. Run the prediction service

### 4. Docker

## Deployment
(Include a link or screenshot if you deploy to cloud.)



