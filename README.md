## Final Project: Toxic Comment Moderation

### Alex Holyk

### Core System Features

This is a multi-component application that includes the following, all deployed on AWS:

- Experiment Tracking & Model Registry: A system to log experiment parameters/metrics and manage model versions.

- ML Model Backend: A FastAPI application to serve your registered model.

- Persistent Data Store: A cloud-native database (SQL or NoSQL) for storing prediction logs, user feedback, or other relevant data.

- Frontend Interface: A user-facing application for interacting with your model.

- Model Monitoring Dashboard: A dashboard to visualize model performance and data drift in production.

- CI/CD Pipeline: An automated workflow to test and validate code changes.

## Production Phases

### Phase 1: Experimentation and Model Management

#### 1.1. Model Development:

- Choose a dataset and train a baseline machine learning model.

Here we use the file src/training/train.py. To test this file from the repo root, run:

`pip install -r requirements.txt`

`export MLFLOW_TRACKING_URI="file:$(pwd)/mlruns"`

`python -m src.training.train \`
`  --train_csv data/train.csv \`
`  --experiment toxicity-baselines \`
`  --registered_model_name toxic-comment \`
`  --max_features 50000`

This should print F1 scores to the console.

#### 1.2. Experiment Tracking:

- Integrate an experiment tracking tool like MLflow or Weights & Biases.

- Log all relevant information for each training run: code version (Git commit), hyperparameters, performance metrics (e.g., accuracy, F1-score), and data versions.

#### 1.3. Model Versioning & Registry:

- Save your trained models as artifacts within your experiment tracking tool.

- Use the Model Registry feature to version your models. Promote your best-performing model to a "Staging" or "Production" stage.

### Phase 2: Backend API and Database Integration

#### 2.1. FastAPI Backend:

- Create a robust FastAPI application.

- This API must load a specific model version (e.g., the latest "Production" model) from your Model Registry and serve predictions.

- Implement at least a /predict endpoint and a /health check endpoint.

#### 2.2. Cloud Database:

- Choose and set up a managed database on AWS.

  - SQL Option: AWS RDS (e.g., PostgreSQL).

  - NoSQL Option: Amazon DynamoDB.

- Your FastAPI service must connect to this database to log every prediction request, its output, and a timestamp. This will be used for monitoring. Your FastAPI service can also cache some predictions to avoid making predictions on frequent requests. E.g., store recommendations for frequent users to DynamoDB and pull recommendations from the store if they already exist for a user.

### Phase 3: Frontend and Live Monitoring

#### 3.1. User Interface:

- Build a user-facing frontend.

  - Option A (Recommended): A Streamlit dashboard.

  - Option B (Advanced): A React-based interface.

- The frontend should allow a user to send data to your FastAPI backend and see the model's prediction.

#### 3.2. Model Monitoring Dashboard:

- This should be a separate frontend application (on a different EC2 server - data will be exchanged through a Database, not JSON files)

- The dashboard must connect to your cloud database (RDS/DynamoDB) and visualize key monitoring metrics from the prediction logs, such as:

  - Prediction latency over time.

  - Distribution of predicted classes (target drift).

  - A mechanism to collect user feedback on model predictions to calculate live accuracy.

### Phase 4: Testing and CI/CD Automation

#### 4.1. Comprehensive Testing:

- Unit Tests: Write tests for individual functions (e.g., data preprocessing logic).

- Integration Tests: Write tests for your FastAPI endpoints to ensure they work as expected. Use pytest.

#### 4.2. CI/CD Pipeline:

- Set up a GitHub Actions workflow (.github/workflows/ci.yml).

- The workflow must automatically trigger on pull requests to the main branch.

- It must run a linter (e.g., flake8 or ruff) and execute your full test suite (pytest). A pull request cannot be merged if these checks fail.

### Phase 5: Containerization and Deployment

#### 5.1. Docker Packaging:

- Containerize your application components (e.g., one container for the FastAPI backend, one for the frontend).

#### 5.2. AWS Deployment:

- Deploy your containerized application to separate EC2 instances with Docker installed.

#### 5.3. Documentation:

- Create a high-quality README.md in your GitHub repository. It must be a complete guide to your project, including setup instructions, deployment steps, and example requests by user.

### Problem statement & labels.

### End-to-end diagram and component list.

### Exact environment variables for each service.

### One-liner run commands (local Docker & AWS).

### MLflow dashboard URL + screenshots.

### Example curl:

`curl -X POST http://<api-ec2>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"comment_text":"You are disgusting."}'`






--------------------------------------------------------------------






## Full Assignment Text:

### Final Course Project: Building a Production-Grade MLOps System

This is the Final project for this course. Your goal is to design, build, and deploy a complete, end-to-end machine learning application that incorporates best practices from across the MLOps lifecycle you learned in the class. You will be responsible for everything from model experimentation and versioning to automated testing, deployment on AWS, and live monitoring.

You are recommended to work in pairs.

**Due Date**: August 26th, 2025

### Project Overview

You are tasked with building a production-ready ML service. You have the freedom to one of the 4 given problems, but the system you build must meet a rigorous set of technical requirements.

### Core System Requirements

Your final system must be a multi-component application that includes the following, all deployed on AWS:

- Experiment Tracking & Model Registry: A system to log experiment parameters/metrics and manage model versions.

- ML Model Backend: A FastAPI application to serve your registered model.

- Persistent Data Store: A cloud-native database (SQL or NoSQL) for storing prediction logs, user feedback, or other relevant data.

- Frontend Interface: A user-facing application for interacting with your model.

- Model Monitoring Dashboard: A dashboard to visualize model performance and data drift in production.

- CI/CD Pipeline: An automated workflow to test and validate code changes.

## Production Phases

### Phase 1: Experimentation and Model Management

#### 1.1. Model Development:

- Choose a dataset and train a baseline machine learning model.

#### 1.2. Experiment Tracking:

- Integrate an experiment tracking tool like MLflow or Weights & Biases.

- Log all relevant information for each training run: code version (Git commit), hyperparameters, performance metrics (e.g., accuracy, F1-score), and data versions.

#### 1.3. Model Versioning & Registry:

- Save your trained models as artifacts within your experiment tracking tool.

- Use the Model Registry feature to version your models. Promote your best-performing model to a "Staging" or "Production" stage.

### Phase 2: Backend API and Database Integration

#### 2.1. FastAPI Backend:

- Create a robust FastAPI application.

- This API must load a specific model version (e.g., the latest "Production" model) from your Model Registry and serve predictions.

- Implement at least a /predict endpoint and a /health check endpoint.

#### 2.2. Cloud Database:

- Choose and set up a managed database on AWS.

  - SQL Option: AWS RDS (e.g., PostgreSQL).

  - NoSQL Option: Amazon DynamoDB.

- Your FastAPI service must connect to this database to log every prediction request, its output, and a timestamp. This will be used for monitoring. Your FastAPI service can also cache some predictions to avoid making predictions on frequent requests. E.g., store recommendations for frequent users to DynamoDB and pull recommendations from the store if they already exist for a user.

### Phase 3: Frontend and Live Monitoring

#### 3.1. User Interface:

- Build a user-facing frontend.

  - Option A (Recommended): A Streamlit dashboard.

  - Option B (Advanced): A React-based interface.

- The frontend should allow a user to send data to your FastAPI backend and see the model's prediction.

#### 3.2. Model Monitoring Dashboard:

- This should be a separate frontend application (on a different EC2 server - data will be exchanged through a Database, not JSON files)

- The dashboard must connect to your cloud database (RDS/DynamoDB) and visualize key monitoring metrics from the prediction logs, such as:

  - Prediction latency over time.

  - Distribution of predicted classes (target drift).

  - A mechanism to collect user feedback on model predictions to calculate live accuracy.

### Phase 4: Testing and CI/CD Automation

#### 4.1. Comprehensive Testing:

- Unit Tests: Write tests for individual functions (e.g., data preprocessing logic).

- Integration Tests: Write tests for your FastAPI endpoints to ensure they work as expected. Use pytest.

#### 4.2. CI/CD Pipeline:

- Set up a GitHub Actions workflow (.github/workflows/ci.yml).

- The workflow must automatically trigger on pull requests to the main branch.

- It must run a linter (e.g., flake8 or ruff) and execute your full test suite (pytest). A pull request cannot be merged if these checks fail.

### Phase 5: Containerization and Deployment

#### 5.1. Docker Packaging:

- Containerize your application components (e.g., one container for the FastAPI backend, one for the frontend).

#### 5.2. AWS Deployment:

- Deploy your containerized application to separate EC2 instances with Docker installed.

#### 5.3. Documentation:

- Create a high-quality README.md in your GitHub repository. It must be a complete guide to your project, including setup instructions, deployment steps, and example requests by user.

#### Topics to choose from:

1. Taxi Fare / ETA Prediction

  Problem: Predict trip fare (or duration) from pickup time/geo and trip features; provide real-time scoring for incoming rides.

  Dataset: NYC TLC Trip Records. Public on AWS Open Data + NYC site

2. Personalized Book Recommender

  Problem: Build a system that recommends books given a user’s favorite titles.

  Dataset: Amazon Review Data — Books subset 

3. U.S. Flight Delay Prediction & Ops Dashboard

  Problem: Predict arrival delays (e.g., >15 minutes) and surface route-level reliability for travelers/ops.

  Dataset: U.S. DOT/BTS On-Time Performance

4. Toxic Comment Moderation

  Problem: Classify user comments into toxicity categories and expose a moderation endpoint with human-review workflow

  Dataset: Jigsaw Toxic Comment Classification (English)

### Deliverables & Submission

1. **GitHub Repository URL**: A link to your public GitHub repository containing all code, configuration files, and documentation.

2. **Experiment Tracking Dashboard URL**: A link to your public MLflow or W&B project dashboard.

Submit them as your final project.