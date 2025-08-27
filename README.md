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

#### 0.1 Set up virtual environment

#### 1.1. Model Development:

- Choose a dataset and train a baseline machine learning model.

Here we use the file src/training/train.py. To test this file from the repo root, first run:

`pip install -r requirements.txt`

You'll need a .env file with variables stored, with code like:

export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export WANDB_ENTITY=<wandb_username_or_team>
export WANDB_PROJECT=toxic-moderation
export WANDB_MODEL_NAME=toxic-comment

Then to run the file, you can run this from the project root:

`python -m src.training.train --train_csv data/train.csv --max_features 50000`

This should print F1 scores to the console. In addition, it creates a run on wandb.com. You can find my run metrics at https://wandb.ai/alexholyk-personal/toxic-moderation/runs/5g8zzk6f/overview under Summary.

#### 1.2. Experiment Tracking:

- Integrate an experiment tracking tool like MLflow or Weights & Biases.

- Log all relevant information for each training run: code version (Git commit), hyperparameters, performance metrics (e.g., accuracy, F1-score), and data versions.

This was integrated into train.py as well. See hyperparameters (vectorizer.*, classifier.*, random_state, test_size), code version (git_sha), and data version (data.sha256 and data.train_csv) at https://wandb.ai/alexholyk-personal/toxic-moderation/runs/5g8zzk6f/overview under Config.

#### 1.3. Model Versioning & Registry:

- Save your trained models as artifacts within your experiment tracking tool.

- Use the Model Registry feature to version your models. Promote your best-performing model to a "Staging" or "Production" stage.

See the models at https://wandb.ai/alexholyk-personal/toxic-moderation/artifacts/.

### Phase 2: Backend API and Database Integration

#### 2.1. FastAPI Backend:

- Create a robust FastAPI application.

- This API must load a specific model version (e.g., the latest "Production" model) from your Model Registry and serve predictions.

- Implement at least a /predict endpoint and a /health check endpoint.

For testing the API in the command line, run:

`uvicorn src.api.main:app --reload --port 8000`

You can run smoke tests by opening another terminal window and running:

`curl -s http://127.0.0.1:8000/health | jq`

...and...

`curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"comment_text": "You are disgusting."}' | jq`

To test via Postman, make sure the API is running. Open Postman, set the request type dropdown to GET, and in the URL bar, enter:

`http://127.0.0.1:8000/health`

and click Send. You should see a JSON response like:

{

  "status": "ok",

  "model": "toxic-comment",

  "model_version": "v2"

}

To test the /predict endpoint, change request type to POST and set the URL to:

`http://127.0.0.1:8000/predict`

Under the Body tab, choose JSON from the dropdown and paste:

{

  "comment_text": "You are disgusting."

}

... or another comment to test if you like. After clicking Send, you should see output like:

{

  "labels": ["toxic","insult"],

  "scores": {

    "toxic": 0.82,

    "severe_toxic": 0.07,

    "obscene": 0.18,

    "threat": 0.01,

    "insult": 0.56,

    "identity_hate": 0.03

  },

  "model_version": "v2"

}

#### 2.2. Cloud Database:

- Choose and set up a managed database on AWS.

  - SQL Option: AWS RDS (e.g., PostgreSQL).

  - NoSQL Option: Amazon DynamoDB.

- Your FastAPI service must connect to this database to log every prediction request, its output, and a timestamp. This will be used for monitoring. Your FastAPI service can also cache some predictions to avoid making predictions on frequent requests. E.g., store recommendations for frequent users to DynamoDB and pull recommendations from the store if they already exist for a user.

Here we choose SQL on AWS RDS, with PostgreSQL.

As a student, I log into AWS Academy's sandbox. I click Start Lab and AWS to get to the console. Go to Aurora and RDS, and Create a database.

Choose Standard create, PostgreSQL, and leave it at the default engine. Choose the Sandbox Template, which leaves you with Single-AZ DB instance deployment. I chose moderation-pg for DB instance identifier, mod_user for Master username. For Master password, I'm choosing this_is_my_password (this information is not private or sensitive, but I will store this password in .env). I leave the default Instance configuration at db.t4g.micro. I'm leaving most Storage and Connectivity settings at their defaults, but changing Public access to Yes to keep my options open for later. The rest of the settings stay default, except add a Security Group Inbound rule with Type PostgreSQL, Port 5432, and Source My IP.

Click Create database, and wait for status to read Available. Copy the endpoint from the console. Include it in .env like so:

APP_DB_URL=postgresql+psycopg2://mod_user:this_is_my_password@<endpoint_text>:5432/moderation

The moderation database isn't yet created. In the terminal use the command:

psql "postgresql://mod_user:this_is_my_password@moderation-pg.c6jskwc2m750.us-east-1.rds.amazonaws.com:5432/postgres"

(If postreSQL isn't installed:

brew install libpq
brew link --force libpq)

Then inside psql:

CREATE DATABASE moderation;
\c moderation

Then create the tables and indexes:

CREATE TABLE IF NOT EXISTS prediction_logs (
  id BIGSERIAL PRIMARY KEY,
  request_id UUID NOT NULL,
  comment_text TEXT NOT NULL,
  input_hash CHAR(64) NOT NULL,
  scores JSONB NOT NULL,
  labels TEXT[] NOT NULL,
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  latency_ms INTEGER NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_created_at ON prediction_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_input_hash ON prediction_logs(input_hash);

CREATE TABLE IF NOT EXISTS feedback (
  id BIGSERIAL PRIMARY KEY,
  request_id UUID NOT NULL REFERENCES prediction_logs(request_id) ON DELETE CASCADE,
  correct BOOLEAN NOT NULL,
  true_labels TEXT[] NULL,
  notes TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

This led to ERROR:  there is no unique constraint matching given keys for referenced table "prediction_logs", which I fixed with:

-- add a UNIQUE on request_id
ALTER TABLE prediction_logs
  ADD CONSTRAINT uq_prediction_logs_request UNIQUE (request_id);

-- Now create feedback referencing that unique column
CREATE TABLE IF NOT EXISTS feedback (
  id BIGSERIAL PRIMARY KEY,
  request_id UUID NOT NULL REFERENCES prediction_logs(request_id) ON DELETE CASCADE,
  correct BOOLEAN NOT NULL,
  true_labels TEXT[] NULL,
  notes TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

Use \quit to get out of psql. Restart the server:

`uvicorn src.api.main:app --reload --port 8000`

Take it for a test run with some smoke tests:

`curl -s http://localhost:8000/health | jq`

`curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"comment_text":"You are disgusting."}' | jq`


### Phase 3: Frontend and Live Monitoring

#### 3.1. User Interface:

- Build a user-facing frontend.

  - Option A (Recommended): A Streamlit dashboard.

  - Option B (Advanced): A React-based interface.

- The frontend should allow a user to send data to your FastAPI backend and see the model's prediction.

Here we implement a Streamlit dashboard. First make sure the backend is up:

`uvicorn src.api.main:app --reload --port 8000`

Then start the streamlit interface:

`streamlit run streamlit_app/app.py`

In this interface, you can enter a test comment, and adjust the decision threshold (which defaults to 0.5). It gives a bar graph showing the confidence for each label and mark the result correct or incorrect. If incorrect, you can adjust the correct labels and submit the correction, which will be logged. To check this log, you can run a query:

`psql "postgresql://mod_user:this_is_my_password@moderation-pg.c6jskwc2m750.us-east-1.rds.amazonaws.com:5432/moderation" -c "SELECT id, request_id, correct, true_labels, notes, created_at FROM feedback ORDER BY created_at DESC LIMIT 5;"`

You can cross-check which prediction the feedback refers to with:

`psql "postgresql://mod_user:this_is_my_password@moderation-pg.c6jskwc2m750.us-east-1.rds.amazonaws.com:5432/moderation" -c "SELECT request_id, comment_text, labels, created_at FROM prediction_logs WHERE request_id IN (SELECT request_id FROM feedback ORDER BY created_at DESC LIMIT 5);"`


#### 3.2. Model Monitoring Dashboard:

- This should be a separate frontend application (on a different EC2 server - data will be exchanged through a Database, not JSON files)

- The dashboard must connect to your cloud database (RDS/DynamoDB) and visualize key monitoring metrics from the prediction logs, such as:

  - Prediction latency over time.

  - Distribution of predicted classes (target drift).

  - A mechanism to collect user feedback on model predictions to calculate live accuracy.

  First we reconnect to the database:

  `psql "postgresql://mod_user:this_is_my_password@moderation-pg.c6jskwc2m750.us-east-1.rds.amazonaws.com:5432/moderation"`

  Then we add some helpful DB indexes:
  Faster joins: feedback → prediction_logs
  `CREATE INDEX IF NOT EXISTS idx_feedback_request_id ON feedback(request_id);`

  Filter by version faster (optional)
  `CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_version ON prediction_logs(model_version);`

  Optional: GIN index for labels array (useful as data grows)
  `CREATE INDEX IF NOT EXISTS idx_prediction_logs_labels_gin ON prediction_logs USING GIN (labels);`

### Phase 4: Testing and CI/CD Automation

#### 4.1. Comprehensive Testing:

- Unit Tests: Write tests for individual functions (e.g., data preprocessing logic).

- Integration Tests: Write tests for your FastAPI endpoints to ensure they work as expected. Use pytest.

Call `pytest -q` from the home directory to verify this.

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