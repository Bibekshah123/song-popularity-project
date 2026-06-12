# 🎵 Song Popularity MLOps Pipeline — Complete Implementation Guide

> **Student:** Bibek Shah | **ID:** 23189619 | **Module:** CMP6230 — Data Management & Machine Learning Operations
> **University:** Birmingham City University | **Date:** May 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Technology Stack](#3-technology-stack)
4. [Folder Structure](#4-folder-structure)
5. [Prerequisites](#5-prerequisites)
6. [Step 1 — Install MariaDB ColumnStore](#6-step-1--install-mariadb-columnstore)
7. [Step 2 — Install Redis](#7-step-2--install-redis)
8. [Step 3 — Python Environment Setup](#8-step-3--python-environment-setup)
9. [Step 4 — Place the Dataset](#9-step-4--place-the-dataset)
10. [Step 5 — Run ETL Ingestion](#10-step-5--run-etl-ingestion)
11. [Step 6 — Run Data Validation](#11-step-6--run-data-validation)
12. [Step 7 — Run Preprocessing](#12-step-7--run-preprocessing)
13. [Step 8 — Start MLflow](#13-step-8--start-mlflow)
14. [Step 9 — Train Models](#14-step-9--train-models)
15. [Step 10 — Model Explainability (SHAP)](#15-step-10--model-explainability-shap)
16. [Step 11 — Run Monitoring (Lifecycle Activity)](#16-step-11--run-monitoring-lifecycle-activity)
17. [Step 12 — Start FastAPI Backend](#17-step-12--start-fastapi-backend)
18. [Step 13 — Start Frontend](#18-step-13--start-frontend)
19. [Step 14 — Docker Compose Deployment](#19-step-14--docker-compose-deployment)
20. [Step 15 — Apache Airflow Setup and DAG Execution](#20-step-15--apache-airflow-setup-and-dag-execution)
21. [Verify Redis Keys](#21-verify-redis-keys)
22. [Verify MariaDB Tables](#22-verify-mariadb-tables)
23. [All Browser URLs](#23-all-browser-urls)
24. [Screenshots Guide for Report](#24-screenshots-guide-for-report)
25. [Troubleshooting](#25-troubleshooting)
26. [Viva Questions and Answers](#26-viva-questions-and-answers)
27. [Presentation Flow for Supervisor](#27-presentation-flow-for-supervisor)
28. [Personal Reflection](#28-personal-reflection)
29. [References](#29-references)

---

## 1. Project Overview

This project implements a **complete end-to-end MLOps pipeline** for predicting the popularity of songs from Spotify audio features using supervised regression.

### What this project does

- Ingests raw song data from a CSV file into a MariaDB ColumnStore data warehouse
- Validates the data for schema correctness, data types, null rates and target range
- Preprocesses the data using median imputation, IQR outlier clipping and StandardScaler
- Trains and compares Linear Regression, Random Forest and XGBoost regression models
- Tracks all experiments using MLflow
- Explains model predictions using SHAP
- Deploys the best model as a REST API using FastAPI
- Provides a frontend web interface for users to enter audio features and get predictions
- Monitors data drift using Evidently AI as an ongoing lifecycle activity
- Orchestrates the batch pipeline using Apache Airflow
- Uses Redis as intermediate metadata storage between Airflow tasks
- Runs all services using Docker Compose

### Why this dataset was selected

The Song Popularity Dataset was selected because music popularity prediction has **real-world value** for:
- **Streaming platforms** — accurately predicting song popularity enables smarter playlist curation, improving user retention and discovery algorithms on platforms such as Spotify and Apple Music
- **Artist promotion** — record labels and marketing teams can allocate promotional budgets more effectively by identifying high-potential tracks before release
- **Recommendation systems** — popularity scores serve as a key signal for collaborative and content-based filtering, enhancing the relevance of personalised recommendations
- **Music trend analysis** — understanding which audio features correlate with popularity helps producers and artists make data-driven creative decisions, reducing the financial risk of new releases

The dataset contains approximately **18,835 songs** with **13 numeric Spotify audio features** and one target variable `song_popularity` (0–100), making it suitable for a regression-based MLOps pipeline. Its moderate size allows rapid iteration during pipeline development while remaining representative of real-world music analytics workloads.

### Key design decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Data warehouse | MariaDB ColumnStore | Columnar engine is optimised for analytical scans over numeric audio features; stores columns adjacently on disk so queries like AVG(energy) or correlation analysis read only the required columns, not entire rows — significantly faster than row-based storage for this workload |
| Intermediate storage | Redis | Lightweight in-memory key-value store for metadata handoff between Airflow tasks; avoids passing large DataFrames through Airflow's XCom system, keeps pipeline stages decoupled, and provides sub-millisecond read/write latency for status flags, row counts and model paths |
| Storage model | One Big Table (OBT) | The dataset has a single entity (song) with no separate artist, album, genre or playlist dimension tables. A star schema would introduce unnecessary joins that provide no analytical benefit. OBT keeps queries simple and scan-efficient for columnar storage. |
| Data strategy | ETL (not ELT) | The raw CSV may contain missing values, outliers, duplicate rows and out-of-range target values. ETL validates and cleans data _before_ loading into the warehouse, preventing bad data from ever entering the analytical layer and ensuring downstream consumers always read clean, consistent data. |
| Monitoring | Lifecycle activity (not DAG task) | Monitoring spans the entire system — ingestion quality, feature distributions, model predictions and API response health. It must run continuously, not once per DAG execution. Evidently AI runs independently of the Airflow pipeline. |

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCE                                      │
│            song_popularity.csv (Kaggle, 18,835 songs)                    │
│            Tool: Kaggle Dataset / Pandas read_csv                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 ── ETL INGESTION                                                │
│  Tool: Python, pandas, SQLAlchemy, pymysql, pyarrow, joblib              │
│  Script: src/etl_ingestion.py                                            │
│  Actions: Column rename → drop duplicates → load to warehouse            │
│           → write Parquet cache → write Redis metadata                   │
├──────────────────────┬───────────────────────────┬───────────────────────┤
│                      │                           │                       │
│                      ▼                           ▼                       │
│  ┌────────────────────────────┐    ┌────────────────────────────┐        │
│  │ MariaDB ColumnStore        │    │ Redis                      │        │
│  │ (Columnar Data Warehouse)  │    │ (In-memory metadata store)  │        │
│  │ songs_analytics table      │    │ ingestion_status: success   │        │
│  │ ingestion_log table        │    │ ingestion_row_count: 18835  │        │
│  └─────────────┬──────────────┘    └─────────────┬──────────────┘        │
│                │                                 │                        │
└────────────────┼─────────────────────────────────┼────────────────────────┘
                 │                                 │
                 ▼                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 2 ── DATA VALIDATION                                              │
│  Tool: Python, pandas, SQLAlchemy, Redis                                 │
│  Script: src/validate_data.py                                            │
│  Actions: Check schema → data types → null rates → target range (0–100)  │
│           → duplicate check → write validation_status to Redis           │
├──────────────────────────────────────────────────────────────────────────┤
│  Redis: validation_status = passed, validation_row_count = 18835         │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 3 ── PREPROCESSING                                                │
│  Tool: Python, pandas, scikit-learn (SimpleImputer, StandardScaler)      │
│  Script: src/preprocess.py                                               │
│  Actions: Drop song_name → median imputation → IQR outlier clipping      │
│           → StandardScaler → 70/15/15 split                              │
│           → save .pkl files + reference.parquet → Redis                  │
├──────────────────────────────────────────────────────────────────────────┤
│  Outputs: train.pkl, val.pkl, test.pkl, preprocessing_pipeline.pkl       │
│           feature_names.pkl, training_reference.parquet                  │
│  Redis: preprocessing_status = success, train_rows = 13184               │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 4 ── MODEL TRAINING                                               │
│  Tool: scikit-learn (LinearRegression, RandomForestRegressor),           │
│         XGBoost, MLflow (tracking), SHAP (explainability)                │
│  Script: src/train_model.py                                              │
│  Actions: Train 3 models → evaluate (R², RMSE, MAE)                      │
│           → log params/metrics/artifacts to MLflow                       │
│           → save best model → SHAP plots → Redis                         │
├──────────────────────────────────────────────────────────────────────────┤
│  Outputs: models/best_model.pkl, reports/shap_*.png                      │
│  MLflow: http://localhost:5000 (experiment: song_popularity_regression)  │
│  Redis: training_status = success, best_model_name = XGBoostRegressor    │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 5 ── MODEL DEPLOYMENT                                             │
│  Tool: FastAPI, Uvicorn, Pydantic, joblib                                │
│  Script: app/backend/main.py                                             │
│  Endpoints: GET /health → {"status": "healthy"}                          │
│             POST /predict → JSON features → {prediction, version, ms}    │
│  Container: song_fastapi_backend (port 8000)                             │
├──────────────────────────────────────────────────────────────────────────┤
│  Frontend: HTML + CSS + JavaScript (port 5500)                           │
│  Container: song_frontend (Nginx)                                        │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ ▲  LIFECYCLE MONITORING (NOT a DAG task)                                │
│ │  Tool: Evidently AI (DataDriftPreset)                                 │
│ │  Script: src/monitor_model.py                                          │
│ │  Scope: Compares current warehouse data against training reference     │
│ │         Detects feature drift, distribution shifts, quality issues     │
│ │         Generates HTML drift report (reports/drift/)                   │
│ │  Runs: Independently, on demand or scheduled, NOT inside Airflow      │
│ └────────────────────────────────────────────────────────────────────────│
│                                                                          │
│ ┌────────────────────────────────────────────────────────────────────────┐
│ │  ORCHESTRATION LAYER                                                   │
│ │  Tool: Apache Airflow (BashOperator, DAG)                             │
│ │  DAG: song_popularity_mlops_pipeline                                   │
│ │  Schedule: Weekly (Sunday 18:00)                                      │
│ │  Tasks: ingestion → validation → preprocessing → training → deploy    │
│ │  Data handoff: Redis between every consecutive task                    │
│ │  Containers: airflow-webserver (8080), airflow-scheduler, postgres     │
│ └────────────────────────────────────────────────────────────────────────┘
│                                                                          │
│ ┌────────────────────────────────────────────────────────────────────────┐
│ │  CONTAINERISATION LAYER                                                │
│ │  Tool: Docker + Docker Compose                                        │
│ │  Services: MariaDB (3307), Redis (6379), MLflow (5000),               │
│ │            FastAPI (8000), Frontend (5500)                             │
│ │  Separate: Airflow Docker Compose (webserver 8080)                    │
│ └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Warehouse | MariaDB ColumnStore | Columnar analytical storage for `songs_analytics` table |
| Intermediate Storage | Redis | Metadata handoff between Airflow pipeline stages |
| Analytical Cache | Apache Parquet | Local compressed columnar cache for processed data |
| ML Framework | scikit-learn, XGBoost | Model training and preprocessing pipeline |
| Experiment Tracking | MLflow | Log parameters, metrics, artefacts and model versions |
| Model Explainability | SHAP | Feature importance and prediction explanation |
| API Framework | FastAPI + Uvicorn | REST API for model predictions |
| Monitoring | Evidently AI | Data drift detection and quality monitoring |
| Orchestration | Apache Airflow | DAG-based workflow automation |
| Containerisation | Docker + Docker Compose | Reproducible service deployment |
| Frontend | HTML + CSS + JavaScript | Browser-based prediction interface |
| Language | Python 3.10+ | All pipeline scripts |
| Operating System | Ubuntu | Development and execution environment |

---

## 4. Folder Structure

```
song-popularity-project/
│
├── .dockerignore
├── .gitignore
├── README.md                              ← This file
├── requirements.txt                       ← Full Python dependencies
├── docker-compose.yml                     ← MariaDB + Redis + MLflow + FastAPI + Frontend
│
├── app/
│   ├── __init__.py
│   └── backend/
│       ├── __init__.py
│       ├── main.py                        ← FastAPI application with CORS
│       ├── Dockerfile                     ← FastAPI container build
│       └── requirements-backend.txt       ← Lightweight deps for Docker
│
├── airflow/
│   └── dags/
│       └── song_popularity_mlops_dag.py   ← Airflow DAG definition
│
├── airflow-docker/                        ← Separate Airflow Docker environment
│   ├── .env                               ← Airflow UID and credentials
│   ├── Dockerfile                         ← Custom Airflow image
│   ├── docker-compose.yml                 ← Airflow webserver + scheduler + postgres
│   ├── requirements.txt                   ← Python packages for Airflow container
│   ├── dags/
│   │   └── song_popularity_mlops_dag.py   ← Same DAG for Airflow
│   ├── logs/
│   ├── plugins/
│   └── config/
│
├── src/
│   ├── config.py                          ← MariaDB + Redis connection settings
│   ├── etl_ingestion.py                   ← Stage 1: CSV → MariaDB + Redis metadata
│   ├── validate_data.py                   ← Stage 2: Schema, type, null, range checks
│   ├── preprocess.py                      ← Stage 3: Impute, clip, scale, split
│   ├── train_model.py                     ← Stage 4: LR + RF + XGBoost + MLflow
│   ├── monitor_model.py                   ← Lifecycle monitoring with Evidently AI
│   └── retrain_model.py                   ← Convenience retraining script
│
├── frontend/
│   ├── index.html                         ← Prediction form UI
│   ├── style.css                          ← Dark theme styling
│   └── script.js                          ← API call to FastAPI /predict
│
├── data/
│   ├── raw/                               ← Place song_popularity.csv here
│   │   └── song_popularity.csv
│   ├── processed/                         ← Generated by preprocessing
│   │   ├── preprocessing_pipeline.pkl
│   │   ├── feature_names.pkl
│   │   ├── train.pkl
│   │   ├── val.pkl
│   │   └── test.pkl
│   └── reference/                         ← Training reference for monitoring
│       └── training_reference.parquet
│
├── models/
│   └── best_model.pkl                     ← Best trained model
│
├── reports/
│   └── drift/
│       └── song_popularity_drift_report.html  ← Evidently AI output
│
├── mlruns/                                ← MLflow experiment data
├── mlartifacts/                           ← MLflow model artefacts
└── notebooks/                             ← Optional Jupyter EDA notebooks
```

---

## 5. Prerequisites

### Operating System
- Ubuntu 20.04 or later (tested on Ubuntu)

### Required Software

| Software | Version | Installation |
|----------|---------|-------------|
| Python | 3.8+ | `sudo apt install python3 python3-venv python3-pip -y` |
| MariaDB | 10.5+ | See Step 1 below |
| Redis | 6+ | See Step 2 below |
| Docker | 24+ | `sudo apt install docker.io docker-compose-plugin -y` |
| Git | 2+ | `sudo apt install git -y` |
| curl | any | `sudo apt install curl -y` |

---

## 6. Step 1 — Install MariaDB ColumnStore

### Install MariaDB

```bash
sudo apt update
sudo apt install mariadb-server mariadb-client libmariadb-dev -y
```

### Start and enable MariaDB

```bash
sudo systemctl start mariadb
sudo systemctl enable mariadb
```

### Secure the installation

```bash
sudo mysql_secure_installation
```

Follow prompts to set root password.

### Create database and user

```bash
sudo mysql -u root -p
```

Inside the MySQL shell:

```sql
CREATE DATABASE song_popularity_db;
CREATE USER 'bibek'@'localhost' IDENTIFIED BY 'bibek123';
GRANT ALL PRIVILEGES ON song_popularity_db.* TO 'bibek'@'localhost';

-- Also allow connections from Docker containers
GRANT ALL PRIVILEGES ON song_popularity_db.* TO 'bibek'@'%' IDENTIFIED BY 'bibek123';

FLUSH PRIVILEGES;
EXIT;
```

### Allow remote connections (needed for Docker/Airflow)

```bash
sudo nano /etc/mysql/mariadb.conf.d/50-server.cnf
```

Find:

```
bind-address = 127.0.0.1
```

Change to:

```
bind-address = 0.0.0.0
```

Restart MariaDB:

```bash
sudo systemctl restart mariadb
```

### Verify

```bash
sudo mysql -u bibek -p song_popularity_db -e "SELECT 1 AS connection_test;"
```

Expected:

```
+-----------------+
| connection_test |
+-----------------+
|               1 |
+-----------------+
```

---

## 7. Step 2 — Install Redis

### Install Redis

```bash
sudo apt install redis-server -y
```

### Start and enable Redis

```bash
sudo systemctl start redis
sudo systemctl enable redis
```

### Verify

```bash
redis-cli ping
```

Expected:

```
PONG
```

---

## 8. Step 3 — Python Environment Setup

### Navigate to project folder

```bash
cd ~/Documents/song-popularity-project
```

### Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Upgrade pip

```bash
pip install --upgrade pip
```

### Install packages step by step (to avoid timeout issues)

```bash
# Core packages
pip install pandas numpy scikit-learn sqlalchemy pymysql redis joblib pyarrow

# FastAPI packages
pip install fastapi uvicorn pydantic email-validator

# ML packages
pip install xgboost

# Experiment tracking
pip install --default-timeout=300 mlflow

# Monitoring
pip install --default-timeout=300 evidently

# Explainability (optional — skip if it fails)
pip install shap || echo "SHAP skipped — not critical"
```

### Verify installation

```bash
python -c "import pandas, sklearn, sqlalchemy, redis, fastapi, joblib, xgboost; print('All core packages OK')"
python -c "import mlflow; print('MLflow OK')"
python -c "import evidently; print('Evidently OK')"
```

---

## 9. Step 4 — Place the Dataset

Download the Song Popularity Dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/yasserh/song-popularity-dataset

Place it in the `data/raw/` folder:

```bash
cp ~/Downloads/song_popularity.csv data/raw/
```

Verify:

```bash
ls data/raw/song_popularity.csv
```

Check the file:

```bash
head -5 data/raw/song_popularity.csv
wc -l data/raw/song_popularity.csv
```

Expected: approximately 18,836 lines (18,835 data rows + 1 header).

---

## 10. Step 5 — Run ETL Ingestion

### What this does

- Reads `data/raw/song_popularity.csv`
- Standardises column names to snake_case
- Validates expected headers
- Removes duplicate rows
- Loads clean data into MariaDB `songs_analytics` table
- Writes run metadata to `ingestion_log` table
- Creates Parquet cache in `data/processed/`
- Stores `ingestion_status`, `row_count`, `timestamp` in Redis

### Run

```bash
cd ~/Documents/song-popularity-project
source venv/bin/activate

python src/etl_ingestion.py
```

### Expected output

```
============================================================
ETL INGESTION COMPLETED SUCCESSFULLY
============================================================
  Rows loaded to MariaDB:  18835
  Columns:                 15
  Null values:             0
  Parquet cache:           data/processed/songs_analytics.parquet
  Redis key set:           ingestion_status = success
============================================================
```

### Verify in MariaDB

```bash
sudo mysql -u bibek -p song_popularity_db -e "SELECT COUNT(*) FROM songs_analytics;"
```

### Verify in Redis

```bash
redis-cli GET ingestion_status
```

Expected: `"success"`

---

## 11. Step 6 — Run Data Validation

### What this does

- Reads `ingestion_status` from Redis to confirm ingestion succeeded
- Reads `songs_analytics` from MariaDB
- Checks: schema, data types, null rates, target range (0–100), duplicates
- Stores `validation_status = passed` in Redis

### Run

```bash
python src/validate_data.py
```

### Expected output

```
============================================================
DATA VALIDATION COMPLETED SUCCESSFULLY
============================================================
  Rows validated:   18835
  Duplicates:       0
  Redis key set:    validation_status = passed
============================================================
```

---

## 12. Step 7 — Run Preprocessing

### What this does

- Reads `validation_status` from Redis to confirm validation passed
- Reads `songs_analytics` from MariaDB
- Removes `song_name` from features
- Applies median imputation, IQR outlier clipping and StandardScaler
- Splits data 70/15/15 into train/validation/test
- Saves preprocessing pipeline, feature names and data splits as `.pkl` files
- Saves training reference data for Evidently monitoring
- Stores `preprocessing_status = success` in Redis

### Run

```bash
python src/preprocess.py
```

### Expected output

```
============================================================
PREPROCESSING COMPLETED SUCCESSFULLY
============================================================
  Training rows:    13184
  Validation rows:  2825
  Test rows:        2826
  Features:         13
  Output path:      data/processed
  Reference data:   data/reference/training_reference.parquet
  Redis key set:    preprocessing_status = success
============================================================
```

### Verify files created

```bash
ls data/processed/
```

Expected:

```
feature_names.pkl
preprocessing_pipeline.pkl
songs_analytics.parquet
test.pkl
train.pkl
val.pkl
```

---

## 13. Step 8 — Start MLflow

MLflow tracks model parameters, metrics and artefacts.

### Open a NEW terminal

```bash
cd ~/Documents/song-popularity-project
source venv/bin/activate

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000
```

### Keep this terminal running

### Verify

Open browser:

```
http://localhost:5000
```

You should see the MLflow dashboard.

---

## 14. Step 9 — Train Models

### What this does

- Reads `preprocessing_status` from Redis
- Trains Linear Regression, Random Forest Regressor and XGBoost Regressor
- Evaluates using R², RMSE and MAE on validation and test sets
- Logs all parameters, metrics and artefacts to MLflow
- Saves the best model as `models/best_model.pkl`
- Stores `training_status`, `best_model_name`, `best_test_r2` in Redis

### Run (in the original terminal)

```bash
python src/train_model.py
```

### Expected output

```
  LinearRegression
    Val  R2=0.1234  RMSE=18.45  MAE=14.67
    Test R2=0.1198  RMSE=18.53  MAE=14.72

  RandomForestRegressor
    Val  R2=0.5678  RMSE=12.34  MAE=9.56
    Test R2=0.5543  RMSE=12.48  MAE=9.61

  XGBoostRegressor
    Val  R2=0.6234  RMSE=11.56  MAE=8.92
    Test R2=0.6123  RMSE=11.72  MAE=8.98

============================================================
MODEL TRAINING COMPLETED SUCCESSFULLY
============================================================
  Best model:   XGBoostRegressor
  Test R2:      0.6123
  Saved to:     models/best_model.pkl
  Redis key:    training_status = success
============================================================
```

> Note: Your exact values will differ.

### Verify

```bash
ls models/best_model.pkl
redis-cli GET best_model_name
redis-cli GET best_test_r2
```

### View in MLflow

Open: `http://localhost:5000`

Click on the `song_popularity_regression` experiment to see all logged runs.

---

## 15. Step 10 — Model Explainability (SHAP)

SHAP generates feature importance plots to explain which audio features most influence predicted popularity.

If your `train_model.py` includes SHAP code, the plots are generated during training.

If you have a separate explainability script:

```bash
python src/explain_model.py
```

Expected artefacts:

```
reports/shap_summary.png
reports/shap_feature_importance.png
```

---

## 16. Step 11 — Run Monitoring (Lifecycle Activity)

### IMPORTANT: Monitoring is NOT a DAG task

Monitoring is an ongoing lifecycle activity that spans the entire system. It is deliberately excluded from the Airflow DAG.

### What this does

- Reads `songs_analytics` from MariaDB
- Loads training reference data from `data/reference/training_reference.parquet`
- Compares reference vs current data using Evidently AI
- Generates an HTML drift report

### Run

```bash
python src/monitor_model.py
```

### Expected output

```
Evidently AI drift report saved: reports/drift/song_popularity_drift_report.html
NOTE: Monitoring is a lifecycle activity, NOT a DAG task.
```

### View the report

```bash
xdg-open reports/drift/song_popularity_drift_report.html
```

Or open the file in any browser.

---

## 17. Step 12 — Start FastAPI Backend

### What this does

- Loads `models/best_model.pkl` and `data/processed/preprocessing_pipeline.pkl`
- Exposes `GET /health` and `POST /predict` endpoints
- Validates input using Pydantic
- Returns predicted popularity, model version and inference latency

### Run

```bash
uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Verify

Open: `http://localhost:8000/docs`

#### Test /health

Click `GET /health` → Try it out → Execute

Expected:

```json
{"status": "healthy"}
```

#### Test /predict

Click `POST /predict` → Try it out → Paste this JSON:

```json
{
  "acousticness": 0.18,
  "danceability": 0.72,
  "energy": 0.81,
  "instrumentalness": 0.02,
  "key": 5,
  "liveness": 0.12,
  "loudness": -5.8,
  "audio_mode": 1,
  "speechiness": 0.07,
  "tempo": 124.5,
  "time_signature": 4,
  "audio_valence": 0.64,
  "song_duration_ms": 215000
}
```

Expected response:

```json
{
  "predicted_popularity": 70.25,
  "model_version": "local-best-model-v1",
  "latency_ms": 15.32
}
```

> Your exact prediction value will differ.

---

## 18. Step 13 — Start Frontend

### Open a NEW terminal

```bash
cd ~/Documents/song-popularity-project/frontend
python3 -m http.server 5500
```

### Open in browser

```
http://localhost:5500
```

### Use the frontend

1. Enter Spotify audio feature values in the form
2. Click **Predict Popularity**
3. See the predicted popularity score, model version and latency

---

## 19. Step 14 — Docker Compose Deployment

Docker Compose runs all 5 services in containers.

### docker-compose.yml services

| Service | Container Name | Port | Purpose |
|---------|---------------|------|---------|
| MariaDB | song_mariadb | 3307 | Columnar data warehouse |
| Redis | song_redis | 6379 | Intermediate metadata storage |
| MLflow | song_mlflow | 5000 | Experiment tracking |
| FastAPI | song_fastapi_backend | 8000 | Prediction API |
| Frontend | song_frontend | 5500 | Nginx serving prediction UI |

### Before starting — kill processes using the same ports

```bash
sudo kill -9 $(sudo lsof -t -i:3307 -i:5000 -i:6379 -i:8000 -i:5500) 2>/dev/null
```

### Start all services

```bash
cd ~/Documents/song-popularity-project
docker compose up -d
```

### Verify

```bash
docker ps
```

Expected:

```
song_mariadb          → 3307
song_redis            → 6379
song_mlflow           → 5000
song_fastapi_backend  → 8000
song_frontend         → 5500
```

### Test

```
http://localhost:5500       ← Frontend
http://localhost:8000/docs  ← FastAPI Swagger UI
http://localhost:5000       ← MLflow Dashboard
```

### Stop all services

```bash
docker compose down
```

---

## 20. Step 15 — Apache Airflow Setup and DAG Execution

Airflow orchestrates the batch pipeline as a Directed Acyclic Graph (DAG).

### DAG Tasks (monitoring is NOT included)

```
data_ingestion → data_validation → preprocessing → model_training → model_deployment
```

### Step 15.1: Navigate to Airflow folder

```bash
cd ~/Documents/song-popularity-project/airflow-docker
```

### Step 15.2: Ensure required folders exist

```bash
mkdir -p dags logs plugins config
```

### Step 15.3: Ensure DAG file exists

```bash
ls dags/song_popularity_mlops_dag.py
```

If missing:

```bash
cp ../airflow/dags/song_popularity_mlops_dag.py dags/
```

### Step 15.4: Ensure .env file exists

```bash
cat .env
```

Should show:

```
AIRFLOW_UID=1000
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
```

If missing:

```bash
echo -e "AIRFLOW_UID=$(id -u)\n_AIRFLOW_WWW_USER_USERNAME=airflow\n_AIRFLOW_WWW_USER_PASSWORD=airflow" > .env
```

### Step 15.5: Fix log permissions

```bash
sudo chmod -R 777 logs
```

### Step 15.6: Initialize Airflow (first time only)

```bash
docker compose up airflow-init
```

Wait until:

```
airflow_init exited with code 0
```

### Step 15.7: Start Airflow services

```bash
docker compose up -d
```

### Step 15.8: Check containers

```bash
docker ps
```

Expected:

```
airflow_webserver   → 8080
airflow_scheduler
airflow_postgres
```

### Step 15.9: Open Airflow UI

```
http://localhost:8080
```

Login:

```
Username: airflow
Password: airflow
```

### Step 15.10: Find and trigger DAG

1. Search for `song_popularity_mlops_pipeline`
2. Click the toggle switch to turn it **ON**
3. Click the **play button ▶** to trigger the DAG

### Step 15.11: Watch the Graph View

Click the DAG name → **Graph** tab

Each task changes colour:

| Colour | Meaning |
|--------|---------|
| Grey | Not started |
| Yellow | Running |
| Green | Success ✅ |
| Red | Failed ❌ |

When all 5 tasks are **green**, the DAG run is successful.

### Step 15.12: View task logs

Click any task → **Logs** tab → see the full terminal output.

### Stop Airflow

```bash
docker compose down
```

---

## 21. Verify Redis Keys

After running the pipeline, check all Redis keys:

```bash
redis-cli
```

Then type each command:

```
GET ingestion_status
GET ingestion_row_count
GET ingestion_timestamp
GET validation_status
GET validation_row_count
GET preprocessing_status
GET processed_data_path
GET train_rows
GET val_rows
GET test_rows
GET feature_count
GET training_status
GET best_model_name
GET best_model_path
GET best_test_r2
```

Expected values:

| Key | Example Value |
|-----|--------------|
| `ingestion_status` | `success` |
| `ingestion_row_count` | `18835` |
| `validation_status` | `passed` |
| `preprocessing_status` | `success` |
| `train_rows` | `13184` |
| `training_status` | `success` |
| `best_model_name` | `XGBoostRegressor` |
| `best_test_r2` | `0.6123` |

---

## 22. Verify MariaDB Tables

```bash
sudo mysql -u bibek -p song_popularity_db
```

### Check songs_analytics table

```sql
SELECT COUNT(*) AS total_rows FROM songs_analytics;
SELECT * FROM songs_analytics LIMIT 5;
```

### Check ingestion_log table

```sql
SELECT * FROM ingestion_log;
```

### Check column names

```sql
DESCRIBE songs_analytics;
```

```sql
EXIT;
```

---

## 23. All Browser URLs

| URL | Service | Purpose |
|-----|---------|---------|
| `http://localhost:5500` | Frontend | Song popularity prediction UI |
| `http://localhost:8000/docs` | FastAPI | Swagger API documentation |
| `http://localhost:8000/health` | FastAPI | Health check endpoint |
| `http://localhost:5000` | MLflow | Experiment tracking dashboard |
| `http://localhost:8080` | Airflow | DAG orchestration UI |

---

## 24. Screenshots Guide for Report

Take these screenshots in order for your report:

### Infrastructure Evidence

| # | What to Screenshot | Where | Report Figure |
|---|-------------------|-------|---------------|
| 1 | `docker ps` showing 5 service containers | Terminal | Docker deployment evidence |
| 2 | MariaDB `SELECT COUNT(*) FROM songs_analytics` | Terminal | Database evidence |
| 3 | MariaDB `SELECT * FROM ingestion_log` | Terminal | Ingestion log evidence |
| 4 | Redis keys showing pipeline metadata | Terminal | Redis metadata evidence |

### Pipeline Stage Evidence

| # | What to Screenshot | Where | Report Figure |
|---|-------------------|-------|---------------|
| 5 | Successful `etl_ingestion.py` output | Terminal | Ingestion evidence |
| 6 | Successful `validate_data.py` output | Terminal | Validation evidence |
| 7 | Successful `preprocess.py` output | Terminal | Preprocessing evidence |
| 8 | `ls data/processed/` showing pkl files | Terminal | Processed files evidence |
| 9 | `ls models/best_model.pkl` | Terminal | Model file evidence |
| 10 | Successful `train_model.py` output | Terminal | Training evidence |

### Tool Evidence

| # | What to Screenshot | Where | Report Figure |
|---|-------------------|-------|---------------|
| 11 | MLflow experiment dashboard with runs | `localhost:5000` | MLflow evidence |
| 12 | FastAPI Swagger UI showing endpoints | `localhost:8000/docs` | FastAPI evidence |
| 13 | FastAPI `/predict` JSON response | `localhost:8000/docs` | Prediction evidence |
| 14 | Frontend with prediction result | `localhost:5500` | Frontend evidence |
| 15 | SHAP summary plot | Generated image | Explainability evidence |
| 16 | SHAP feature importance plot | Generated image | Explainability evidence |

### Monitoring Evidence

| # | What to Screenshot | Where | Report Figure |
|---|-------------------|-------|---------------|
| 17 | Evidently AI drift report | HTML file in browser | Monitoring evidence |

### Airflow Evidence

| # | What to Screenshot | Where | Report Figure |
|---|-------------------|-------|---------------|
| 18 | Airflow DAG list page | `localhost:8080` | Airflow evidence |
| 19 | DAG Graph View — all tasks green | `localhost:8080` | Successful DAG evidence |
| 20 | Task log for data_ingestion | Click task → Logs | Task log evidence |
| 21 | Task log for model_training | Click task → Logs | Task log evidence |

### Data Analysis Evidence

| # | What to Screenshot | Where | Report Figure |
|---|-------------------|-------|---------------|
| 22 | Song Popularity by Energy Level (boxplot) | Colab / Jupyter | RQ1 |
| 23 | Danceability by Popularity Level (boxplot) | Colab / Jupyter | RQ2 |
| 24 | Musical Key vs Popularity Level (bar chart) | Colab / Jupyter | RQ3 |
| 25 | Correlation Heatmap | Colab / Jupyter | Correlation Analysis |

---

## 25. Troubleshooting

### Port already in use

```bash
# Kill all conflicting processes
sudo kill -9 $(sudo lsof -t -i:3306 -i:3307 -i:5000 -i:6379 -i:8000 -i:5500 -i:8080) 2>/dev/null
```

### Docker build timeout (slow internet)

Use lightweight `requirements-backend.txt` for FastAPI Dockerfile:

```
fastapi
uvicorn
pydantic
email-validator
joblib
numpy
scikit-learn
pandas
pyarrow
xgboost
```

### CORS error in frontend

Ensure `app/backend/main.py` has:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### ModuleNotFoundError

```bash
source venv/bin/activate
pip install <missing_package>
```

### Airflow DAG not showing

```bash
cd ~/Documents/song-popularity-project/airflow-docker
docker compose restart airflow-scheduler
```

Wait 30 seconds and refresh the browser.

### MariaDB connection refused from Docker

Ensure `bind-address = 0.0.0.0` in `/etc/mysql/mariadb.conf.d/50-server.cnf` and restart MariaDB.

### Redis connection refused from Docker

Make sure `src/config.py` auto-detects Docker:

```python
import os
IS_DOCKER = os.path.exists("/.dockerenv")
_HOST = "host.docker.internal" if IS_DOCKER else "localhost"
```

### Model file not found

Run training first:

```bash
python src/train_model.py
```

### Frontend says "Failed to fetch"

1. Ensure FastAPI is running on port 8000
2. Ensure CORS is configured with `allow_origins=["*"]`
3. Ensure `frontend/script.js` has `const API_URL = "http://127.0.0.1:8000/predict"`

---

## 26. Viva Questions and Answers

### Q: Why did you select the Song Popularity Dataset?

> Real-world impact was the primary driver. Music streaming platforms (Spotify, Apple Music) serve millions of users daily and rely on accurate popularity signals for playlist curation, artist discovery and recommendation algorithms. Record labels invest significantly in identifying high-potential tracks before release. This dataset enables a practical MLOps pipeline that mirrors a genuine industry use case — predicting song popularity from audio features. The 18,835 rows and 13 numeric features are sufficient to demonstrate the full pipeline while keeping iteration fast during development.

### Q: Why MariaDB ColumnStore?

> MariaDB ColumnStore was selected because the pipeline performs column-based analytical reads over numeric audio features. A columnar engine stores each column adjacently on disk, so queries like `AVG(danceability)`, `MAX(energy)` or correlation analysis scan only the required columns rather than full rows. This is fundamentally more efficient than row-based MySQL or PostgreSQL for this analytical workload. ColumnStore also supports standard SQL syntax, so no specialised query language is needed.

### Q: Why Redis between stages?

> Redis serves as lightweight in-memory metadata storage between Airflow tasks. Each pipeline stage writes a status flag, row count and relevant metrics to Redis. The next stage reads these before executing, ensuring it only runs after the previous stage completed successfully. This decouples the tasks — Airflow does not need to pass large DataFrames through XCom, and the pipeline remains traceable even if a task fails mid-way.

### Q: Why OBT instead of Star Schema?

> The dataset contains a single entity — the song — with no separate artist, album, genre or playlist dimension tables. A star schema would introduce joins on tables that do not exist, adding complexity with zero analytical benefit. One Big Table keeps queries simple, scan-efficient for the columnar engine, and matches the flat CSV structure of the source data.

### Q: Why ETL instead of ELT?

> The raw CSV may contain null values in numeric columns, duplicate rows, and out-of-range target values (song_popularity should be 0–100). ETL validates and cleans this data _before_ loading into MariaDB, ensuring the warehouse only ever contains consistent, analysis-ready data. ELT would load raw data first and clean it later, which risks downstream consumers reading invalid data and contradicts the principle of trustworthy analytical storage.

### Q: Why is monitoring not in the DAG?

> Monitoring is an ongoing lifecycle activity, not a discrete build step. It must observe data quality across ingestion, feature distributions after preprocessing, prediction distributions from the API, and drift against the training reference — continuously and independently. Adding it to the DAG would imply it runs once per pipeline execution and stops, which defeats its purpose. Evidently AI runs independently, on demand or on its own schedule, and is deliberately excluded from the Airflow DAG.

### Q: How does the Airflow DAG work?

> Each task runs a Python script using BashOperator. The dependency chain ensures each task runs only after the previous one succeeds. Redis stores metadata between tasks so each stage can verify the previous stage completed.

### Q: What models did you compare?

> Linear Regression as a baseline, Random Forest Regressor and XGBoost Regressor. Tree-based models captured non-linear relationships better. The best model was selected based on R² score on unseen test data.

### Q: How does FastAPI serve predictions?

> FastAPI loads the trained model and preprocessing pipeline. When a POST request is sent to /predict with audio features as JSON, it validates the input using Pydantic, applies the preprocessing pipeline, runs the model prediction and returns the predicted popularity, model version and latency.

### Q: What does Evidently AI do?

> Evidently AI compares reference training data against current or simulated production data to detect data drift and feature distribution changes. It generates an HTML report showing which features have drifted and by how much.

---

## 27. Presentation Flow for Supervisor

### Demo Day — Terminal Setup

```
Terminal 1: sudo systemctl start mariadb && sudo systemctl start redis
Terminal 2: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
Terminal 3: uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
Terminal 4: cd frontend && python3 -m http.server 5500
Terminal 5: cd airflow-docker && docker compose up -d
```

### Browser Tabs to Open

```
Tab 1: http://localhost:5000       ← MLflow
Tab 2: http://localhost:8000/docs  ← FastAPI
Tab 3: http://localhost:5500       ← Frontend
Tab 4: http://localhost:8080       ← Airflow
```

### Presentation Order

| Step | What to Show | What to Say |
|------|-------------|-------------|
| 1 | Architecture diagram | "This is the end-to-end MLOps architecture for song popularity prediction." |
| 2 | Dataset description | "The Song Popularity Dataset was selected for its real-world value in music analytics." |
| 3 | MariaDB query | "MariaDB ColumnStore stores the validated songs_analytics table." |
| 4 | Redis keys | "Redis stores metadata between Airflow stages." |
| 5 | ETL terminal output | "ETL ingestion cleans and loads data into the warehouse." |
| 6 | Validation output | "Validation confirms schema, types, nulls and target range." |
| 7 | Preprocessing output | "Preprocessing creates train/val/test splits with imputation and scaling." |
| 8 | MLflow dashboard | "MLflow tracks all experiments, parameters and metrics." |
| 9 | Training output | "Three models were compared. The best model was saved." |
| 10 | SHAP plots | "SHAP explains which audio features most influence predictions." |
| 11 | FastAPI Swagger | "The model is deployed as a REST API with /predict and /health." |
| 12 | FastAPI /predict | "Real-time predictions with audio features return popularity scores." |
| 13 | Frontend | "Users can enter features and get predictions from the browser." |
| 14 | Evidently report | "Evidently AI monitors data drift as a lifecycle activity." |
| 15 | Airflow Graph View | "Airflow orchestrates the batch pipeline. Monitoring is not a DAG task." |
| 16 | docker ps | "Docker Compose runs all services in containers." |
| 17 | Reflection | "Main challenges were service integration, Docker networking and Redis communication." |

---

## 28. Personal Reflection

### Challenges Encountered

The first major challenge was **Docker networking** between containers and host services. The FastAPI backend container needed to connect to MariaDB running on the host (during development) and also to MariaDB running in a container (during Docker Compose deployment). This required implementing conditional host resolution in `src/config.py` that checks for `/.dockerenv` to switch between `localhost` and `host.docker.internal`.

**Service integration** posed another difficulty. Each pipeline stage — ingestion, validation, preprocessing, training — had to pass metadata through Redis while remaining independently executable. Ensuring that Redis keys from one stage were correctly set and read by the next required careful coordination of key naming conventions and write-then-read sequencing.

**Evidently AI** was not always available in the Python environment due to dependency conflicts with XGBoost and MLflow. I implemented a fallback path in `monitor_model.py` that generates a pandas describe HTML table when Evidently cannot be imported, ensuring the monitoring step never blocks the pipeline.

Column name inconsistencies between the Kaggle CSV, the MariaDB schema and the preprocessing pipeline caused silent failures during early development. I addressed this by centralising the expected column list in `src/config.py` so that all scripts reference a single source of truth.

### Lessons Learned

Designing for **failure tolerance** was the most important lesson. Every pipeline stage now checks Redis for the previous stage's status before executing, and the monitoring script gracefully degrades to a fallback report when Evidently is unavailable. This taught me that MLOps pipelines must handle partial failures without cascading.

The columnar storage advantage of **MariaDB ColumnStore** became clear when comparing query performance. Aggregate queries like `AVG(energy)` across 18,835 rows returned in milliseconds because only the `energy` column was scanned, not entire rows. This validated the architectural choice of columnar storage over row-based databases.

Using **Redis as a metadata bus** between Airflow tasks proved far more practical than Airflow's native XCom system. XCom is designed for small payloads and becomes unwieldy with larger metadata. Redis provided sub-millisecond reads and writes with simple key-value semantics, making the pipeline both faster and more maintainable.

### How Issues Were Identified and Resolved

Issues were identified through a combination of:
- **Print-debugging** each script during standalone execution before integrating into the Airflow DAG
- **Redis key inspection** using `redis-cli GET` to verify metadata was being written correctly between stages
- **Airflow task logs** which captured stdout and stderr from each BashOperator task, making it easy to trace which stage failed and why
- **Docker logs** (`docker logs <container>`) for diagnosing container startup failures and network connectivity issues

### Improvements Made Throughout the Project

1. **Early version**: All logic was in a single monolithic script. This was refactored into separate stage scripts (`etl_ingestion.py`, `validate_data.py`, `preprocess.py`, `train_model.py`) to enable independent execution and testing.
2. **Column config centralisation**: Column names were initially hardcoded in each script. They were moved to `src/config.py` as `EXPECTED_COLUMNS` to eliminate duplication and prevent schema drift.
3. **Monitoring fallback**: The monitoring script was enhanced to degrade gracefully when Evidently AI is unavailable, ensuring the pipeline never breaks due to a missing optional dependency.
4. **Docker host detection**: The `IS_DOCKER` flag in `config.py` was added after repeated connection failures between containers, solving the host resolution problem once and for all.
5. **Parquet cache**: Initially the pipeline re-read the CSV from MariaDB for every stage. Adding a Parquet cache in `data/processed/` reduced read times significantly and provided a portable backup of the cleaned dataset.

---

## 29. References

- Géron, A. (2022) *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.* 3rd edn. Sebastopol: O'Reilly Media.
- Kimball, R. and Ross, M. (2013) *The Data Warehouse Toolkit.* 3rd edn. Indianapolis: John Wiley & Sons.
- Kreuzberger, D., Kühl, N. and Hirschl, S. (2023) 'Machine learning operations (MLOps): Overview, definition, and architecture', *IEEE Access*, 11, pp. 31866–31879.
- Sculley, D. et al. (2015) 'Hidden technical debt in machine learning systems', *Advances in Neural Information Processing Systems*, 28, pp. 2503–2511.
- Yasserh (2022) *Song Popularity Dataset* [Kaggle dataset]. Available at: https://www.kaggle.com/datasets/yasserh/song-popularity-dataset (Accessed: 20 October 2024).

---

> **Project by Bibek Shah (23189619) — Birmingham City University — CMP6230 — May 2026**