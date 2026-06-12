# Presentation Guide — Song Popularity MLOps Pipeline

> **Student:** Bibek Shah | **ID:** 23189619
> Use this guide during your supervisor demo. Each step has what to **show** and what to **say**.

---

## Before the Demo (Preparation)

### Start all services in order:

```bash
# Terminal 1: Activate environment
cd ~/Documents/song-popularity-project
source venv/bin/activate

# Terminal 2: Start MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000

# Terminal 3: Start FastAPI
uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 4: Start Frontend
cd frontend && python3 -m http.server 5500

# Terminal 5: Start Airflow (if showing)
cd ~/Documents/song-popularity-project/airflow-docker && docker compose up -d
```

### Open browser tabs ready:

| Tab | URL | What |
|-----|-----|------|
| 1 | `http://localhost:5000` | MLflow Dashboard |
| 2 | `http://localhost:8000/docs` | FastAPI Swagger |
| 3 | `http://localhost:5500` | Frontend UI |
| 4 | `http://localhost:8080` | Airflow (login: airflow/airflow) |

---

## Presentation Script (18 Steps)

### Step 1 — Architecture Overview (30 sec)

> **Show:** README architecture diagram (Section 2)
> **Say:** *"This is the end-to-end MLOps architecture. Data flows from a CSV file through ETL ingestion into MariaDB ColumnStore, then through validation, preprocessing, model training with MLflow tracking, and finally deployment as a FastAPI REST API with a browser frontend. Redis handles metadata handoff between Airflow stages. Evidently AI monitors drift as a lifecycle activity — not a DAG task."*

**Key points to hit:**
- ETL (not ELT) — clean before loading
- MariaDB ColumnStore — columnar for analytical queries
- Redis — decouples Airflow tasks
- Monitoring = lifecycle, not DAG

---

### Step 2 — Dataset Justification (30 sec)

> **Show:** Dataset description in README
> **Say:** *"I selected the Song Popularity Dataset because of its real-world impact. Streaming platforms like Spotify need accurate popularity predictions for playlist curation and recommendations. Record labels use this data for promotional decisions. The dataset has 18,835 songs with 13 numeric audio features — enough to demonstrate a full MLOps pipeline without being too large for rapid iteration."*

---

### Step 3 — ETL Ingestion (45 sec)

> **Show:** Terminal → Run or show cached output of `python src/etl_ingestion.py`

```bash
source venv/bin/activate
python src/etl_ingestion.py
```

> **Say:** *"ETL ingestion reads the raw CSV, standardises column names to snake_case, removes duplicate rows, and loads clean data into the MariaDB ColumnStore warehouse. It also writes a Parquet cache for faster subsequent reads and stores metadata in Redis — including ingestion status, row count, and timestamp."*

**Then show MariaDB:**
```bash
mysql -u bibek -pbibek123 song_popularity_db -e "SELECT COUNT(*) FROM songs_analytics;"
```

**Then show Redis:**
```bash
redis-cli GET ingestion_status
redis-cli GET ingestion_row_count
```

---

### Step 4 — Data Validation (30 sec)

> **Show:** Terminal → `python src/validate_data.py`
> **Say:** *"Validation confirms schema correctness, data types, null rates below 30%, and that song_popularity is within the 0–100 range. If any check fails, Redis gets validation_status=failed and the pipeline stops."*

**Show:**
```bash
redis-cli GET validation_status
```

---

### Step 5 — Preprocessing (30 sec)

> **Show:** Terminal → `python src/preprocess.py`
> **Say:** *"Preprocessing removes the song_name column, applies IQR outlier clipping, median imputation for any missing values, and StandardScaler normalisation. It splits data 70/15/15 into train, validation, and test sets, saving them as pickle files."*

**Show output files:**
```bash
ls data/processed/
```

---

### Step 6 — MLflow Experiment Tracking (45 sec)

> **Show:** Browser → `http://localhost:5000`
> **Say:** *"MLflow tracks all experiments. Click on the song_popularity_regression experiment to see three runs: Linear Regression, Random Forest, and XGBoost. Each run logs parameters like n_estimators, metrics like R² and RMSE, and the model artifact itself."*

**Click through:** Experiment → Runs → Compare metrics

---

### Step 7 — Model Training (45 sec)

> **Show:** Terminal → `python src/train_model.py` (or show cached output)
> **Say:** *"Three models are trained and compared: Linear Regression as a baseline, Random Forest Regressor, and XGBoost Regressor. The best model is selected based on test R² score. XGBoost typically performs best because it captures non-linear relationships in the audio features."*

**Show Redis:**
```bash
redis-cli GET best_model_name
redis-cli GET best_test_r2
```

---

### Step 8 — SHAP Explainability (30 sec)

> **Show:** SHAP plots (if generated) or mention MLflow artifacts
> **Say:** *"SHAP explains which audio features most influence predictions. Loudness, energy, and danceability are typically the top contributors. This provides interpretability — we can explain not just what the model predicts, but why."*

---

### Step 9 — MariaDB ColumnStore (30 sec)

> **Show:** Terminal
```bash
mysql -u bibek -pbibek123 song_popularity_db -e "DESCRIBE songs_analytics;"
mysql -u bibek -pbibek123 song_popularity_db -e "SELECT * FROM ingestion_log;"
```
> **Say:** *"MariaDB ColumnStore stores columns adjacently on disk. For analytical queries like AVG(energy), only the energy column is scanned, not entire rows — making it much faster than row-based MySQL for this workload."*

---

### Step 10 — Redis Metadata (30 sec)

> **Show:** Terminal
```bash
redis-cli GET ingestion_status
redis-cli GET validation_status
redis-cli GET preprocessing_status
redis-cli GET training_status
redis-cli GET best_model_name
redis-cli GET best_test_r2
```
> **Say:** *"Redis stores lightweight metadata between every pipeline stage. Each task reads the previous stage's status before executing, keeping stages decoupled without passing large DataFrames through Airflow's XCom."*

---

### Step 11 — FastAPI Deployment (45 sec)

> **Show:** Browser → `http://localhost:8000/docs`
> **Say:** *"The best model is deployed as a FastAPI REST API with two endpoints. GET /health returns the service status. POST /predict accepts 13 audio features as JSON, applies the preprocessing pipeline, runs the model, and returns the predicted popularity score with model version and latency."*

**Test /predict in Swagger UI:**
```json
{
  "acousticness": 0.18, "danceability": 0.72, "energy": 0.81,
  "instrumentalness": 0.02, "key": 5, "liveness": 0.12,
  "loudness": -5.8, "audio_mode": 1, "speechiness": 0.07,
  "tempo": 124.5, "time_signature": 4, "audio_valence": 0.64,
  "song_duration_ms": 215000
}
```

**Show response:** `{"predicted_popularity": 48.7, "model_version": "local-best-model-v1", "latency_ms": 3.78}`

---

### Step 12 — Frontend UI (30 sec)

> **Show:** Browser → `http://localhost:5500`
> **Say:** *"The frontend provides a dark-themed form where users can enter Spotify audio features. Clicking Predict Popularity sends the data to the FastAPI backend and displays the result. This makes the model accessible to non-technical users."*

**Action:** Fill in some values → Click Predict → Show the result

---

### Step 13 — Evidently AI Monitoring (30 sec)

> **Show:** Browser → Open `reports/drift/song_popularity_drift_report.html`
> **Say:** *"Evidently AI compares current warehouse data against the training reference to detect data drift. If feature distributions shift over time — for example, if the average danceability of incoming songs changes — the model's accuracy may degrade. This is a lifecycle activity, not a DAG task, because monitoring must run continuously."*

**Emphasise:** "Monitoring is NOT in the DAG"

---

### Step 14 — Airflow Orchestration (45 sec)

> **Show:** Browser → `http://localhost:8080` → DAG → Graph View
> **Say:** *"Apache Airflow orchestrates the batch pipeline as a DAG with 5 tasks: ingestion → validation → preprocessing → training → deployment. Each task runs a Python script via BashOperator. Redis handles metadata handoff between tasks. The DAG runs weekly, but can be triggered manually."*

**Action:** Show the DAG toggle → Trigger a run → Show Graph View with green tasks

---

### Step 15 — Docker Compose (30 sec)

> **Show:** Terminal → `docker ps`
> **Say:** *"Docker Compose runs all 5 services in containers: MariaDB (port 3307), Redis (6379), MLflow (5000), FastAPI backend (8000), and Nginx frontend (5500). Airflow has its own separate Docker Compose with Postgres, webserver, and scheduler."*

**Show all 8 containers running.**

---

### Step 16 — Design Decisions (45 sec)

> **Say** (no screen needed):
> - *"OBT instead of Star Schema — the dataset has one entity (song), no dimension tables. Star schema would add unnecessary joins."*
> - *"ETL instead of ELT — validate and clean before loading prevents bad data in the warehouse."*
> - *"MariaDB ColumnStore — columnar storage is optimised for analytical scans over numeric features."*
> - *"Redis between stages — lightweight metadata handoff keeps Airflow tasks decoupled."*

---

### Step 17 — Personal Reflection (30 sec)

> **Say:**
> - *"Main challenges were Docker networking between containers and host services, and Evidently AI API changes between versions."*
> - *"Key lesson: design for failure tolerance. Every stage checks Redis for the previous stage's status. Monitoring degrades gracefully with a fallback report."*
> - *"I improved the project by centralising column config in config.py, adding a Parquet cache for faster reads, and implementing Docker host detection for seamless local/container switching."*

---

### Step 18 — EDA Notebook (optional, 30 sec)

> **Show:** Jupyter notebook or Colab
> **Say:** *"I also created an EDA notebook with 10 visualisations exploring relationships between audio features and popularity. Key findings: loudness and energy have the strongest correlation with popularity, and higher-energy songs tend to be more popular."*

---

## Quick Reference — Terminal Commands

```bash
source venv/bin/activate

# Pipeline
python src/etl_ingestion.py
python src/validate_data.py
python src/preprocess.py
python src/train_model.py
python src/monitor_model.py

# Verify
mysql -u bibek -pbibek123 song_popularity_db -e "SELECT COUNT(*) FROM songs_analytics;"
redis-cli GET ingestion_status
redis-cli GET best_model_name

# Start services
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
cd frontend && python3 -m http.server 5500

# Airflow
cd airflow-docker && docker compose up -d
```

## All URLs

| URL | Service |
|-----|---------|
| `http://localhost:5500` | Frontend UI |
| `http://localhost:8000/docs` | FastAPI Swagger |
| `http://localhost:5000` | MLflow Dashboard |
| `http://localhost:8080` | Airflow (airflow/airflow) |
