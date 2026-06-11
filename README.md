# Song Popularity MLOps — Revised Implementation

## Architecture (Based on Supervisor Comments)
| Component | Technology | Purpose |
|---|---|---|
| Data Warehouse | **MariaDB ColumnStore** | Columnar analytical warehouse for `songs_analytics` |
| Intermediate Storage | **Redis** | Metadata handoff between Airflow stages |
| Experiment Tracking | **MLflow** | Parameters, metrics, model artefacts |
| API | **FastAPI** | `/predict` and `/health` endpoints |
| Containerisation | **Docker + Docker Compose** | Reproducible services |
| Orchestration | **Apache Airflow** | DAG: ingestion → validation → preprocessing → training → deployment |
| Monitoring | **Evidently AI** | Lifecycle drift detection (NOT a DAG task) |
| Frontend | **HTML/CSS/JS** | localhost:5500 prediction UI |
| Cache | **Parquet** | Local analytical cache |

## Quick Start

### 1. Prerequisites
```bash
sudo apt install mariadb-server redis-server docker.io docker-compose-plugin -y
sudo systemctl start mariadb redis docker
```

### 2. MariaDB Setup
```bash
sudo mysql -u root -p
```
```sql
CREATE DATABASE song_popularity_db;
CREATE USER 'bibek'@'localhost' IDENTIFIED BY 'bibek123';
GRANT ALL PRIVILEGES ON song_popularity_db.* TO 'bibek'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

### 3. Python Environment
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 4. Place Dataset
```bash
cp ~/Downloads/song_popularity.csv data/raw/
```

### 5. Run Pipeline Manually
```bash
python src/etl_ingestion.py
python src/validate_data.py
python src/preprocess.py
```

### 6. Start MLflow (new terminal)
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
```

### 7. Train Model
```bash
python src/train_model.py
```

### 8. Monitoring (separate — NOT in DAG)
```bash
python src/monitor_model.py
```

### 9. Start FastAPI
```bash
uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 10. Start Frontend
```bash
cd frontend && python3 -m http.server 5500
```

### 11. Docker Services
```bash
docker compose up -d
```

### 12. Airflow
```bash
cd airflow-docker
docker compose up airflow-init
docker compose up -d
# Open http://localhost:8080  (airflow / airflow)
```

## Verify Redis
```bash
redis-cli
GET ingestion_status
GET validation_status
GET preprocessing_status
GET training_status
GET best_model_name
GET best_test_r2
```

## Verify MariaDB
```bash
sudo mysql -u bibek -p song_popularity_db -e "SELECT COUNT(*) FROM songs_analytics;"
```

## Airflow DAG (monitoring NOT included)
```
data_ingestion → data_validation → preprocessing → model_training → model_deployment
```
