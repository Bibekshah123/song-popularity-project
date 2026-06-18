"""
Project-wide configuration for the Song Popularity MLOps pipeline.
All paths are resolved relative to PROJECT_ROOT so the code runs
both locally and inside Docker (mounted at /host_project/song-popularity-mlops).
"""
from pathlib import Path
import os

# ── Root ──────────────────────────────────────────────────────────────────────
# Inside Docker the project is mounted at /host_project/song-popularity-mlops
# Locally it lives wherever the repo is cloned.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/host_project/song-popularity-mlops"))

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

CSV_PATH = RAW_DIR / "song_popularity.csv"

# ── Model / artifact paths ────────────────────────────────────────────────────
MODEL_DIR = PROJECT_ROOT / "models"

# ── Expected columns (must match the CSV header after lowercasing) ─────────────
EXPECTED_COLUMNS = [
    "song_name",
    "song_popularity",
    "song_duration_ms",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "audio_mode",
    "speechiness",
    "tempo",
    "time_signature",
    "audio_valence",
]

# ── MariaDB ───────────────────────────────────────────────────────────────────
MARIADB_HOST = os.environ.get("MARIADB_HOST", "mariadb")
MARIADB_PORT = int(os.environ.get("MARIADB_PORT", "3306"))
MARIADB_USER = os.environ.get("MARIADB_USER", "airflow")
MARIADB_PASSWORD = os.environ.get("MARIADB_PASSWORD", "airflow")
MARIADB_DB   = os.environ.get("MARIADB_DB", "songs")

DB_PATH = (
    f"mysql+pymysql://{MARIADB_USER}:{MARIADB_PASSWORD}"
    f"@{MARIADB_HOST}:{MARIADB_PORT}/{MARIADB_DB}"
)

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB   = int(os.environ.get("REDIS_DB", "0"))

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
