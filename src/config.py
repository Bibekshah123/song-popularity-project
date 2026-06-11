from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "song_popularity.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "drift"

# ── MariaDB ColumnStore connection ──
DB_USER = "bibek"
DB_PASSWORD = "bibek123"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "song_popularity_db"
DB_PATH = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── Redis connection ──
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# ── Expected columns ──
EXPECTED_COLUMNS = [
    "song_name",
    "song_popularity",
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
    "song_duration_ms",
]
