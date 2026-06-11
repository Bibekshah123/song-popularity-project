"""
Stage 2 — Data Validation
Reads ingestion status from Redis → validates MariaDB table → stores validation result in Redis
"""
import pandas as pd
import redis
from sqlalchemy import create_engine
from config import DB_PATH, EXPECTED_COLUMNS
from config import REDIS_HOST, REDIS_PORT, REDIS_DB


def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

    status = r.get("ingestion_status")
    if status is None or status.decode() != "success":
        r.set("validation_status", "failed")
        r.set("validation_error", "Ingestion did not complete successfully.")
        raise ValueError("Ingestion not successful. Check Redis key: ingestion_status")

    engine = create_engine(DB_PATH)
    df = pd.read_sql("SELECT * FROM songs_analytics", engine)

    if df.empty:
        r.set("validation_status", "failed")
        raise ValueError("songs_analytics table is empty.")

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        r.set("validation_status", "failed")
        raise ValueError(f"Missing columns: {missing}")

    if df["song_popularity"].min() < 0 or df["song_popularity"].max() > 100:
        r.set("validation_status", "failed")
        raise ValueError("song_popularity must be between 0 and 100.")

    numeric_cols = [c for c in EXPECTED_COLUMNS if c != "song_name"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            r.set("validation_status", "failed")
            raise TypeError(f"{col} must be numeric.")

    null_rates = df.isnull().mean()
    high_null = null_rates[null_rates > 0.30]
    if len(high_null) > 0:
        r.set("validation_status", "failed")
        raise ValueError(f"High null-rate columns: {list(high_null.index)}")

    dup_count = df.duplicated().sum()

    r.set("validation_status", "passed")
    r.set("validation_row_count", str(len(df)))
    r.set("validation_duplicate_count", str(dup_count))

    print("=" * 60)
    print("DATA VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Rows validated:   {len(df)}")
    print(f"  Duplicates:       {dup_count}")
    print(f"  Redis key set:    validation_status = passed")
    print("=" * 60)
    print("\nNull rate per column:")
    print(null_rates.to_string())


if __name__ == "__main__":
    main()
