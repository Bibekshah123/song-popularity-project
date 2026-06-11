"""
Stage 1 — ETL Data Ingestion
Reads CSV → validates → writes to MariaDB ColumnStore → stores metadata in Redis
"""
import os
import pandas as pd
import redis
from sqlalchemy import create_engine
from datetime import datetime
from config import CSV_PATH, DB_PATH, EXPECTED_COLUMNS, PROCESSED_DIR
from config import REDIS_HOST, REDIS_PORT, REDIS_DB


def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    return df


def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

    if not os.path.exists(CSV_PATH):
        r.set("ingestion_status", "failed")
        r.set("ingestion_error", f"File not found: {CSV_PATH}")
        raise FileNotFoundError(f"File not found: {CSV_PATH}. Place song_popularity.csv in data/raw/")

    df = pd.read_csv(CSV_PATH)
    df = clean_column_names(df)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        r.set("ingestion_status", "failed")
        r.set("ingestion_error", f"Missing columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    df = df[EXPECTED_COLUMNS]
    df = df.drop_duplicates()

    engine = create_engine(DB_PATH)
    started = datetime.utcnow()

    df.to_sql("songs_analytics", engine, if_exists="replace", index=False)

    log_df = pd.DataFrame([{
        "schema_version": "v1",
        "row_count": len(df),
        "column_count": df.shape[1],
        "null_count": int(df.isnull().sum().sum()),
        "started_at_utc": str(started),
        "finished_at_utc": str(datetime.utcnow()),
        "status": "success",
        "error_message": None,
    }])
    log_df.to_sql("ingestion_log", engine, if_exists="append", index=False)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "songs_analytics.parquet", index=False)

    r.set("ingestion_status", "success")
    r.set("ingestion_row_count", str(len(df)))
    r.set("ingestion_column_count", str(df.shape[1]))
    r.set("ingestion_null_count", str(int(df.isnull().sum().sum())))
    r.set("ingestion_timestamp", datetime.utcnow().isoformat())

    print("=" * 60)
    print("ETL INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Rows loaded to MariaDB:  {len(df)}")
    print(f"  Columns:                 {df.shape[1]}")
    print(f"  Null values:             {int(df.isnull().sum().sum())}")
    print(f"  Parquet cache:           {PROCESSED_DIR / 'songs_analytics.parquet'}")
    print(f"  Redis key set:           ingestion_status = success")
    print("=" * 60)


if __name__ == "__main__":
    main()
