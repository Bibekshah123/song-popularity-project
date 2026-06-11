"""
Stage 3 — Data Preprocessing
Reads validation status from Redis → reads MariaDB → generates train/val/test → stores path in Redis
"""
import joblib
import pandas as pd
import redis
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import DB_PATH, PROCESSED_DIR, REFERENCE_DIR
from config import REDIS_HOST, REDIS_PORT, REDIS_DB


def iqr_clip(df):
    df = df.copy()
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)
    return df


def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

    status = r.get("validation_status")
    if status is None or status.decode() != "passed":
        r.set("preprocessing_status", "failed")
        raise ValueError("Validation not passed. Check Redis key: validation_status")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    engine = create_engine(DB_PATH)
    df = pd.read_sql("SELECT * FROM songs_analytics", engine)

    target = "song_popularity"
    X = df.drop(columns=["song_name", target])
    y = df[target]

    feature_names = list(X.columns)
    X = iqr_clip(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    X_train_p = pipeline.fit_transform(X_train)
    X_val_p = pipeline.transform(X_val)
    X_test_p = pipeline.transform(X_test)

    joblib.dump(pipeline, PROCESSED_DIR / "preprocessing_pipeline.pkl")
    joblib.dump(feature_names, PROCESSED_DIR / "feature_names.pkl")
    joblib.dump((X_train_p, y_train), PROCESSED_DIR / "train.pkl")
    joblib.dump((X_val_p, y_val), PROCESSED_DIR / "val.pkl")
    joblib.dump((X_test_p, y_test), PROCESSED_DIR / "test.pkl")

    # Save reference snapshot for Evidently monitoring
    ref_df = pd.DataFrame(X_train_p, columns=feature_names)
    ref_df[target] = y_train.values
    ref_df.to_parquet(REFERENCE_DIR / "training_reference.parquet", index=False)

    r.set("preprocessing_status", "success")
    r.set("processed_data_path", str(PROCESSED_DIR))
    r.set("train_rows", str(X_train_p.shape[0]))
    r.set("val_rows", str(X_val_p.shape[0]))
    r.set("test_rows", str(X_test_p.shape[0]))
    r.set("feature_count", str(len(feature_names)))

    print("=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Training rows:    {X_train_p.shape[0]}")
    print(f"  Validation rows:  {X_val_p.shape[0]}")
    print(f"  Test rows:        {X_test_p.shape[0]}")
    print(f"  Features:         {len(feature_names)}")
    print(f"  Output path:      {PROCESSED_DIR}")
    print(f"  Reference data:   {REFERENCE_DIR / 'training_reference.parquet'}")
    print(f"  Redis key set:    preprocessing_status = success")
    print("=" * 60)


if __name__ == "__main__":
    main()
