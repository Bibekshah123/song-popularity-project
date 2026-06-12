"""
Model Monitoring — LIFECYCLE ACTIVITY (NOT a DAG task)
Compares reference training data against current data for drift detection.
"""
import pandas as pd
from sqlalchemy import create_engine
from config import DB_PATH, REPORT_DIR, REFERENCE_DIR

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except Exception:
    EVIDENTLY_AVAILABLE = False


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    engine = create_engine(DB_PATH)
    df = pd.read_sql("SELECT * FROM songs_analytics", engine)
    df = df.drop(columns=["song_name"], errors="ignore")

    ref_path = REFERENCE_DIR / "training_reference.parquet"
    if ref_path.exists():
        reference = pd.read_parquet(ref_path)
    else:
        reference = df.sample(frac=0.5, random_state=42)

    current = df.drop(reference.index, errors="ignore")
    if len(current) < 10:
        current = df.sample(frac=0.5, random_state=99)

    output = REPORT_DIR / "song_popularity_drift_report.html"

    if EVIDENTLY_AVAILABLE:
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        snapshot.save_html(str(output))
        print(f"Evidently AI drift report saved: {output}")
    else:
        summary = df.describe().to_html()
        output.write_text(f"<html><body><h1>Drift Summary (fallback)</h1>{summary}</body></html>")
        print(f"Fallback report saved: {output}")

    print("NOTE: Monitoring is a lifecycle activity, NOT a DAG task.")


if __name__ == "__main__":
    main()
