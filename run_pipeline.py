"""One-command pipeline: ingestion → validation → preprocessing → training."""
import os
import subprocess
import sys
import time

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

STAGES = [
    ("ETL Ingestion",    [sys.executable, "src/etl_ingestion.py"]),
    ("Data Validation",  [sys.executable, "src/validate_data.py"]),
    ("Preprocessing",    [sys.executable, "src/preprocess.py"]),
    ("Model Training",   [sys.executable, "src/train_model.py"]),
]


def main():
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
    start = time.time()
    for name, cmd in STAGES:
        print(f"\n{'=' * 60}")
        print(f"  STAGE: {name}")
        print(f"{'=' * 60}")
        subprocess.check_call(cmd, env={**os.environ})
        print(f"  ✓ {name} completed")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
