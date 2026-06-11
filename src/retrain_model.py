"""Convenience script to retrain the model after new data or drift detection."""
import subprocess, sys

def main():
    print("Retrain pipeline starting...")
    subprocess.check_call([sys.executable, "src/preprocess.py"])
    subprocess.check_call([sys.executable, "src/train_model.py"])
    print("Retrain pipeline complete.")

if __name__ == "__main__":
    main()
