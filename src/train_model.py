"""
Stage 4 — Model Training
Reads preprocessing status from Redis → trains models → logs to MLflow → stores metadata in Redis
"""
import joblib
import numpy as np
import redis
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from config import PROCESSED_DIR, MODEL_DIR
from config import REDIS_HOST, REDIS_PORT, REDIS_DB

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def evaluate(model, X, y):
    preds = model.predict(X)
    return r2_score(y, preds), np.sqrt(mean_squared_error(y, preds)), mean_absolute_error(y, preds)


def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

    status = r.get("preprocessing_status")
    if status is None or status.decode() != "success":
        r.set("training_status", "failed")
        raise ValueError("Preprocessing not done. Check Redis key: preprocessing_status")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("song_popularity_regression")

    X_train, y_train = joblib.load(PROCESSED_DIR / "train.pkl")
    X_val, y_val = joblib.load(PROCESSED_DIR / "val.pkl")
    X_test, y_test = joblib.load(PROCESSED_DIR / "test.pkl")

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoostRegressor"] = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, objective="reg:squarederror",
        )

    best_model = None
    best_name = None
    best_r2 = -999

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            vr2, vrmse, vmae = evaluate(model, X_val, y_val)
            tr2, trmse, tmae = evaluate(model, X_test, y_test)

            mlflow.log_param("model_name", name)
            for pn, pv in model.get_params().items():
                try:
                    mlflow.log_param(pn, pv)
                except Exception:
                    pass
            mlflow.log_metrics({"val_r2": vr2, "val_rmse": vrmse, "val_mae": vmae,
                                "test_r2": tr2, "test_rmse": trmse, "test_mae": tmae})
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"\n  {name}")
            print(f"    Val  R2={vr2:.4f}  RMSE={vrmse:.4f}  MAE={vmae:.4f}")
            print(f"    Test R2={tr2:.4f}  RMSE={trmse:.4f}  MAE={tmae:.4f}")

            if tr2 > best_r2:
                best_r2, best_model, best_name = tr2, model, name

    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

    r.set("training_status", "success")
    r.set("best_model_name", best_name)
    r.set("best_model_path", str(MODEL_DIR / "best_model.pkl"))
    r.set("best_test_r2", str(round(best_r2, 4)))

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Best model:   {best_name}")
    print(f"  Test R2:      {best_r2:.4f}")
    print(f"  Saved to:     {MODEL_DIR / 'best_model.pkl'}")
    print(f"  Redis key:    training_status = success")
    print("=" * 60)


if __name__ == "__main__":
    main()
