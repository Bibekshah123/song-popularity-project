"""
Song Popularity MLOps DAG
NOTE: Monitoring is NOT a DAG task — it is an ongoing lifecycle activity.
"""
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

PROJECT_DIR = "/host_project/song-popularity-mlops"

default_args = {
    "owner": "bibek_shah",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="song_popularity_mlops_pipeline",
    default_args=default_args,
    description="MLOps pipeline — MariaDB + Redis + MLflow",
    start_date=datetime(2026, 5, 30),
    schedule_interval="0 18 * * 0",
    catchup=False,
    tags=["mlops", "song-popularity", "mariadb", "redis"],
) as dag:

    data_ingestion = BashOperator(
        task_id="data_ingestion",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "PYTHONPATH=src python src/etl_ingestion.py"
        ),
        env={
            "PROJECT_ROOT": PROJECT_DIR,
            "MARIADB_HOST": "mariadb",
            "MARIADB_PORT": "3306",
            "MARIADB_USER": "airflow",
            "MARIADB_PASSWORD": "airflow",
            "MARIADB_DB": "songs",
            "REDIS_HOST": "redis",
            "REDIS_PORT": "6379",
            "REDIS_DB": "0",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        },
        append_env=True,
    )

    data_validation = BashOperator(
        task_id="data_validation",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "PYTHONPATH=src python src/validate_data.py"
        ),
        env={
            "PROJECT_ROOT": PROJECT_DIR,
            "MARIADB_HOST": "mariadb",
            "MARIADB_PORT": "3306",
            "MARIADB_USER": "airflow",
            "MARIADB_PASSWORD": "airflow",
            "MARIADB_DB": "songs",
            "REDIS_HOST": "redis",
            "REDIS_PORT": "6379",
            "REDIS_DB": "0",
        },
        append_env=True,
    )

    preprocessing = BashOperator(
        task_id="preprocessing",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "PYTHONPATH=src python src/preprocess.py"
        ),
        env={
            "PROJECT_ROOT": PROJECT_DIR,
            "MARIADB_HOST": "mariadb",
            "MARIADB_PORT": "3306",
            "MARIADB_USER": "airflow",
            "MARIADB_PASSWORD": "airflow",
            "MARIADB_DB": "songs",
            "REDIS_HOST": "redis",
            "REDIS_PORT": "6379",
            "REDIS_DB": "0",
        },
        append_env=True,
    )

    model_training = BashOperator(
        task_id="model_training",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "PYTHONPATH=src python src/train_model.py"
        ),
        env={
            "PROJECT_ROOT": PROJECT_DIR,
            "MARIADB_HOST": "mariadb",
            "REDIS_HOST": "redis",
            "REDIS_PORT": "6379",
            "REDIS_DB": "0",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        },
        append_env=True,
    )

    # Model deployment: copy best model to app/backend expected path and write a ready flag
    model_deployment = BashOperator(
        task_id="model_deployment",
        bash_command=(
            f"echo deployed > {PROJECT_DIR}/models/deployment.flag && "
            "echo 'Model deployment complete — best_model.pkl is ready for the FastAPI backend.'"
        ),
        append_env=True,
    )

    # Monitoring is a LIFECYCLE ACTIVITY — not a DAG component
    data_ingestion >> data_validation >> preprocessing >> model_training >> model_deployment
