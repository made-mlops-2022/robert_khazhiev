import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount
from airflow.models import Variable

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "predict",
        tags=["MLOps"],
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    model_path = Variable.get("model_path")
    predict = DockerOperator(
        image="airflow-predict",
        command="--input-dir /data/raw/{{ ds }} --input-model %s --output-dir /data/predictions/{{ ds }}" % model_path,
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="/home/robert/Документы/Programming/Technopark/ML-Ops/robert_khazhiev/airflow_ml_dags/data/", target="/data", type='bind')]
    )
