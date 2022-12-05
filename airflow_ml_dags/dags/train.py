import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "train_model",
        tags=["MLOps"],
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(1),
) as dag:
    process_data = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="/home/robert/Документы/Programming/Technopark/ML-Ops/robert_khazhiev/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    split_data = DockerOperator(
        image="airflow-split",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/splited/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="/home/robert/Документы/Programming/Technopark/ML-Ops/robert_khazhiev/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    train_model = DockerOperator(
        image="airflow-train-model",
        command="--input-dir /data/splited/{{ ds }} --output-dir /data/model/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-train-model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="/home/robert/Документы/Programming/Technopark/ML-Ops/robert_khazhiev/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    val_model = DockerOperator(
        image="airflow-val-model",
        command="--input-dir /data/splited/{{ ds }} --input-model /data/model/{{ ds }}/GradientBoosting.pkl --output-metrics-dir /data/metrics/{{ ds }}/",
        network_mode="bridge",
        task_id="docker-airflow-val-model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="/home/robert/Документы/Programming/Technopark/ML-Ops/robert_khazhiev/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    process_data >> split_data >> train_model >> val_model
