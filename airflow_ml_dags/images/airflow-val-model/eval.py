import os
import pandas as pd
import click
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import pickle
import json


@click.command("eval_model")
@click.option("--input-model")
@click.option("--input-dir")
@click.option("--output-metrics-dir")
def eval_model(input_model: str, input_dir: str, output_metrics_dir: str):
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
    target = pd.read_csv(os.path.join(input_dir, "y_test.csv"))

    with open(input_model, 'rb') as f:
        model = pickle.load(f)

    predicts = model.predict(X_test)

    metrics = {
        "f1_score": f1_score(target, predicts, average='weighted'),
        "recall": recall_score(target, predicts, average='weighted'),
        "precision": precision_score(target, predicts, average='weighted'),
        "accuracy": accuracy_score(target, predicts)
    }
    os.makedirs(output_metrics_dir, exist_ok=True)
    with open(os.path.join(output_metrics_dir, "GBMetrics.csv"), "w") as metric_file:
        json.dump(metrics, metric_file)

    


if __name__ == '__main__':
    eval_model()