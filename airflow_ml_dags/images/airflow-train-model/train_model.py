import os
import pandas as pd
import click
from sklearn.ensemble import GradientBoostingClassifier
import pickle


@click.command("train_model")
@click.option("--input-dir")
@click.option("--output-dir")
def train_model(input_dir: str, output_dir: str):
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

    model = GradientBoostingClassifier(
            random_state=42
        )
    model.fit(X_train, y_train)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "GradientBoosting.pkl"), "wb") as f:
        pickle.dump(model, f)

    


if __name__ == '__main__':
    train_model()