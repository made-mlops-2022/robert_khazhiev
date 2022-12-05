import os
import pandas as pd
import pickle
import click
import numpy as np


@click.command("predict")
@click.option("--input-dir")
@click.option("--input-model")
@click.option("--output-dir")
def predict(input_dir: str, input_model: str, output_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    
    with open(input_model, 'rb') as f:
        model = pickle.load(f)
    
    predictions = model.predict(data)

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "predictions.csv"),
               predictions, delimiter=",")


if __name__ == '__main__':
    predict()