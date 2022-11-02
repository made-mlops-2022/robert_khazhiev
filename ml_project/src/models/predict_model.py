import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from typing import Dict


def predict_model_func(
    model: Pipeline, features: pd.DataFrame
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    return {
        "f1_score": f1_score(target, predicts),
        "recall": recall_score(target, predicts),
        "precision": precision_score(target, predicts),
        "accuracy": accuracy_score(target, predicts)
    }


def deserialize_model(input: str) -> object:
    with open(input, 'rb') as f:
        model = pickle.load(f)
    return model
