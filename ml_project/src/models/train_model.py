import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from typing import Union
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.enities.train_params import TrainingParams

SklearnClassifierModel = Union[GaussianNB, GradientBoostingClassifier]


def train_model_func(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassifierModel:
    if train_params.model_type == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(
            n_estimators=train_params.n_estimators, random_state=train_params.random_state, max_depth=train_params.max_depth
        )
    elif train_params.model_type == "GaussianNB":
        model = GaussianNB()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def create_inference_pipeline(
    model, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])
