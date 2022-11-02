import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.enities import FeatureParams
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


class MyColumnNormalization(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X, y)

    def transform(self, X: pd.DataFrame, y=None):
        X_ = X.copy()
        x_mean = X_[self.feature_name].mean()
        x_std = X_[self.feature_name].std()
        X_[self.feature_name] = (X_[self.feature_name] - x_mean)/x_std
        return X_


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "normilize_pipeline",
                build_normilize_pipeline(params.feature_to_normilize),
                [params.feature_to_normilize],
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def build_normilize_pipeline(feature_name: str) -> Pipeline:
    norm_pipeline = Pipeline(
        [
            ("my_normilize", MyColumnNormalization(feature_name)),
        ]
    )
    return norm_pipeline


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="median")), ]
    )
    return num_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info('imputing missing values by median')
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df), columns=numerical_df.columns, index=None)


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return transformer.transform(df)


def drop_features(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info('droping features')
    return df.drop(columns=params.features_to_drop)


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    logger = logging.getLogger(__name__)
    logger.info('extracting target value')
    target = df[params.target_col]
    return target
