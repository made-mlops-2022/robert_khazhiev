import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.enities import FeatureParams



def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="median")),]
    )
    return num_pipeline

def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info('imputing missing values by median')
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df), columns=numerical_df.columns, index=None)

def drop_features(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info('droping features')
    return df.drop(columns=params.features_to_drop)

def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    logger = logging.getLogger(__name__)
    logger.info('extracting target value')
    target = df[params.target_col]
    return target
