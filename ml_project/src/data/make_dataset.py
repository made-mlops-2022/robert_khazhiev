# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from boto3 import client
from sklearn.model_selection import train_test_split
from typing import Tuple, NoReturn

from src.enities import read_training_pipeline_params
from src.enities import SplittingParams
from src.enities import FeatureParams
from src.features import drop_features


def download_data_from_s3(s3_bucket: str, s3_path: str, output: str) -> NoReturn:
    s3 = client("s3")
    logger = logging.getLogger(__name__)
    logger.info('downloading data from s3')
    s3.download_file(s3_bucket, s3_path, output)


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    logger = logging.getLogger(__name__)
    logger.info('reading data from csv')
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :rtype: object
    """
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data


def process_data(input_filepath: str, output_filepath: str, feat_params: FeatureParams):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # TODO: Внедрить поддержку S3
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    df = read_data(input_filepath)
    df = drop_features(df, feat_params)
    df.to_csv(output_filepath, index_label=False)


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main_command(config_path: str):
    training_params = read_training_pipeline_params(config_path)
    process_data(training_params.input_data_path,
                 training_params.output_proccessed_data_path, training_params.feature_params)


if __name__ == '__main__':
    main_command()
