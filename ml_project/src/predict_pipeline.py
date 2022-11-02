
import logging
import sys

import click
import numpy as np
from src.data.make_dataset import process_data

from src.data import read_data
from src.enities.predict_pipeline_params import (
    PredictingPipelineParams,
    read_predicting_pipeline_params
)
from src.models import (
    predict_model_func,
    deserialize_model
)
# import mlflow

# from src.models.model_fit_predict import create_inference_pipeline

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(config_path: str):
    predicting_pipeline_params = read_predicting_pipeline_params(config_path)
    # TODO: implement MlFlow
    return run_predict_pipeline(predicting_pipeline_params)


def run_predict_pipeline(predicting_pipeline_params: PredictingPipelineParams):
    # downloading_params = training_pipeline_params.downloading_params
    # if downloading_params:
    #     os.makedirs(downloading_params.output_folder, exist_ok=True)
    #     for path in downloading_params.paths:
    #         download_data_from_s3(
    #             downloading_params.s3_bucket,
    #             path,
    #             os.path.join(downloading_params.output_folder, Path(path).name),
    #         )
    # TODO: S3 support

    logger.info(
        f"start predict pipeline with params {predicting_pipeline_params}")
    process_data(predicting_pipeline_params.input_data_path,
                 predicting_pipeline_params.output_proccessed_data_path, predicting_pipeline_params.feature_params)
    predict_df = read_data(
        predicting_pipeline_params.output_proccessed_data_path)
    logger.info(f"predict_df.shape is {predict_df.shape}")

    # inference_pipeline = create_inference_pipeline(model, transformer)
    logger.info("loading model")
    model = deserialize_model(predicting_pipeline_params.input_model_path)
    logger.info("predicting")
    predicts = predict_model_func(model, predict_df)

    np.savetxt(predicting_pipeline_params.output_predict_path,
               predicts, delimiter=",")
    logger.info("predicts saved")

    return predicts


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict_pipeline_command()
