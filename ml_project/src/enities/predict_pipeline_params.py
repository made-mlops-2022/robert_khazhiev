from typing import Optional

from dataclasses import dataclass

from .download_params import DownloadParams
from .feature_params import FeatureParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictingPipelineParams:
    input_data_path: str
    output_proccessed_data_path: str
    input_model_path: str
    output_predict_path: str
    feature_params: FeatureParams
    downloading_params: Optional[DownloadParams] = None
    use_mlflow: bool = False
    mlflow_uri: str = "http://18.156.5.226/"
    mlflow_experiment: str = "inference_demo"


PredictingPipelineParamsSchema = class_schema(PredictingPipelineParams)


def read_predicting_pipeline_params(path: str) -> PredictingPipelineParams:
    with open(path, "r") as input_stream:
        schema = PredictingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
