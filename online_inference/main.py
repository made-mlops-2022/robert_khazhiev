import sys
sys.path.append("../ml_project")

from os.path import exists
from fastapi import FastAPI, Response
from src.predict_pipeline import predict_pipeline
from src.enities.predict_pipeline_params import read_predicting_pipeline_params


app = FastAPI()


@app.get("/predict")
def read_predict():
    arr = predict_pipeline("../ml_project/configs/online_predict_config.yaml")
    return arr.tolist()


@app.get("/health")
def read_health():
    try:
        predicting_pipeline_params = read_predicting_pipeline_params(
            "../ml_project/configs/online_predict_config.yaml")
    except FileNotFoundError:
        return Response(status_code=503)

    model_path = predicting_pipeline_params.input_model_path
    if exists(model_path):
        return Response(status_code=200)
    else:
        return Response(status_code=503)
