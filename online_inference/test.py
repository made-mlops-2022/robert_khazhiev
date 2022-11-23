from main import app
from fastapi.testclient import TestClient
import os

client = TestClient(app)


def test_read_health_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_read_heath_503():
    os.rename("../ml_project/configs/online_predict_config.yaml",
              "../ml_project/configs/online_predict_config1.yaml")
    response = client.get("/health")
    assert response.status_code == 503

    os.rename("../ml_project/configs/online_predict_config1.yaml",
              "../ml_project/configs/online_predict_config.yaml")
    os.rename("../ml_project/models/1.0-GradBoost.pkl",
              "../ml_project/models/1.0-GradBoost1.pkl")
    response = client.get("/health")
    assert response.status_code == 503

    os.rename("../ml_project/models/1.0-GradBoost1.pkl",
              "../ml_project/models/1.0-GradBoost.pkl")
    response = client.get("/health")
    assert response.status_code == 200
