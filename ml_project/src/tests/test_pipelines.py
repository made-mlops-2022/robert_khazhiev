import unittest
from unittest import mock
from src.predict_pipeline import (
    predict_pipeline,
    run_predict_pipeline
)
from src.train_pipeline import (
    train_pipeline,
    run_train_pipeline
)
import numpy as np
import pandas as pd
from os.path import exists
from os import remove


class TestPredictPipeline(unittest.TestCase):

    @mock.patch("src.predict_pipeline.run_predict_pipeline")
    @mock.patch("src.predict_pipeline.read_predicting_pipeline_params")
    def test_predict_pipeline(self, read_params_mock: mock.Mock, run_predict_mock: mock.Mock):
        cfg_path = "path"
        run_predict_mock.return_value = run_predict_mock
        read_params_mock.return_value = read_params_mock

        self.assertEqual(predict_pipeline(cfg_path), run_predict_mock)
        read_params_mock.assert_called_once_with(cfg_path)
        run_predict_mock.assert_called_once_with(read_params_mock)

    @mock.patch("numpy.savetxt")
    @mock.patch("src.predict_pipeline.predict_model_func")
    @mock.patch("src.predict_pipeline.deserialize_model")
    @mock.patch("src.predict_pipeline.read_data")
    @mock.patch("src.predict_pipeline.process_data")
    def test_run_predict_pipeline(
            self,
            process_data_mock: mock.Mock,
            read_data_mock: mock.Mock,
            deserialize_model_mock: mock.Mock,
            predict_model_func_mock: mock.Mock,
            savetxt_mock: mock.Mock):
        predicting_params_mock = mock.Mock()

        read_data_mock.return_value = read_data_mock
        deserialize_model_mock.return_value = deserialize_model_mock.model_mock
        predict_model_func_mock.return_value = predict_model_func_mock

        self.assertEqual(run_predict_pipeline(
            predicting_params_mock), predict_model_func_mock)
        process_data_mock.assert_called_once_with(predicting_params_mock.input_data_path,
                                                  predicting_params_mock.output_proccessed_data_path,
                                                  predicting_params_mock.feature_params)
        read_data_mock.assert_called_once_with(
            predicting_params_mock.output_proccessed_data_path)
        deserialize_model_mock.assert_called_once_with(
            predicting_params_mock.input_model_path)
        predict_model_func_mock.assert_called_once_with(
            deserialize_model_mock.model_mock, read_data_mock)
        savetxt_mock.assert_called_once_with(
            predicting_params_mock.output_predict_path, predict_model_func_mock, delimiter=",")


class TestTrainPipeline(unittest.TestCase):

    @mock.patch("src.train_pipeline.open")
    @mock.patch("src.train_pipeline.mlflow")
    @mock.patch("src.train_pipeline.run_train_pipeline")
    @mock.patch("src.train_pipeline.read_training_pipeline_params")
    def test_train_pipeline(
            self,
            read_training_pipeline_params_mock: mock.Mock,
            run_train_pipeline_mock: mock.Mock,
            mlflow_mock: mock.Mock,
            open_mock: mock.Mock):
        read_training_pipeline_params_mock.return_value = read_training_pipeline_params_mock
        run_train_pipeline_mock.return_value = (
            run_train_pipeline_mock.model, run_train_pipeline_mock.path)
        cfg_path = "path"

        read_training_pipeline_params_mock.use_mlflow.return_value = False
        self.assertEqual(train_pipeline(cfg_path), None)
        read_training_pipeline_params_mock.assert_called_once_with(cfg_path)
        run_train_pipeline_mock.assert_called_once_with(
            read_training_pipeline_params_mock)

        read_training_pipeline_params_mock.use_mlflow.return_value = True
        self.assertEqual(train_pipeline(cfg_path), None)
        run_train_pipeline_mock.assert_called_with(
            read_training_pipeline_params_mock)
        mlflow_mock.set_tracking_uri.assert_called_with(
            read_training_pipeline_params_mock.mlflow_uri)
        mlflow_mock.set_experiment.assert_called_with(
            read_training_pipeline_params_mock.mlflow_experiment)
        mlflow_mock.start_run.assert_called()
        mlflow_mock.log_artifact.assert_called()
        mlflow_mock.log_metrics.assert_called()

    @mock.patch("src.train_pipeline.create_inference_pipeline")
    @mock.patch("src.train_pipeline.build_transformer")
    @mock.patch("src.train_pipeline.open")
    @mock.patch("json.dump")
    @mock.patch("src.train_pipeline.evaluate_model")
    @mock.patch("src.train_pipeline.predict_model_func")
    @mock.patch("src.train_pipeline.serialize_model")
    @mock.patch("src.train_pipeline.train_model_func")
    @mock.patch("src.train_pipeline.extract_target")
    @mock.patch("src.train_pipeline.split_train_val_data")
    @mock.patch("src.train_pipeline.read_data")
    @mock.patch("src.train_pipeline.process_data")
    def test_run_train_pipeline(
            self, process_data_mock: mock.Mock, read_data_mock: mock.Mock,
            split_train_val_data_mock: mock.Mock,
            extract_target_mock: mock.Mock,
            train_model_func_mock: mock.Mock,
            serialize_model_mock: mock.Mock, predict_model_func_mock: mock.Mock,
            evaluate_model_mock: mock.Mock, json_dump_mock: mock.Mock, open_mock: mock.Mock,
            build_transformer_mock: mock.Mock,
            create_inference_pipeline_mock: mock.Mock):
        training_pipeline_params_mock = mock.Mock()
        read_data_mock.return_value = read_data_mock
        split_train_val_data_mock.return_value = (
            split_train_val_data_mock.train_df, split_train_val_data_mock.val_df)
        extract_target_mock.return_value = extract_target_mock
        train_model_func_mock.return_value = train_model_func_mock
        predict_model_func_mock.return_value = predict_model_func_mock
        evaluate_model_mock.return_value = evaluate_model_mock
        serialize_model_mock.return_value = serialize_model_mock
        build_transformer_mock.return_value = build_transformer_mock
        create_inference_pipeline_mock.return_value = create_inference_pipeline_mock

        self.assertEqual(run_train_pipeline(
            training_pipeline_params_mock), (serialize_model_mock, evaluate_model_mock))
        process_data_mock.assert_called_once_with(
            training_pipeline_params_mock.input_data_path,
            training_pipeline_params_mock.output_proccessed_data_path,
            training_pipeline_params_mock.feature_params)

        read_data_mock.assert_called_once_with(
            training_pipeline_params_mock.output_proccessed_data_path)
        split_train_val_data_mock.assert_called_once_with(
            read_data_mock, training_pipeline_params_mock.splitting_params)
        self.assertEqual(extract_target_mock.call_count, 2)
        split_train_val_data_mock.train_df.drop.assert_called_once_with(
            training_pipeline_params_mock.feature_params.target_col, 1)
        split_train_val_data_mock.val_df.drop.assert_called_once_with(
            training_pipeline_params_mock.feature_params.target_col, 1)
        train_model_func_mock.assert_called_once_with(
            build_transformer_mock.transform(),
            extract_target_mock,
            training_pipeline_params_mock.train_params)
        predict_model_func_mock.assert_called_once_with(
            create_inference_pipeline_mock, split_train_val_data_mock.val_df.drop())
        evaluate_model_mock.assert_called_once_with(
            predict_model_func_mock, extract_target_mock)
        json_dump_mock.assert_called()
        serialize_model_mock.assert_called_once_with(
            create_inference_pipeline_mock, training_pipeline_params_mock.output_model_path)


def generate_train_csv(n_samples: int):
    df = pd.DataFrame({'age': np.random.randint(29, 78, size=n_samples),
                       'sex': np.random.randint(0, 2, size=n_samples),
                       'cp': np.random.randint(0, 4, size=n_samples),
                       'trestbps': np.random.randint(94, 201, size=n_samples),
                       'chol': np.random.randint(126, 565, size=n_samples),
                       'fbs': np.random.randint(0, 2, size=n_samples),
                       'restecg': np.random.randint(0, 3, size=n_samples),
                       'thalach': np.random.randint(71, 203, size=n_samples),
                       'exang': np.random.randint(0, 2, size=n_samples),
                       'oldpeak': np.random.randint(29, 77, size=n_samples),
                       'slope': np.random.randint(0, 7, size=n_samples),
                       'ca': np.random.randint(0, 4, size=n_samples),
                       'thal': np.random.randint(0, 3, size=n_samples),
                       'condition': np.random.randint(0, 2, size=n_samples)
                       })
    df.to_csv("data/raw/generated_train_temp_data.csv", index=False)


def generate_predict_csv(n_samples: int):
    df = pd.DataFrame({'age': np.random.randint(29, 78, size=n_samples),
                       'sex': np.random.randint(0, 2, size=n_samples),
                       'cp': np.random.randint(0, 4, size=n_samples),
                       'trestbps': np.random.randint(94, 201, size=n_samples),
                       'chol': np.random.randint(126, 565, size=n_samples),
                       'fbs': np.random.randint(0, 2, size=n_samples),
                       'restecg': np.random.randint(0, 3, size=n_samples),
                       'thalach': np.random.randint(71, 203, size=n_samples),
                       'exang': np.random.randint(0, 2, size=n_samples),
                       'oldpeak': np.random.randint(29, 77, size=n_samples),
                       'slope': np.random.randint(0, 7, size=n_samples),
                       'ca': np.random.randint(0, 4, size=n_samples),
                       'thal': np.random.randint(0, 3, size=n_samples)
                       })
    df.to_csv("data/raw/generated_predict_temp_data.csv", index=False)


class IntegrationTest(unittest.TestCase):
    def test_whole_pipeline(self):
        generate_train_csv(200)
        self.assertTrue(exists("data/raw/generated_train_temp_data.csv"))
        train_pipeline("src/tests/tests_artifacts/test_train_config.yaml")
        self.assertTrue(exists("data/processed/generated_train_temp_data.csv"))
        self.assertTrue(exists("models/temp_test_model.pkl"))
        self.assertTrue(exists("models/temp_test_metrics.json"))

        generate_predict_csv(100)
        self.assertTrue(exists("data/raw/generated_predict_temp_data.csv"))
        predict_pipeline("src/tests/tests_artifacts/test_predict_config.yaml")
        self.assertTrue(
            exists("data/processed/generated_predict_temp_data.csv"))
        self.assertTrue(exists("models/predictions/temp_test_predictions.csv"))

    def tearDown(self):
        files_to_delete = [
            'data/raw/generated_train_temp_data.csv',
            'data/processed/generated_train_temp_data.csv',
            'models/temp_test_model.pkl',
            'models/temp_test_metrics.json',
            'data/processed/generated_predict_temp_data.csv',
            'models/predictions/temp_test_predictions.csv',
            'data/raw/generated_predict_temp_data.csv'
        ]
        for i in files_to_delete:
            try:
                remove(i)
            except FileNotFoundError:
                pass
            except Exception as e:
                print(e)
