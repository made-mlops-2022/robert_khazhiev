import unittest
from unittest import mock
import pandas as pd
import numpy as np
from src.features.build_features import (
    MyColumnNormalization,
    build_normilize_pipeline,
    build_numerical_pipeline,
    process_numerical_features,
    make_features,
    drop_features,
    extract_target,
    build_transformer
)


class TestTransformers(unittest.TestCase):
    def test_my_norm_transformer(self):
        df = pd.DataFrame({
            "test_col": np.random.normal(3, 2.5, size=100)
        })
        transformer = MyColumnNormalization("test_col")
        transformed_df = transformer.fit_transform(df)
        self.assertAlmostEqual(transformed_df["test_col"].mean(), 0)
        self.assertAlmostEqual(transformed_df["test_col"].std(), 1)

    @mock.patch("src.features.build_features.MyColumnNormalization")
    @mock.patch("src.features.build_features.Pipeline")
    def test_build_normilize_pipeline(self, Pipeline_mock: mock.Mock, MyColumnNormalization_mock: mock.Mock):
        Pipeline_mock.return_value = Pipeline_mock
        MyColumnNormalization_mock.return_value = MyColumnNormalization_mock
        self.assertEqual(build_normilize_pipeline("test"), Pipeline_mock)
        Pipeline_mock.assert_called_once_with(
            [("my_normilize", MyColumnNormalization_mock), ]
        )
        MyColumnNormalization_mock.assert_called_once_with("test")

    @mock.patch("src.features.build_features.SimpleImputer")
    @mock.patch("src.features.build_features.Pipeline")
    def test_build_numerical_pipeline(self, Pipeline_mock: mock.Mock, SimpleImputer_mock: mock.Mock):
        Pipeline_mock.return_value = Pipeline_mock
        SimpleImputer_mock.return_value = SimpleImputer_mock
        self.assertEqual(build_numerical_pipeline(), Pipeline_mock)
        Pipeline_mock.assert_called_once_with(
            [("impute", SimpleImputer_mock), ]
        )
        SimpleImputer_mock.assert_called_once_with(
            missing_values=np.nan, strategy="median")

    @mock.patch("src.features.build_features.ColumnTransformer")
    @mock.patch("src.features.build_features.build_normilize_pipeline")
    @mock.patch("src.features.build_features.build_numerical_pipeline")
    def test_build_transformer(
            self, build_numerical_pipeline_mock: mock.Mock,
            build_normilize_pipeline_mock: mock.Mock,
            ColumnTransformer_mock: mock.Mock):
        build_numerical_pipeline_mock.return_value = build_numerical_pipeline_mock
        build_normilize_pipeline_mock.return_value = build_normilize_pipeline_mock
        ColumnTransformer_mock.return_value = ColumnTransformer_mock
        feature_params_mock = mock.Mock()

        self.assertEqual(build_transformer(
            feature_params_mock), ColumnTransformer_mock)
        ColumnTransformer_mock.assert_called_once_with(
            [
                (
                    "normilize_pipeline",
                    build_normilize_pipeline_mock,
                    [feature_params_mock.feature_to_normilize],
                ),
                (
                    "numerical_pipeline",
                    build_numerical_pipeline_mock,
                    feature_params_mock.numerical_features,
                ),
            ]
        )
        build_normilize_pipeline_mock.assert_called_once_with(
            feature_params_mock.feature_to_normilize)
        build_numerical_pipeline_mock.assert_called_once()


class TestUtils(unittest.TestCase):

    @mock.patch("pandas.DataFrame")
    @mock.patch("src.features.build_features.build_numerical_pipeline")
    def test_process_numerical_features(
            self, build_numerical_pipeline_mock: mock.Mock,
            DataFrame_mock: mock.Mock):
        build_numerical_pipeline_mock.return_value = build_numerical_pipeline_mock
        DataFrame_mock.return_value = DataFrame_mock
        num_mock = mock.Mock()
        self.assertEqual(process_numerical_features(num_mock), DataFrame_mock)
        build_numerical_pipeline_mock.assert_called()
        build_numerical_pipeline_mock.fit_transform.assert_called_once_with(
            num_mock)
        DataFrame_mock.assert_called_once_with(build_numerical_pipeline_mock.fit_transform(
            num_mock), columns=num_mock.columns, index=None)

    def test_make_features(self):
        transf = mock.Mock()
        df = mock.Mock()
        transf.transform.return_value = transf
        self.assertEqual(make_features(transf, df), transf)
        transf.transform.assert_called_once_with(df)

    def test_drop_features(self):
        df = mock.Mock()
        df.drop.return_value = df
        self.assertEqual(drop_features(df, mock.Mock()), df)
        df.drop.assert_called_once()

    def test_extract_target(self):
        df = mock.Mock()
        params = mock.Mock()
        df_d = {
            params.target_col: df
        }
        self.assertEqual(extract_target(df_d, params), df)
