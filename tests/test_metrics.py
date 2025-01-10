import enum
import pytest
import pandas as pd
import numpy as np

from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose

from clearbox_engine.preprocessor.preprocessor import Preprocessor
from clearbox_engine.dataset.dataset import Dataset
from clearbox_engine.metrics.distinguishability.TSTR import TSTRScore
from clearbox_engine.metrics.distinguishability.detection import DetectionScore
from clearbox_engine.metrics.distinguishability.query_power import QueryPower
from clearbox_engine.metrics.distinguishability.mutual_information import MutualInformation

from pytest_fixtures import (
    ames_dataset,
    ames_synthetic_dataset,
    auto_imports_dataset,
    auto_imports_synthetic_dataset,
    auto_imports_validation,
    boston_housing_dataset,
    boston_housing_synthetic_dataset,
    hospital_readmission_dataset,
    glass_dataset,
    glass_validation,
    uci_adult_dataset,
    uci_adult_dataset_w_col_types,
    uci_adult_synthetic_dataset,
    uci_adult_validation,
    hepatitis_dataset,
    hepatitis_synthetic_dataset,
)


@pytest.mark.parametrize(
    "dataset, synthetic_dataset, validation_dataset",
    [
        (
            pytest.lazy_fixture("auto_imports_dataset"),
            pytest.lazy_fixture("auto_imports_synthetic_dataset"),
            pytest.lazy_fixture("auto_imports_validation"),
        ),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            pytest.lazy_fixture("uci_adult_synthetic_dataset"),
            pytest.lazy_fixture("uci_adult_validation"),
        ),
    ],
)
def test_TSTR_score_w_preprocessor(
    dataset: Dataset,
    synthetic_dataset: Dataset,
    validation_dataset: Dataset,
):
    preprocessor = Preprocessor(dataset)
    score = TSTRScore(dataset, synthetic_dataset, validation_dataset, preprocessor).get()

    if dataset.regression:
        assert score["task"] == "regression"
        assert "MSE" in score
        assert "MAE" in score
        assert "r2_score" in score
        assert "RMSE" in score
        assert "max_error" in score
        assert score["score"] >= 0
        assert score["score"] <= 1
    else:
        assert score["task"] == "classification"
        assert "accuracy" in score
        assert len(score["metrics"]["training"]) == dataset.get_n_classes()
        assert len(score["metrics"]["synthetic"]) == dataset.get_n_classes()
        assert score["score"] >= 0
        assert score["score"] <= 1

    (
        numerical_features_sizes,
        categorical_features_sizes,
    ) = preprocessor.get_features_sizes()
    assert len(score["feature_importances"]["training"]) == sum(numerical_features_sizes) + len(categorical_features_sizes)
    assert len(score["feature_importances"]["synthetic"]) == sum(numerical_features_sizes) + len(categorical_features_sizes)



@pytest.mark.parametrize(
    "dataset, synthetic_dataset, validation_dataset",
    [
        (
            pytest.lazy_fixture("auto_imports_dataset"),
            pytest.lazy_fixture("auto_imports_synthetic_dataset"),
            pytest.lazy_fixture("auto_imports_validation")
        ),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            pytest.lazy_fixture("uci_adult_synthetic_dataset"),
            pytest.lazy_fixture("uci_adult_validation"),
        ),
    ],
)
def test_TSTR_score_no_preprocessor(
    dataset: Dataset,
    synthetic_dataset: Dataset,
    validation_dataset: Dataset,
):
    TSTR = TSTRScore(dataset, synthetic_dataset, validation_dataset)
    score = TSTR.get()

    if dataset.regression:
        assert score["task"] == "regression"
        assert "MSE" in score
        assert "MAE" in score
        assert "r2_score" in score
        assert "RMSE" in score
        assert "max_error" in score
        assert score["score"] >= 0
        assert score["score"] <= 1
    else:
        assert score["task"] == "classification"
        assert "accuracy" in score
        assert len(score["metrics"]["training"]) == dataset.get_n_classes()
        assert len(score["metrics"]["synthetic"]) == dataset.get_n_classes()
        assert score["score"] >= 0
        assert score["score"] <= 1

    (
        numerical_features_sizes,
        categorical_features_sizes,
    ) = TSTR.preprocessor.get_features_sizes()
    assert len(score["feature_importances"]["training"]) == sum(numerical_features_sizes) + len(categorical_features_sizes)
    assert len(score["feature_importances"]["synthetic"]) == sum(numerical_features_sizes) + len(categorical_features_sizes)


@pytest.mark.parametrize(
    "dataset, synthetic_dataset",
    [
        (
            pytest.lazy_fixture("ames_dataset"),
            pytest.lazy_fixture("ames_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("auto_imports_dataset"),
            pytest.lazy_fixture("auto_imports_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("boston_housing_dataset"),
            pytest.lazy_fixture("boston_housing_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            pytest.lazy_fixture("uci_adult_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("hepatitis_dataset"),
            pytest.lazy_fixture("hepatitis_synthetic_dataset"),
        ),
    ],
)
def test_detection_score_w_preprocessor(
    dataset: Dataset,
    synthetic_dataset: Dataset,
):
    preprocessor = Preprocessor(dataset)
    score = DetectionScore(dataset, synthetic_dataset, preprocessor).get()

    assert "accuracy" in score
    assert score["accuracy"] >= 0
    assert score["accuracy"] <= 1
    assert "ROC_AUC" in score
    assert score["ROC_AUC"] >= 0
    assert score["ROC_AUC"] <= 1
    assert score["score"] >= 0
    assert score["score"] <= 1

    (
        numerical_features_sizes,
        categorical_features_sizes,
    ) = preprocessor.get_features_sizes()
    assert len(score["feature_importances"]) == sum(numerical_features_sizes) + len(categorical_features_sizes)


@pytest.mark.parametrize(
    "dataset, synthetic_dataset",
    [
        (
            pytest.lazy_fixture("ames_dataset"),
            pytest.lazy_fixture("ames_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("auto_imports_dataset"),
            pytest.lazy_fixture("auto_imports_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("boston_housing_dataset"),
            pytest.lazy_fixture("boston_housing_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            pytest.lazy_fixture("uci_adult_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("hepatitis_dataset"),
            pytest.lazy_fixture("hepatitis_synthetic_dataset"),
        ),
    ],
)
def test_detection_score_no_preprocessor(
    dataset: Dataset,
    synthetic_dataset: Dataset,
):
    DS = DetectionScore(dataset, synthetic_dataset)
    score = DS.get()

    assert "accuracy" in score
    assert score["accuracy"] >= 0
    assert score["accuracy"] <= 1
    assert "ROC_AUC" in score
    assert score["ROC_AUC"] >= 0
    assert score["ROC_AUC"] <= 1
    assert score["score"] >= 0
    assert score["score"] <= 1

    (
        numerical_features_sizes,
        categorical_features_sizes,
    ) = DS.preprocessor.get_features_sizes()
    assert len(score["feature_importances"]) == sum(numerical_features_sizes) + len(categorical_features_sizes)


@pytest.mark.parametrize(
    "dataset, synthetic_dataset",
    [
        (
            pytest.lazy_fixture("ames_dataset"),
            pytest.lazy_fixture("ames_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("auto_imports_dataset"),
            pytest.lazy_fixture("auto_imports_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("boston_housing_dataset"),
            pytest.lazy_fixture("boston_housing_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            pytest.lazy_fixture("uci_adult_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("hepatitis_dataset"),
            pytest.lazy_fixture("hepatitis_synthetic_dataset"),
        ),
    ],
)
def test_query_power_w_preprocessor(
    dataset: Dataset,
    synthetic_dataset: Dataset,
):
    preprocessor = Preprocessor(dataset)
    score = QueryPower(dataset, synthetic_dataset, preprocessor).get()

    assert len(score["queries"]) <= 5
    assert score["score"] >= 0
    assert score["score"] <= 1


@pytest.mark.parametrize(
    "dataset, synthetic_dataset",
    [
        (
            pytest.lazy_fixture("ames_dataset"),
            pytest.lazy_fixture("ames_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("auto_imports_dataset"),
            pytest.lazy_fixture("auto_imports_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("boston_housing_dataset"),
            pytest.lazy_fixture("boston_housing_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            pytest.lazy_fixture("uci_adult_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("hepatitis_dataset"),
            pytest.lazy_fixture("hepatitis_synthetic_dataset"),
        ),
    ],
)
def test_query_power_n_preprocessor(
    dataset: Dataset,
    synthetic_dataset: Dataset,
):
    score = QueryPower(dataset, synthetic_dataset).get()

    assert len(score["queries"]) <= 5
    assert score["score"] >= 0
    assert score["score"] <= 1


@pytest.mark.parametrize(
    "dataset, synthetic_dataset",
    [
        (
            pytest.lazy_fixture("ames_dataset"),
            pytest.lazy_fixture("ames_dataset"),
        ),
        (
            pytest.lazy_fixture("auto_imports_dataset"),
            pytest.lazy_fixture("auto_imports_dataset"),
        ),
        (
            pytest.lazy_fixture("boston_housing_dataset"),
            pytest.lazy_fixture("boston_housing_dataset"),
        ),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            pytest.lazy_fixture("uci_adult_dataset"),
        ),
        (
            pytest.lazy_fixture("hepatitis_dataset"),
            pytest.lazy_fixture("hepatitis_dataset"),
        ),
    ],
)
def test_query_power_same_dataset(
    dataset: Dataset,
    synthetic_dataset: Dataset,
):
    score = QueryPower(dataset, synthetic_dataset).get()

    for query in score["queries"]:
        assert query["original_df"] == query["synthetic_df"]


@pytest.mark.parametrize(
    "dataset, synthetic_dataset",
    [
        (
            pytest.lazy_fixture("ames_dataset"),
            pytest.lazy_fixture("ames_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("auto_imports_dataset"),
            pytest.lazy_fixture("auto_imports_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("boston_housing_dataset"),
            pytest.lazy_fixture("boston_housing_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            pytest.lazy_fixture("uci_adult_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("hepatitis_dataset"),
            pytest.lazy_fixture("hepatitis_synthetic_dataset"),
        ),
    ],
)
def test_mutual_information_w_preprocessor(
    dataset: Dataset,
    synthetic_dataset: Dataset,
):
    preprocessor = Preprocessor(dataset)
    score = MutualInformation(dataset, synthetic_dataset, preprocessor).get()

    assert len(score["original_mutual_information"]) == len(score["features"])
    assert len(score["synthetic_mutual_information"]) == len(score["features"])
    assert len(score["diff_correlation_matrix"]) == len(score["features"])

    for i, _ in enumerate(score["features"]):
        assert len(score["original_mutual_information"][i]) == len(score["features"])
        assert len(score["synthetic_mutual_information"][i]) == len(score["features"])
        assert len(score["diff_correlation_matrix"][i]) == len(score["features"])
        assert_allclose(
                    score["original_mutual_information"][i][i],
                    1,
                    atol=1e-6,
                )
        assert_allclose(
                    score["synthetic_mutual_information"][i][i],
                    1,
                    atol=1e-6,
                )
        assert_allclose(
                    score["diff_correlation_matrix"][i][i],
                    0,
                    atol=1e-6,
                )


@pytest.mark.parametrize(
    "dataset, synthetic_dataset",
    [
        (
            pytest.lazy_fixture("ames_dataset"),
            pytest.lazy_fixture("ames_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("auto_imports_dataset"),
            pytest.lazy_fixture("auto_imports_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("boston_housing_dataset"),
            pytest.lazy_fixture("boston_housing_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            pytest.lazy_fixture("uci_adult_synthetic_dataset"),
        ),
        (
            pytest.lazy_fixture("hepatitis_dataset"),
            pytest.lazy_fixture("hepatitis_synthetic_dataset"),
        ),
    ],
)
def test_mutual_information_no_preprocessor(
    dataset: Dataset,
    synthetic_dataset: Dataset,
):
    MI = MutualInformation(dataset, synthetic_dataset)
    score = MI.get()

    assert len(score["original_mutual_information"]) == len(score["features"])
    assert len(score["synthetic_mutual_information"]) == len(score["features"])
    assert len(score["diff_correlation_matrix"]) == len(score["features"])

    for i, _ in enumerate(score["features"]):
        assert len(score["original_mutual_information"][i]) == len(score["features"])
        assert len(score["synthetic_mutual_information"][i]) == len(score["features"])
        assert len(score["diff_correlation_matrix"][i]) == len(score["features"])
        assert_allclose(
                    score["original_mutual_information"][i][i],
                    1,
                    atol=1e-6,
                )
        assert_allclose(
                    score["synthetic_mutual_information"][i][i],
                    1,
                    atol=1e-6,
                )
        assert_allclose(
                    score["diff_correlation_matrix"][i][i],
                    0,
                    atol=1e-6,
                )


@pytest.mark.parametrize(
    "dataset, synthetic_dataset",
    [
        (
            pytest.lazy_fixture("ames_dataset"),
            pytest.lazy_fixture("ames_dataset"),
        ),
        (
            pytest.lazy_fixture("auto_imports_dataset"),
            pytest.lazy_fixture("auto_imports_dataset"),
        ),
        (
            pytest.lazy_fixture("boston_housing_dataset"),
            pytest.lazy_fixture("boston_housing_dataset"),
        ),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            pytest.lazy_fixture("uci_adult_dataset"),
        ),
        (
            pytest.lazy_fixture("hepatitis_dataset"),
            pytest.lazy_fixture("hepatitis_dataset"),
        ),
    ],
)
def test_mutual_information_same_dataset(
    dataset: Dataset,
    synthetic_dataset: Dataset,
):
    MI = MutualInformation(dataset, synthetic_dataset)
    score = MI.get()

    for i, _ in enumerate(score["features"]):
        for j, _ in enumerate(score["features"]):
            assert_allclose(
                        score["original_mutual_information"][i][j],
                        score["synthetic_mutual_information"][i][j],
                        atol=1e-6,
                    )
            assert_allclose(
                        score["diff_correlation_matrix"][i][j],
                        0,
                        atol=1e-6,
                    )
