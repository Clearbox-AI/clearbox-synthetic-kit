import pytest
import pandas as pd
import numpy as np

from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose

from clearbox_engine.preprocessor.preprocessor import Preprocessor
from clearbox_engine.dataset.dataset import Dataset

from pytest_fixtures import (
    ames_dataset,
    auto_imports_dataset,
    boston_housing_dataset,
    hospital_readmission_dataset,
    glass_dataset,
    glass_validation,
    uci_adult_dataset,
    uci_adult_dataset_w_col_types,
    hepatitis_dataset,
)


@pytest.mark.parametrize(
    "dataset, expected_numerical_features, expected_categorical_features, expected_datetime_features",
    [
        # (pytest.lazy_fixture("ames_dataset")),
        # (pytest.lazy_fixture("auto_imports_dataset")),
        # (pytest.lazy_fixture("boston_housing_dataset")),
        # (pytest.lazy_fixture("hospital_readmission_dataset")),
        # (pytest.lazy_fixture("glass_dataset")),
        # (pytest.lazy_fixture("glass_validation")),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            ["age", "capital_gain", "capital_loss", "hours_per_week"],
            [
                "work_class",
                "education",
                "marital_status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native_country",
            ],
            [],
        ),
        (
            pytest.lazy_fixture("hepatitis_dataset"),
            [
                "age",
                "bilirubin",
                "alk_phosphate",
                "sgot",
                "albumin",
                "protime",
            ],
            [
                "sex",
                "steroid",
                "antivirals",
                "fatigue",
                "malaise",
                "anorexia",
                "liver_big",
                "liver_firm",
                "spleen_palpable",
                "spiders",
                "ascites",
                "varices",
                "histology",
            ],
            [],
        ),
    ],
)
def test_infer_feature_types_without_column_types(
    dataset: Dataset,
    expected_numerical_features: list,
    expected_categorical_features: list,
    expected_datetime_features: list,
):
    (
        numerical_features,
        categorical_features,
        datetime_features,
    ) = Preprocessor._infer_feature_types(dataset)

    assert numerical_features == expected_numerical_features
    assert categorical_features == expected_categorical_features
    assert datetime_features == expected_datetime_features


@pytest.mark.parametrize(
    "dataset, expected_numerical_features, expected_categorical_features, expected_datetime_features",
    [
        # (pytest.lazy_fixture("ames_dataset")),
        # (pytest.lazy_fixture("auto_imports_dataset")),
        # (pytest.lazy_fixture("boston_housing_dataset")),
        # (pytest.lazy_fixture("hospital_readmission_dataset")),
        # (pytest.lazy_fixture("glass_dataset")),
        # (pytest.lazy_fixture("glass_validation")),
        (
            pytest.lazy_fixture("uci_adult_dataset_w_col_types"),
            ["age", "capital_gain", "capital_loss", "hours_per_week"],
            [
                "work_class",
                "education",
                "marital_status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native_country",
            ],
            [],
        ),
        # (pytest.lazy_fixture("hepatitis_dataset")),
    ],
)
def test_infer_feature_types_with_column_types(
    dataset: Dataset,
    expected_numerical_features: list,
    expected_categorical_features: list,
    expected_datetime_features: list,
):
    (
        numerical_features,
        categorical_features,
        datetime_features,
    ) = Preprocessor._infer_feature_types(dataset)
    assert numerical_features == expected_numerical_features
    assert categorical_features == expected_categorical_features
    assert datetime_features == expected_datetime_features


@pytest.mark.parametrize(
    "dataset, too_much_info",
    [
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            {"native_country": ["United-States", "Cuba"]},
        ),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            {"capital_gain": [0]},
        ),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            {"native_country": ["United-States", "Cuba"], "capital_gain": [0]},
        ),
        (
            pytest.lazy_fixture("uci_adult_dataset"),
            {},
        ),
    ],
)
def test_shrink_labels(dataset: Dataset, too_much_info: dict):
    counter = {}
    for column_name in too_much_info:
        to_replace_counter = 0
        for value in too_much_info[column_name]:
            to_replace_counter += (dataset.data[column_name] == value).sum()
        counter[column_name] = to_replace_counter

    shrinked_data = Preprocessor._shrink_labels(dataset.data, too_much_info)

    for column_name in too_much_info:
        for value in too_much_info[column_name]:
            assert (shrinked_data[column_name] == value).sum() == 0
        if dataset.data[column_name].dtype == "object":
            assert (shrinked_data[column_name] == "*").sum() == counter[column_name]
        else:
            assert (shrinked_data[column_name] == -999999).sum() == counter[column_name]


@pytest.mark.parametrize(
    "dataset, threshold, discarded_features",
    [
        #         # (pytest.lazy_fixture("ames_dataset")),
        #         # (pytest.lazy_fixture("auto_imports_dataset")),
        #         # (pytest.lazy_fixture("boston_housing_dataset")),
        (
            pytest.lazy_fixture("hospital_readmission_dataset"),
            0.02,
            (
                ['glimepiride-pioglitazone', 'metformin-rosiglitazone'],
                {
                    "race": ["Hispanic", "Other", "Asian"],
                    "glu_serum_test": ["Normal", "Abnormal"],
                },
            ),
        ),
        #         # (pytest.lazy_fixture("glass_dataset")),
        #         # (pytest.lazy_fixture("glass_validation")),
        #         (pytest.lazy_fixture("uci_adult_dataset")),
        #         (pytest.lazy_fixture("hepatitis_dataset")),
    ],
)
def test_feature_selection(
    dataset: Dataset, threshold: float, discarded_features: tuple
):
    (
        numerical_features,
        categorical_features,
        _,
    ) = Preprocessor._infer_feature_types(dataset)

    preprocessor = Preprocessor.__new__(Preprocessor)
    shrinked_data = Preprocessor._feature_selection(
        preprocessor,
        X=dataset.data,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        threshold=threshold,
    )

    assert preprocessor.discarded == discarded_features


@pytest.mark.parametrize(
    "dataset, n_bins, num_transformer_type",
    [
        (pytest.lazy_fixture("ames_dataset"), 0, "Quantile"),
        (pytest.lazy_fixture("ames_dataset"), 0, "Power"),
        (pytest.lazy_fixture("ames_dataset"), 0, "Scaler"),
        (pytest.lazy_fixture("ames_dataset"), 3, ""),
        (pytest.lazy_fixture("auto_imports_dataset"), 0, "Quantile"),
        (pytest.lazy_fixture("auto_imports_dataset"), 0, "Power"),
        (pytest.lazy_fixture("auto_imports_dataset"), 0, "Scaler"),
        (pytest.lazy_fixture("auto_imports_dataset"), 3, ""),
        (pytest.lazy_fixture("boston_housing_dataset"), 0, "Quantile"),
        (pytest.lazy_fixture("boston_housing_dataset"), 0, "Power"),
        (pytest.lazy_fixture("boston_housing_dataset"), 0, "Scaler"),
        (pytest.lazy_fixture("boston_housing_dataset"), 3, ""),
        (pytest.lazy_fixture("glass_dataset"), 0, "Quantile"),
        (pytest.lazy_fixture("glass_dataset"), 0, "Power"),
        (pytest.lazy_fixture("glass_dataset"), 0, "Scaler"),
        (pytest.lazy_fixture("glass_dataset"), 3, ""),
        (pytest.lazy_fixture("uci_adult_dataset"), 0, "Quantile"),
        (pytest.lazy_fixture("uci_adult_dataset"), 0, "Power"),
        (pytest.lazy_fixture("uci_adult_dataset"), 0, "Scaler"),
        (pytest.lazy_fixture("uci_adult_dataset"), 3, ""),
        (pytest.lazy_fixture("hepatitis_dataset"), 0, "Quantile"),
        (pytest.lazy_fixture("hepatitis_dataset"), 0, "Power"),
        (pytest.lazy_fixture("hepatitis_dataset"), 0, "Scaler"),
        (pytest.lazy_fixture("hepatitis_dataset"), 3, ""),
    ],
)
def test_transform(dataset: Dataset, n_bins: int, num_transformer_type: str):
    preprocessor = Preprocessor(
        dataset, n_numerical_bins=n_bins, num_transformer_type=num_transformer_type
    )

    dataset.data = preprocessor._shrink_labels(dataset.data, preprocessor.discarded[1])

    len_one_hot_encoded_cols = 0
    for cat in preprocessor.get_categorical_features():
        len_one_hot_encoded_cols += len(dataset.data[cat].unique())

    preprocessed_data = preprocessor.transform(dataset.get_x())

    # Check that the number of columns is correct
    assert (
        preprocessed_data.shape[1]
        == len(preprocessor.get_numerical_features())
        + len(preprocessor.get_datetime_features())
        + len_one_hot_encoded_cols
    )

    # Check that the number of rows is correct
    assert len(dataset.data) == preprocessed_data.shape[0]

    # Check that all values are not NAN
    assert np.isnan(preprocessed_data).all() == False

    # Check that the preprocessed numerical features follow the correct strategy
    for i, feature in enumerate(preprocessor.get_numerical_features()):
        if n_bins > 0:
            assert (
                np.unique(preprocessed_data[:, i], return_counts=True)[1].all()
                <= n_bins
            )
        else:
            if num_transformer_type == "Power":
                assert_allclose(np.mean(preprocessed_data[:, i]), 0, atol=1e-6)
            elif num_transformer_type == "Quantile":
                assert_allclose(
                    abs(np.min(preprocessed_data[:, i])),
                    abs(np.max(preprocessed_data[:, i])),
                    atol=1e-6,
                )
            else:
                assert_allclose(
                    np.min(preprocessed_data[:, i]),
                    0,
                    atol=1e-6,
                )
                assert_allclose(
                    np.max(preprocessed_data[:, i]),
                    1,
                    atol=1e-6,
                )

    # Check that the preprocessed categorical features are either 0 or 1 (one hot encoding correctness)
    for i in range(len_one_hot_encoded_cols):
        assert (
            preprocessed_data[:, len(preprocessor.get_numerical_features()) + i].all()
            == 0
            or preprocessed_data[:, len(preprocessor.get_numerical_features()) + i].all()
            == 1
        )


@pytest.mark.parametrize(
    "dataset",
    [
        (pytest.lazy_fixture("ames_dataset")),
        (pytest.lazy_fixture("auto_imports_dataset")),
        (pytest.lazy_fixture("boston_housing_dataset")),
        (pytest.lazy_fixture("hospital_readmission_dataset")),
        (pytest.lazy_fixture("glass_dataset")),
        (pytest.lazy_fixture("glass_validation")),
        (pytest.lazy_fixture("uci_adult_dataset")),
        (pytest.lazy_fixture("hepatitis_dataset")),
    ],
)
def test_reverse_transform(dataset: Dataset):
    preprocessor = Preprocessor(dataset)

    preprocessed_data = preprocessor.transform(dataset.get_x())
    reversed_data = preprocessor.reverse_transform(preprocessed_data)
    discarded_columns = [discarded for discarded in preprocessor.discarded[0]]

    assert reversed_data.shape[0] == dataset.get_x().shape[0]
    assert reversed_data.shape[1] == (dataset.get_x().shape[1] - len(discarded_columns))
    assert [
        column for column in dataset.get_x() if column not in discarded_columns
    ] == reversed_data.columns.tolist()
    assert reversed_data.isnull().values.any() == False
