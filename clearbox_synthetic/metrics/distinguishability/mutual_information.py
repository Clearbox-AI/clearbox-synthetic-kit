"""
This module defines the MutualInformation class, which calculates the mutual information
between features in original and synthetic datasets. The comparison helps assess the 
similarity between the feature relationships in both datasets.
"""

import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score
from clearbox_synthetic import Dataset, Preprocessor


class MutualInformation:
    """
    A class for calculating and comparing the mutual information between features in 
    original and synthetic datasets.

    Attributes:
        original_dataset (Dataset): The original dataset.
        synthetic_dataset (Dataset): The synthetic dataset.
        preprocessor (Preprocessor): The preprocessor for handling data transformation.
    """

    original_dataset: Dataset
    synthetic_dataset: Dataset
    preprocessor: Preprocessor

    def __init__(
        self,
        original_dataset: Dataset,
        synthetic_dataset: Dataset,
        preprocessor: Preprocessor = None,
    ) -> None:
        """
        Initializes the MutualInformation class.

        Args:
            original_dataset (Dataset): The original dataset object.
            synthetic_dataset (Dataset): The synthetic dataset object.
            preprocessor (Preprocessor, optional): Preprocessor for data transformation.
                                                   Defaults to None, using a default 
                                                   preprocessor with 10 ordinal bins.
        """
        self.original_dataset = original_dataset
        self.synthetic_dataset = synthetic_dataset
        self.preprocessor = (
            preprocessor
            if preprocessor is not None
            else Preprocessor(original_dataset, n_numerical_bins=10)
        )

    def get(self, features_to_hide: list = []) -> dict:
        """
        Calculates mutual information for features in both datasets and compares them.

        Args:
            features_to_hide (list, optional): List of features to exclude from the analysis.
                                               Defaults to an empty list.

        Returns:
            dict: A dictionary containing mutual information matrices for the original and
                  synthetic datasets, a difference matrix, and an overall similarity score.
        """
        # Transform and reverse transform the original dataset
        original_df = self.preprocessor.transform(
            self.original_dataset.get_x().sample(
                n=min(500, self.original_dataset.data.shape[0])
            )
        )
        original_df = self.preprocessor.reverse_transform(original_df)

        # Fill missing values in original data
        for column in original_df.columns:
            if original_df[column].dtype == 'object':
                original_df[column] = original_df[column].fillna("other")
            else:
                original_df[column] = pd.cut(original_df[column].fillna(0), 5)

        # Transform and reverse transform the synthetic dataset
        synthetic_df = self.preprocessor.transform(
            self.synthetic_dataset.get_x().sample(
                n=min(500, self.synthetic_dataset.data.shape[0])
            )
        )
        synthetic_df = self.preprocessor.reverse_transform(synthetic_df)

        # Fill missing values in synthetic data
        for column in synthetic_df.columns:
            if synthetic_df[column].dtype == 'object':
                synthetic_df[column] = synthetic_df[column].fillna("other")
            else:
                synthetic_df[column] = pd.cut(synthetic_df[column].fillna(0), 5)

        mutual_information = {
            "features": [
                feature
                for feature in synthetic_df.columns
                if feature not in self.preprocessor.get_datetime_features()
                and feature not in features_to_hide
            ]
        }

        # Initialize matrices for mutual information calculations
        original_mutual_information = [
            [0 for _ in range(len(mutual_information["features"]))]
            for _ in range(len(mutual_information["features"]))
        ]
        synthetic_mutual_information = [
            [0 for _ in range(len(mutual_information["features"]))]
            for _ in range(len(mutual_information["features"]))
        ]
        diff_correlation_matrix = [
            [0 for _ in range(len(mutual_information["features"]))]
            for _ in range(len(mutual_information["features"]))
        ]

        # Calculate mutual information for each pair of features
        for i, feature_i in enumerate(mutual_information["features"]):
            for j, feature_j in enumerate(mutual_information["features"]):
                try:
                    result = float(
                        round(
                            normalized_mutual_info_score(
                                original_df[feature_i], original_df[feature_j]
                            ),
                            4,
                        )
                    )
                except Exception:
                    result = float(0)

                original_mutual_information[i][j] = (
                    "NaN" if pd.isnull(result) else float(result)
                )

                try:
                    result = float(
                        round(
                            normalized_mutual_info_score(
                                synthetic_df[feature_i], synthetic_df[feature_j]
                            ),
                            4,
                        )
                    )
                except Exception:
                    result = float(0)

                synthetic_mutual_information[i][j] = (
                    "NaN" if pd.isnull(result) else float(result)
                )

                diff_correlation_matrix[i][j] = (
                    "NaN"
                    if (
                        original_mutual_information[i][j] == "NaN"
                        or synthetic_mutual_information[i][j] == "NaN"
                    )
                    else float(
                        round(
                            abs(
                                original_mutual_information[i][j]
                                - synthetic_mutual_information[i][j]
                            ),
                            4,
                        )
                    )
                )

        # Store results in the dictionary
        mutual_information["original_mutual_information"] = original_mutual_information
        mutual_information["synthetic_mutual_information"] = synthetic_mutual_information
        mutual_information["diff_correlation_matrix"] = diff_correlation_matrix

        # Calculate overall similarity score
        score = sum(
            sum(float(value) for value in row if value != "NaN")
            for row in diff_correlation_matrix
        )
        mutual_information["score"] = round(
            float(1 - score / (len(mutual_information["features"]) ** 2)), 4
        )

        return mutual_information
