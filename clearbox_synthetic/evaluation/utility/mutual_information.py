"""
The ``MutualInformation`` module provides a robust method for assessing the 
statistical similarity of feature relationships between original and synthetic datasets. 
By computing mutual information scores and comparing feature dependencies, it ensures 
that synthetic data maintains meaningful structure while remaining privacy-compliant.
"""

import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score
from clearbox_synthetic.utils.dataset.dataset import Dataset
from clearbox_synthetic.utils.preprocessor.preprocessor import Preprocessor


class MutualInformation:
    """
    A class for calculating and comparing the mutual information between features in 
    original and synthetic datasets.

    Attributes
    ----------
    original_dataset : Dataset
        The original dataset containing real-world data.
    synthetic_dataset : Dataset
        The synthetic dataset generated for evaluation.
    preprocessor : Preprocessor
        The preprocessor responsible for handling data transformation before computing 
        mutual information.
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
        Initializes the MutualInformation class with both original and synthetic datasets.

        Parameters
        ----------
        original_dataset : Dataset
            The original dataset containing real-world data.
        synthetic_dataset : Dataset
            The synthetic dataset generated for evaluation.
        preprocessor : Preprocessor, optional
            The preprocessor responsible for handling data transformation.
            If None, a default preprocessor with 10 ordinal bins is used. 
            Default is None.
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
        Calculates mutual information matrices for features in both datasets and compares them.

        Parameters
        ----------
        features_to_hide : list, optional
            List of features to exclude from the analysis. 
            Defaults to an empty list.

        Returns
        -------
        dict 
            A dictionary containing mutual information matrices for the original and synthetic datasets, a difference matrix, and an overall similarity score.

        Examples
        --------
        
        Example of dictionary returned:

        .. code-block:: python
            
            {
                "features": ["age", "income", "education"],
                "original_mutual_information": [
                    [1.0, 0.45, 0.32],
                    [0.45, 1.0, 0.56],
                    [0.32, 0.56, 1.0]
                ],
                "synthetic_mutual_information": [
                    [1.0, 0.42, 0.29],
                    [0.42, 1.0, 0.52],
                    [0.29, 0.52, 1.0]
                ],
                "diff_correlation_matrix": [
                    [0.0, 0.03, 0.03],
                    [0.03, 0.0, 0.04],
                    [0.03, 0.04, 0.0]
                ],
                "score": 0.92  # (1 - sum of differences / total feature pairs)
            }

        .. note::
            - Mutual information values range from 0 to 1, where 0 means no dependency and 1 means perfect dependency between features.
            - Low difference matrix values and a high similarity score indicate strong alignment between the original and synthetic datasets
        
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
