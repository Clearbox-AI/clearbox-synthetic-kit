"""
The FeaturesComparison class compares the statistical
properties of features between an original dataset and a synthetic dataset. It
provides detailed statistics for numerical, categorical, and datetime features.
"""

import pandas as pd
from clearbox_synthetic.utils.dataset.dataset import Dataset
from clearbox_synthetic.utils.preprocessor.preprocessor import Preprocessor


class FeaturesComparison:
    """
    The FeaturesComparison class compares the statistical
    properties of features between an original dataset and a synthetic dataset. It
    provides detailed statistics for numerical, categorical, and datetime features.

    Attributes
    ----------
    original_dataset : Dataset
        The original dataset containing real-world data.
    synthetic_dataset : Dataset
        The synthetic dataset generated for evaluation.
    preprocessor : Preprocessor
        The preprocessor responsible for handling feature extraction and transformation.

    Methods
    -------
    get(features_to_hide: list = []) -> dict
        Computes detailed statistics for numerical, categorical, and datetime features, 
        comparing the original and synthetic datasets.
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
        Initializes the FeaturesComparison class.

        Parameters
        ----------
        original_dataset : Dataset
            The original dataset containing real-world data.
        synthetic_dataset : Dataset
            The synthetic dataset generated for evaluation.
        preprocessor : Preprocessor, optional
            The preprocessor responsible for handling feature extraction and transformation.
            If None, a default preprocessor is used. Default is None.
        """
        self.original_dataset = original_dataset
        self.synthetic_dataset = synthetic_dataset
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor(original_dataset)
        )

    def get(self, features_to_hide: list = []) -> dict:
        """
        Compares features between the original and synthetic datasets, providing statistics
        such as mean, standard deviation, quartiles, and unique values.

        Parameters
        ----------
        features_to_hide : list, optional
            List of features to exclude from the comparison. 
            Defaults to an empty list.

        Returns
        -------
        dict 
            A dictionary containing the statistical comparison for each feature.

        Process
        -------
        1. Numerical Feature Analysis

            - Computes mean, standard deviation, quartiles, min/max values.
            - Measures data spread and central tendency.
            - Tracks missing values for completeness analysis.

        2. Categorical Feature Analysis

            - Measures unique values and categorical distributions.
            - Identifies the most frequent categories and their relative frequency.

        3. Datetime Feature Analysis
            - Compares date ranges (min/max).
            - Identifies most common timestamps.
            - Tracks missing datetime values.

        4. Returns a structured dictionary summarizing the differences between the original and synthetic datasets.
        
        >>> The get method returns a dictionary in the following format:
        {
            "age": {
                "type": "number",
                "na_values": {"training": 10, "synthetic": 12},
                "unique_values": {"training": 35, "synthetic": 34},
                "mean": {"training": 45.2, "synthetic": 44.8},
                "std": {"training": 12.1, "synthetic": 11.9},
                "min": {"training": 18, "synthetic": 20},
                "first_quartile": {"training": 30.0, "synthetic": 29.5},
                "second_quartile": {"training": 45.0, "synthetic": 44.0},
                "third_quartile": {"training": 60.0, "synthetic": 59.5},
                "max": {"training": 85, "synthetic": 83}
            },
            "gender": {
                "type": "categorical",
                "na_values": {"training": 0, "synthetic": 0},
                "unique_values": {"training": 2, "synthetic": 2},
                "values": {
                    "training": [{"value": "Male", "count": 550, "freq": 0.55},
                                {"value": "Female", "count": 450, "freq": 0.45}],
                    "synthetic": [{"value": "Male", "count": 540, "freq": 0.54},
                                {"value": "Female", "count": 460, "freq": 0.46}]
                }
            },
            "purchase_date": {
                "type": "datetime",
                "na_values": {"training": 5, "synthetic": 6},
                "unique_values": {"training": 400, "synthetic": 398},
                "min": {"training": "2015-01-01", "synthetic": "2015-01-02"},
                "max": {"training": "2022-12-31", "synthetic": "2022-12-30"},
                "most_frequent": {"training": "2020-06-15", "synthetic": "2020-06-16"}
            }
        }

        Notes
        -----
        - A close match in statistical properties between original and synthetic data 
        suggests a high-quality synthetic dataset.
        - Significant deviations indicate potential inconsistencies in synthetic data generation.
        """
        features_comparison = {}

        # Compare ordinal (numerical) features
        for feature in self.preprocessor.get_numerical_features():
            if feature not in features_to_hide:
                features_comparison[feature] = {
                    "type": "number",
                    "na_values": {
                        "training": "NaN" if pd.isnull(
                            float(self.original_dataset.data[feature].isnull().sum())
                        ) else round(float(self.original_dataset.data[feature].isnull().sum()), 4),
                        "synthetic": "NaN" if pd.isnull(
                            float(self.synthetic_dataset.data[feature].isnull().sum())
                        ) else round(float(self.synthetic_dataset.data[feature].isnull().sum()), 4),
                    },
                    "unique_values": {
                        "training": "NaN" if pd.isnull(
                            float(self.original_dataset.data[feature].nunique())
                        ) else round(float(self.original_dataset.data[feature].nunique()), 4),
                        "synthetic": "NaN" if pd.isnull(
                            float(self.synthetic_dataset.data[feature].nunique())
                        ) else round(float(self.synthetic_dataset.data[feature].nunique()), 4),
                    },
                    "mean": {
                        "training": "NaN" if pd.isnull(
                            float(self.original_dataset.data[feature].mean())
                        ) else round(float(self.original_dataset.data[feature].mean()), 4),
                        "synthetic": "NaN" if pd.isnull(
                            float(self.synthetic_dataset.data[feature].mean())
                        ) else round(float(self.synthetic_dataset.data[feature].mean()), 4),
                    },
                    "std": {
                        "training": "NaN" if pd.isnull(
                            float(self.original_dataset.data[feature].std())
                        ) else round(float(self.original_dataset.data[feature].std()), 4),
                        "synthetic": "NaN" if pd.isnull(
                            float(self.synthetic_dataset.data[feature].std())
                        ) else round(float(self.synthetic_dataset.data[feature].std()), 4),
                    },
                    "min": {
                        "training": "NaN" if pd.isnull(
                            float(self.original_dataset.data[feature].min())
                        ) else round(float(self.original_dataset.data[feature].min()), 4),
                        "synthetic": "NaN" if pd.isnull(
                            float(self.synthetic_dataset.data[feature].min())
                        ) else round(float(self.synthetic_dataset.data[feature].min()), 4),
                    },
                    "first_quartile": {
                        "training": "NaN" if pd.isnull(
                            float(self.original_dataset.data[feature].quantile(0.25))
                        ) else round(float(self.original_dataset.data[feature].quantile(0.25)), 4),
                        "synthetic": "NaN" if pd.isnull(
                            float(self.synthetic_dataset.data[feature].quantile(0.25))
                        ) else round(float(self.synthetic_dataset.data[feature].quantile(0.25)), 4),
                    },
                    "second_quartile": {
                        "training": "NaN" if pd.isnull(
                            float(self.original_dataset.data[feature].quantile(0.5))
                        ) else round(float(self.original_dataset.data[feature].quantile(0.5)), 4),
                        "synthetic": "NaN" if pd.isnull(
                            float(self.synthetic_dataset.data[feature].quantile(0.5))
                        ) else round(float(self.synthetic_dataset.data[feature].quantile(0.5)), 4),
                    },
                    "third_quartile": {
                        "training": "NaN" if pd.isnull(
                            float(self.original_dataset.data[feature].quantile(0.75))
                        ) else round(float(self.original_dataset.data[feature].quantile(0.75)), 4),
                        "synthetic": "NaN" if pd.isnull(
                            float(self.synthetic_dataset.data[feature].quantile(0.75))
                        ) else round(float(self.synthetic_dataset.data[feature].quantile(0.75)), 4),
                    },
                    "max": {
                        "training": "NaN" if pd.isnull(
                            float(self.original_dataset.data[feature].max())
                        ) else round(float(self.original_dataset.data[feature].max()), 4),
                        "synthetic": "NaN" if pd.isnull(
                            float(self.synthetic_dataset.data[feature].max())
                        ) else round(float(self.synthetic_dataset.data[feature].max()), 4),
                    },
                }

        # Compare categorical features
        for feature in self.preprocessor.get_categorical_features():
            if feature not in features_to_hide:
                features_comparison[feature] = {
                    "type": "categorical",
                    "na_values": {
                        "training": "NaN" if pd.isnull(
                            float(self.original_dataset.data[feature].isnull().sum())
                        ) else round(float(self.original_dataset.data[feature].isnull().sum()), 4),
                        "synthetic": "NaN" if pd.isnull(
                            float(self.synthetic_dataset.data[feature].isnull().sum())
                        ) else round(float(self.synthetic_dataset.data[feature].isnull().sum()), 4),
                    },
                    "unique_values": {
                        "training": "NaN" if pd.isnull(
                            float(self.original_dataset.data[feature].nunique())
                        ) else round(float(self.original_dataset.data[feature].nunique()), 4),
                        "synthetic": "NaN" if pd.isnull(
                            float(self.synthetic_dataset.data[feature].nunique())
                        ) else round(float(self.synthetic_dataset.data[feature].nunique()), 4),
                    },
                    "values": {
                        "training": [
                            {
                                "value": "NaN" if pd.isnull(index) else index,
                                "count": int(row["count"]),
                                "freq": int(row["freq"]),
                            }
                            for index, row in self.original_dataset.value_counts(
                                feature
                            ).iterrows()
                        ],
                        "synthetic": [
                            {
                                "value": "NaN" if pd.isnull(index) else index,
                                "count": int(row["count"]),
                                "freq": int(row["freq"]),
                            }
                            for index, row in self.synthetic_dataset.value_counts(
                                feature
                            ).iterrows()
                        ],
                    },
                }

        # Compare datetime features
        for feature in self.preprocessor.get_datetime_features():
            if feature not in features_to_hide:
                features_comparison[feature] = {
                    "type": "datetime",
                    "na_values": {
                        "training": "NaN" if pd.isnull(
                            float(self.original_dataset.data[feature].isnull().sum())
                        ) else round(float(self.original_dataset.data[feature].isnull().sum()), 4),
                        "synthetic": "NaN" if pd.isnull(
                            float(self.synthetic_dataset.data[feature].isnull().sum())
                        ) else round(float(self.synthetic_dataset.data[feature].isnull().sum()), 4),
                    },
                    "unique_values": {
                        "training": "NaN" if pd.isnull(
                            float(self.original_dataset.data[feature].nunique())
                        ) else round(float(self.original_dataset.data[feature].nunique()), 4),
                        "synthetic": "NaN" if pd.isnull(
                            float(self.synthetic_dataset.data[feature].nunique())
                        ) else round(float(self.synthetic_dataset.data[feature].nunique()), 4),
                    },
                    "min": {
                        "training": "NaN" if pd.isnull(
                            self.original_dataset.data[feature].min()
                        ) else self.original_dataset.data[feature].min(),
                        "synthetic": "NaN" if pd.isnull(
                            self.synthetic_dataset.data[feature].min()
                        ) else self.synthetic_dataset.data[feature].min(),
                    },
                    "max": {
                        "training": "NaN" if pd.isnull(
                            self.original_dataset.data[feature].max()
                        ) else self.original_dataset.data[feature].max(),
                        "synthetic": "NaN" if pd.isnull(
                            self.synthetic_dataset.data[feature].max()
                        ) else self.synthetic_dataset.data[feature].max(),
                    },
                    "most_frequent": {
                        "training": "NaN" if pd.isnull(
                            self.original_dataset.data[feature].value_counts().idxmax()
                        ) else self.original_dataset.data[feature].value_counts().idxmax(),
                        "synthetic": "NaN" if pd.isnull(
                            self.synthetic_dataset.data[feature].value_counts().idxmax()
                        ) else self.synthetic_dataset.data[feature].value_counts().idxmax(),
                    },
                }

        return features_comparison
