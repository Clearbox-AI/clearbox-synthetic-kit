import pandas as pd
from clearbox_synthetic.utils.dataset.dataset import Dataset
from clearbox_synthetic.utils.preprocessor.preprocessor import Preprocessor


class FeaturesComparison:
    """
    The FeaturesComparison class compares the statistical
    properties of features between an original dataset and a synthetic dataset. It
    provides detailed statistics for numerical, categorical, and datetime features.

    Attributes:
        original_dataset (Dataset): The original dataset.
        synthetic_dataset (Dataset): The synthetic dataset.
        preprocessor (Preprocessor): The preprocessor for handling feature extraction and transformation.
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

        Args:
            original_dataset (Dataset): The original dataset object.
            synthetic_dataset (Dataset): The synthetic dataset object.
            preprocessor (Preprocessor, optional): Preprocessor for handling feature extraction.
                                                   Defaults to None.
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

        Args:
            features_to_hide (list, optional): List of features to exclude from the comparison.
                                               Defaults to an empty list.

        Returns:
            dict: A dictionary containing the statistical comparison for each feature.
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
