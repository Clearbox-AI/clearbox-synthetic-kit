"""
This module provides the DatetimeTransformer class, a custom scikit-learn transformer
for handling datetime features. It imputes missing values and formats datetime values
for machine learning tasks.
"""

import dateinfer
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class DatetimeTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for handling datetime features in data preprocessing.
    It infers datetime formats, handles missing values using median imputation,
    and formats datetime values for use in machine learning models.

    Attributes:
        datetime_formats (list): List to store inferred datetime formats for each column.
        dividers (list): List to store dividers used to normalize datetime values.
    """

    def __init__(self) -> None:
        """Initializes the DatetimeTransformer object."""
        self.datetime_formats = []
        self.dividers = []

    def _find_datetime_format(self, data):
        """
        Infers the datetime format for each column in the given data.

        Args:
            data (DataFrame): The input data containing datetime columns.

        Raises:
            Exception: If the datetime format cannot be inferred, "N" is added to the list.
        """
        try:
            for datetime_column in data:
                self.datetime_formats.append(
                    dateinfer.infer(
                        data[datetime_column].head(min(500, len(data))).astype(str)
                    )
                )
        except Exception:
            self.datetime_formats.append("N")

    def _find_divider(self, data):
        """
        Determines an appropriate divider to normalize datetime values.

        Args:
            data (array-like): The input datetime data.

        Returns:
            float: The divider used to normalize datetime values.
        """
        if (data > 1e9).any():
            return 1e9
        else:
            return 1

    def fit(self, X, y=None):
        """
        Fits the transformer to the data. It infers datetime formats, imputes missing values,
        and computes dividers to normalize datetime values.

        Args:
            X (DataFrame): The input data to fit.
            y (array-like, optional): The target values (not used).

        Returns:
            self: The fitted DatetimeTransformer object.
        """
        self._find_datetime_format(X)

        data = X.copy().astype("datetime64", errors="ignore").astype("int")
        self.imputer = SimpleImputer(strategy="median")
        self.imputer.fit(data)

        data = self.imputer.transform(data.astype("int", errors="ignore"))
        for i in range(data.shape[-1]):
            divider = self._find_divider(data[:, i])
            self.dividers.append(divider)
            data[:, i] = data[:, i] // divider

        return self

    def get_feature_names(self):
        """
        Returns an empty list as the transformer does not generate new feature names.

        Returns:
            list: An empty list.
        """
        return []

    def transform(self, X, y=None):
        """
        Transforms the input datetime data by imputing missing values and normalizing
        using precomputed dividers.

        Args:
            X (DataFrame): The input data to transform.
            y (array-like, optional): The target values (not used).

        Returns:
            array-like: The transformed data.
        """
        X = X.copy().astype("datetime64", errors="ignore").astype("int")
        X = self.imputer.transform(X)
        for i in range(X.shape[-1]):
            X[:, i] = X[:, i] // self.dividers[i]

        return X

    def inverse_transform(self, X, y=None):
        """
        Converts the normalized datetime values back to their original datetime format.

        Args:
            X (array-like): The normalized datetime data to inverse transform.
            y (array-like, optional): The target values (not used).

        Returns:
            array-like: The original datetime values as strings.
        """
        X = X.copy()
        datetimes = np.empty(shape=X.shape, dtype="object")

        for i in range(X.shape[-1]):
            for j, date in enumerate(X[:, i]):
                datetimes[:, i][j] = datetime.fromtimestamp(date).strftime(
                    self.datetime_formats[i]
                )

        return datetimes
