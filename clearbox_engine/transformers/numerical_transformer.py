"""
numerical_transformer.py

This module contains the NumericalTransformer class, which is a custom scikit-learn transformer 
used for preprocessing ordinal features. It handles imputation of missing values, scaling, 
and optional binning of features for ordinal transformation.

Dependencies:
    - numpy
    - sklearn.base.BaseEstimator
    - sklearn.base.TransformerMixin
    - sklearn.impute.SimpleImputer
    - sklearn.preprocessing.MinMaxScaler
    - sklearn.preprocessing.KBinsDiscretizer
    - sklearn.preprocessing.PowerTransformer
    - sklearn.preprocessing.QuantileTransformer
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    KBinsDiscretizer,
    PowerTransformer,
    QuantileTransformer,
)


class NumericalTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for handling ordinal features in data preprocessing. This class 
    provides options for binning features, scaling, and imputing missing values.

    Attributes:
        n_bins (int): Number of bins for discretization. If 0, no binning is applied.
        transformer_type (str): Type of transformer to use for scaling. Options include 
                                "Quantile", "Power", and "MinMax".
        na_fill_value (float or None): Value used to fill missing values. If None, uses the 
                                       "most_frequent" strategy.
        scaler (object): The scaler object used for transforming the data.
        min (array-like): Minimum values of the features, used for normalization.
        max (array-like): Maximum values of the features, used for normalization.
    """

    def __init__(self, n_bins: int = 0, transformer_type="Quantile", na_fill_value=None) -> None:
        """
        Initializes the NumericalTransformer with the specified parameters.

        Args:
            n_bins (int): Number of bins for discretization. Defaults to 0 (no binning).
            transformer_type (str): Type of transformer to use for scaling. Options are "Quantile", 
                                    "Power", and "MinMax". Defaults to "Quantile".
            na_fill_value (float or None): Value used to fill missing values. If None, uses 
                                           "most_frequent" strategy. Defaults to None.
        """
        self.n_bins = n_bins
        self.transformer_type = transformer_type
        self.na_fill_value = na_fill_value
        self.scaler = None
        self.min = None
        self.max = None

    def fit(self, X, y=None):
        """
        Fits the transformer to the data. It computes the minimum and maximum values 
        for normalization, imputes missing values, and fits the scaler or discretizer 
        based on the configuration.

        Args:
            X (array-like): The input data to fit.
            y (array-like, optional): The target values (not used).

        Returns:
            self: The fitted NumericalTransformer object.
        """
        data = X.copy()
        
        # Determine the strategy for imputing missing values
        if self.na_fill_value is None:
            strategy = "most_frequent"
        else:
            strategy = "constant"

        # Compute min and max for normalization
        self.min = np.nanmin(data, axis=0)
        self.max = np.nanmax(data, axis=0)

        # Normalize the data
        data = (data - self.min) / (self.max - self.min)

        # Impute missing values
        self.imputer = SimpleImputer(
            strategy=strategy, add_indicator=False, fill_value=self.na_fill_value
        )
        self.imputer.fit(data)
        data = self.imputer.transform(data)

        # Apply binning or scaling based on the configuration
        if self.n_bins > 0:
            self.est = KBinsDiscretizer(
                n_bins=self.n_bins, encode="ordinal", strategy="kmeans"
            )
            self.est.fit(data)
            data = self.est.transform(data)
        else:
            if self.transformer_type == "Power":
                self.scaler = PowerTransformer()
            elif self.transformer_type == "Quantile":
                self.scaler = QuantileTransformer(
                    output_distribution="normal", random_state=0
                )
            else:
                self.scaler = MinMaxScaler()

            self.scaler.fit(data)
            data = self.scaler.transform(data)

        return self

    def get_feature_names(self):
        """
        Returns an empty list since this transformer does not generate new feature names.

        Returns:
            list: An empty list.
        """
        return []

    def transform(self, X, y=None):
        """
        Transforms the input data by normalizing, imputing missing values, and 
        applying binning or scaling.

        Args:
            X (array-like): The input data to transform.
            y (array-like, optional): The target values (not used).

        Returns:
            array-like: The transformed data.
        """
        X = X.copy()

        # Normalize the data
        X = (X - self.min) / (self.max - self.min)

        # Impute missing values
        X = self.imputer.transform(X)

        # Apply binning or scaling
        if self.n_bins > 0:
            X = self.est.transform(X)
        else:
            X = self.scaler.transform(X)

        return X

    def inverse_transform(self, X, y=None):
        """
        Inversely transforms the data back to its original scale.

        Args:
            X (array-like): The transformed data to inverse transform.
            y (array-like, optional): The target values (not used).

        Returns:
            array-like: The inversely transformed data.
        """
        X = X.copy()

        # Apply inverse transformation for binning or scaling
        if self.n_bins > 0:
            X = self.est.inverse_transform(X)
        else:
            X = self.scaler.inverse_transform(X)

        # Replace values below the fill value with NaN
        X[X <= self.na_fill_value] = np.nan

        # Denormalize the data
        X = self.min + (self.max - self.min) * X

        return X
