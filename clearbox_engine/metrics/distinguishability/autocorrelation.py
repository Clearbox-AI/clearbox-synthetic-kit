"""
autocorrelation.py

This module provides functionality to compute and compare the autocorrelation 
between original and synthetic datasets using the Autocorrelation class.

Dependencies:
    - json
    - pandas
    - numpy
    - clearbox_engine (Dataset, Preprocessor)
"""

import json
import pandas as pd
import numpy as np
from clearbox_engine import Dataset, Preprocessor


def _autocorr(x: pd.Series) -> np.ndarray:
    """
    Computes the autocorrelation of a given time series.

    Args:
        x (pd.Series): Input time series data.

    Returns:
        np.ndarray: Autocorrelation values.
    """
    result = np.correlate(x, x, mode="full")
    return result[result.size // 2:]


class Autocorrelation:
    """
    A class to compute and compare autocorrelation between original and synthetic datasets.

    Attributes:
        original_dataset (Dataset): The original dataset object.
        synthetic_dataset (Dataset): The synthetic dataset object.
        preprocessor (Preprocessor): Preprocessor for handling the dataset.
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
        Initializes the Autocorrelation class.

        Args:
            original_dataset (Dataset): The original dataset object.
            synthetic_dataset (Dataset): The synthetic dataset object.
            preprocessor (Preprocessor, optional): Preprocessor for handling the dataset. 
                                                   Defaults to None.
        """
        self.original_dataset = original_dataset
        self.synthetic_dataset = synthetic_dataset
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor(original_dataset)
        )

    def get(self, feature: str, id: str = None) -> dict:
        """
        Computes the autocorrelation for a specified feature and compares it 
        between the original and synthetic datasets.

        Args:
            feature (str): The feature for which autocorrelation is computed.
            id (str, optional): Identifier for grouping data (used for sequence analysis). 
                                Defaults to None.

        Returns:
            dict: A dictionary containing autocorrelation results and areas under the curve 
                  for both original and synthetic data.
        """
        # Process original data
        original_data = self.original_dataset.data.copy()
        if self.original_dataset.sequence_index:
            original_data = original_data.set_index(self.original_dataset.sequence_index)
        if id:
            original_data = original_data.loc[
                original_data[self.original_dataset.group_by] == id
            ]

        original_x = np.array(original_data[feature])
        original_z = _autocorr(original_x)
        original_z = original_z / float(original_z.max())
        original_area = round(float(np.trapz(original_z)), 4)

        # Process synthetic data
        synthetic_data = self.synthetic_dataset.data.copy()
        if self.original_dataset.sequence_index:
            synthetic_data = synthetic_data.set_index(self.original_dataset.sequence_index)
        if id and self.original_dataset.group_by:
            synthetic_data = synthetic_data.loc[
                synthetic_data[self.original_dataset.group_by] == id
            ]

        synthetic_x = np.array(synthetic_data[feature])
        synthetic_z = _autocorr(synthetic_x)
        synthetic_z = synthetic_z / float(synthetic_z.max())
        synthetic_area = round(float(np.trapz(synthetic_z)), 4)

        # Compile results
        autocorrelation = {
            "original": json.dumps(original_z.tolist()),
            "original_area": original_area,
            "synthetic": json.dumps(synthetic_z.tolist()),
            "synthetic_area": synthetic_area,
            "diff_area": round(float(abs(original_area - synthetic_area)), 4),
        }

        return autocorrelation
