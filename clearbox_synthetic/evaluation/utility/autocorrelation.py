"""
The ``Autocorrelation`` class provides a tool for validating synthetic time-series data. 
By measuring how well synthetic datasets preserve temporal dependencies, it ensures 
that models trained on synthetic data generalize well to real-world scenarios.
"""

import json
import pandas as pd
import numpy as np
from clearbox_synthetic.utils.dataset.dataset import Dataset
from clearbox_synthetic.utils.preprocessor.preprocessor import Preprocessor


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
    Provides functionality to compute and compare the autocorrelation between original and synthetic datasets.

    Attributes
    ----------
    original_dataset : Dataset
        The original dataset containing real-world time-series data.
    synthetic_dataset : Dataset
        The synthetic dataset generated for evaluation.
    preprocessor : Preprocessor
        The preprocessor responsible for handling feature extraction and transformation.
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
        Initializes the Autocorrelation class with the original and synthetic datasets.

        Parameters
        ----------
        original_dataset : Dataset
            The original dataset containing real-world time-series data.
        synthetic_dataset : Dataset
            The synthetic dataset generated for evaluation.
        preprocessor : Preprocessor, optional
            The preprocessor responsible for handling feature extraction and transformation.
            If None, a default preprocessor is used. Default is None
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

        Parameters
        ----------
        feature : str
            The feature for which autocorrelation is computed.
        id : str, optional
            Identifier for grouping data (used for sequence-based analysis). Defaults to None.

        Returns
        -------
        dict
            A dictionary containing autocorrelation results and areas under the curve for both original and synthetic data.

        Notes
        -----
        The method operates through the following steps:

        1. Extracts and preprocesses the feature from both datasets.
        2. Computes the autocorrelation curve for both the original and synthetic feature values.
        3. Normalizes the curves to ensure they are on the same scale.
        4. Calculates the area under the autocorrelation curve (AUC) using numerical integration.
        5. Computes the absolute difference (`diff_area`) between original and synthetic AUCs.
        6. Returns results in a structured dictionary.

        Examples
        --------
        Example of dictionary returned:

        .. code-block:: python

            >>> results = autocorrelation.get(feature="temperature")
            >>> print(results)
            {
                "original": "[autocorrelation values of original dataset]",
                "original_area": 2.354,  # AUC of original dataset
                "synthetic": "[autocorrelation values of synthetic dataset]",
                "synthetic_area": 2.290,  # AUC of synthetic dataset
                "diff_area": 0.064  # Difference in AUC
            }

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
