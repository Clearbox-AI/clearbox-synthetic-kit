"""
The ``ReconstructionError`` class calculates the reconstruction
error for both original and synthetic datasets using a specified TabularEngine.
The reconstruction error histograms are generated to compare the quality of synthetic data.
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from clearbox_synthetic.utils.dataset.dataset import Dataset
from clearbox_synthetic.utils.preprocessor.preprocessor import Preprocessor
from clearbox_synthetic.generation import TabularEngine


class ReconstructionError:
    """
    The ReconstructionError class calculates the reconstruction
    error for both original and synthetic datasets using a specified TabularEngine.
    The reconstruction error histograms are generated to compare the quality of synthetic data.

    Attributes
    ----------
    original_dataset : Dataset
        The original dataset containing real-world data.
    synthetic_dataset : Dataset
        The synthetic dataset generated for evaluation.
    preprocessor : Preprocessor
        The preprocessor responsible for handling data transformation.
        If not provided, a default preprocessor based on the original dataset is used.
    engine : TabularEngine
        The engine used to compute reconstruction errors for input data.
    """

    original_dataset: Dataset
    synthetic_dataset: Dataset
    preprocessor: Preprocessor

    def __init__(
        self,
        original_dataset: Dataset,
        synthetic_dataset: Dataset,
        engine: TabularEngine,
        preprocessor: Preprocessor = None,
    ) -> None:
        """
        Initializes the ReconstructionError class with both original and synthetic datasets.

        Parameters
        ----------
        original_dataset : Dataset
            The original dataset containing real-world data.
        synthetic_dataset : Dataset
            The synthetic dataset generated for evaluation.
        engine : TabularEngine
            The engine used to compute reconstruction errors.
        preprocessor : Preprocessor, optional
            The preprocessor responsible for handling data transformation.
            If None, a default preprocessor based on the original dataset is used. 
            Default is None.
        """
        self.original_dataset = original_dataset
        self.synthetic_dataset = synthetic_dataset
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor(original_dataset)
        )
        self.engine = engine

    def get(self) -> dict:
        """
        Calculates and returns the reconstruction error histograms for both the original
        and synthetic datasets. The histograms allow for a comparison of how well the
        synthetic data replicates the original data's distribution.

        Returns
        -------
        dict
            A dictionary containing bin edges and the histograms of reconstruction
            errors for both the original and synthetic datasets.

        Notes
        -----
        The method operates through the following steps:

        1. Prepares the Data

            - Extracts features (``X``) and target (``Y``) from both datasets.
            - If the dataset has a target column, it applies:
                - One-hot encoding for categorical targets (classification problems).
                - Normalization for continuous targets (regression problems).
            - Applies preprocessing transformations to ensure feature consistency.
        
        2. Computes Reconstruction Error

            - Passes the transformed features (``X``) and target (``Y``) to the TabularEngine.
            - Calculates reconstruction error for both datasets.

        3. Generates Reconstruction Error Histograms

            - Creates bin edges for histogram visualization.
            - Computes histogram values for original and synthetic datasets.
            - Normalizes the histograms to ensure density comparisons.

        4. Returns Histogram Data

            - Outputs bin edges and reconstruction error distributions for visualization.

        .. note::
            A high similarity in histograms suggests that the synthetic dataset maintains feature patterns well, while large discrepancies indicate differences in feature distributions between datasets.

        Examples
        --------
        Example of dictionary returned:

        .. code-block:: python

            {
                "bin_edges": [0.01, 0.02, 0.03, ...],  # Center points of histogram bins
                "original_hist": [0.12, 0.25, 0.33, ...],  # Frequency distribution for the original dataset
                "synthetic_hist": [0.11, 0.26, 0.32, ...]  # Frequency distribution for the synthetic dataset
            }

        """
        if self.original_dataset.target_column is None:
            # Transform features only
            original_ds = self.preprocessor.transform(self.original_dataset.get_x())
            original_reconstruction_error = self.engine.reconstruction_error(
                original_ds
            )

            synthetic_ds = self.preprocessor.transform(self.synthetic_dataset.get_x())
            synthetic_reconstruction_error = self.engine.reconstruction_error(
                synthetic_ds
            )
        else:
            # Handle datasets with target columns
            if not self.original_dataset.regression:
                # Encode categorical target variable
                y_encoder = OneHotEncoder(handle_unknown="ignore")
                y_encoder.fit(self.original_dataset.get_y().to_numpy().reshape(-1, 1))

                original_y = y_encoder.transform(
                    self.original_dataset.get_y().to_numpy().reshape(-1, 1)
                ).toarray()

                synthetic_y = y_encoder.transform(
                    self.synthetic_dataset.get_y().to_numpy().reshape(-1, 1)
                ).toarray()
            else:
                # Use normalized target for regression tasks
                original_y = self.original_dataset.get_normalized_y()
                synthetic_y = self.synthetic_dataset.get_normalized_y()

            # Transform features and compute reconstruction error
            original_ds = self.preprocessor.transform(self.original_dataset.get_x())
            original_reconstruction_error = self.engine.reconstruction_error(
                original_ds, original_y
            )

            synthetic_ds = self.preprocessor.transform(self.synthetic_dataset.get_x())
            synthetic_reconstruction_error = self.engine.reconstruction_error(
                synthetic_ds, synthetic_y
            )

        # Compute histogram for reconstruction errors
        hist, bin_edges = np.histogram(
            original_reconstruction_error,
            bins=100,
            range=(
                min(
                    original_reconstruction_error.min(),
                    synthetic_reconstruction_error.min(),
                ),
                max(
                    original_reconstruction_error.max(),
                    synthetic_reconstruction_error.max(),
                ),
            ),
            density=True,
        )

        # Create bins for the histogram
        bins = [
            round(float((bin_edges[i] + bin_edges[i - 1]) / 2), 4)
            for i in range(1, len(bin_edges))
        ]

        # Generate histogram values for the original dataset
        train_hist = [round(float(value), 4) for value in hist]

        # Generate histogram values for the synthetic dataset
        hist, _ = np.histogram(
            synthetic_reconstruction_error, bins=bin_edges, density=True
        )
        synthetic_hist = [round(float(value), 4) for value in hist]

        # Construct the final histogram dictionary
        histogram = {
            "bin_edges": bins,
            "original_hist": train_hist,
            "synthetic_hist": synthetic_hist,
        }

        return histogram
