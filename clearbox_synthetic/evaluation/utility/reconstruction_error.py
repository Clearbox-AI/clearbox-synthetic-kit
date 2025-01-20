"""
The ReconstructionError class calculates the reconstruction
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

    Attributes:
        original_dataset (Dataset): The original dataset.
        synthetic_dataset (Dataset): The synthetic dataset.
        preprocessor (Preprocessor): The preprocessor for handling data transformation.
        engine (TabularEngine): The engine used for calculating reconstruction error.
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
        Initializes the ReconstructionError class.

        Args:
            original_dataset (Dataset): The original dataset object.
            synthetic_dataset (Dataset): The synthetic dataset object.
            engine (TabularEngine): The engine used to compute reconstruction error.
            preprocessor (Preprocessor, optional): The preprocessor for data transformation.
                                                   Defaults to None, using a default 
                                                   preprocessor for the original dataset.
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

        Returns:
            dict: A dictionary containing bin edges and the histograms of reconstruction
                  errors for both the original and synthetic datasets.
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
