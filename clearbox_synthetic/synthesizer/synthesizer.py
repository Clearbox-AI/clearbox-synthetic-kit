"""
This module defines the Synthesizer abstract base class, which provides a framework for generating
synthetic datasets using various techniques. The class is intended to be extended by specific
implementations of data synthesizers.
"""

import abc
import numpy as np
import pandas as pd
import scipy

from clearbox_synthetic.dataset.dataset import Dataset
from clearbox_synthetic.engine.tabular_engine import TabularEngine
from clearbox_synthetic.preprocessor.preprocessor import Preprocessor


class Synthesizer(metaclass=abc.ABCMeta):
    """
    Abstract base class for synthesizers used to generate synthetic data from a dataset.

    Attributes:
        dataset (Dataset): The dataset to be used for generating synthetic data.
        preprocessor (Preprocessor): The preprocessor for transforming data.
        engine (TabularEngine): The engine used for data synthesis.
        sampled_indexes (np.ndarray): Indexes of the sampled instances from the dataset.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        """
        Check if a subclass implements the required methods.
        """
        return hasattr(subclass, "generate") and callable(subclass.fit)

    def __init__(self, dataset: Dataset, engine: TabularEngine, preprocessor: Preprocessor = None):
        """
        Initialize the Synthesizer.

        Args:
            dataset (Dataset): The dataset to be used for generating synthetic data.
            engine (TabularEngine): The engine used for data synthesis.
            preprocessor (Preprocessor, optional): The preprocessor for transforming data.
                Defaults to None, in which case a new Preprocessor is created.
        """
        self.dataset = dataset
        self.preprocessor = preprocessor if preprocessor is not None else Preprocessor(dataset)
        self.engine = engine
        self.sampled_indexes = None

    def _sample_vae(self, x, recon_x):
        """
        Sample data from a Variational Autoencoder (VAE) using the original and reconstructed data.

        Args:
            x (np.ndarray): Original input data.
            recon_x (np.ndarray): Reconstructed data from the VAE.

        Returns:
            pd.DataFrame: The inverse-transformed synthetic data.
        """
        preprocessed_x = self.preprocessor.transform(x)

        n_numerical_features = (
            self.preprocessor.get_features_sizes()[0][0] if self.preprocessor.get_features_sizes()[0] else 0
        )
        categorical_features_sizes = self.preprocessor.get_features_sizes()[1]

        numerical_features_sampled = np.zeros((preprocessed_x.shape[0], n_numerical_features))

        for i in range(n_numerical_features):
            if isinstance(preprocessed_x[:, i], scipy.sparse.csr_matrix):
                converted_input = preprocessed_x[:, i].toarray().reshape(1, -1)[0]
            else:
                converted_input = preprocessed_x[:, i]

            numerical_features_sampled[:, i] = (
                recon_x[:, i] + self.engine.search_params["gauss_s"] * np.random.randn(recon_x.shape[0])
            )

        categorical_features_sampled = np.zeros(
            (preprocessed_x.shape[0], preprocessed_x.shape[1] - n_numerical_features)
        )
        view_decoded = recon_x[:, n_numerical_features:]

        for i in range(preprocessed_x.shape[0]):
            w2 = 0  # index categorical label in preprocessed space
            w3 = 0  # index categorical feature
            features = preprocessed_x[i, n_numerical_features:] > 0
            if isinstance(features, scipy.sparse.csr_matrix):
                features = features.toarray().reshape(1, -1)[0]

            for w in categorical_features_sizes:
                if (features[w2:w2 + w]).sum() == 0:
                    # Indicates a NaN or unknown value
                    categorical_features_sampled[i, w3] = 0.0
                else:
                    distribution = view_decoded[i, w2:w2 + w]
                    distribution = np.asarray(distribution).astype("float64")
                    distribution /= distribution.sum()
                    pick = np.random.choice(w, p=distribution)
                    categorical_features_sampled[i, w2 + pick] = 1.0
                w2 += w
                w3 += 1

        e = np.hstack([numerical_features_sampled, categorical_features_sampled])
        return self.preprocessor.inverse_preprocessor(e)

    def _force_temporal_precedence(self, synthetic_dataset: pd.DataFrame):
        """
        Ensure temporal precedence in the generated synthetic data based on datetime features.

        Args:
            synthetic_dataset (pd.DataFrame): The synthetic dataset to enforce temporal precedence on.
        """
        datetime_features = [
            column
            for column in self.dataset.column_types.keys()
            if self.dataset.column_types[column] == "datetime"
        ]
        if datetime_features:
            sample = self.dataset.data[datetime_features].head(1)
            column_names = sample.columns.tolist()
            values = sample.values.tolist()
            sorted_columns = [x for _, x in sorted(zip(values, column_names))]

            for index, row in synthetic_dataset[datetime_features].iterrows():
                column_names = row.to_frame().T.columns.tolist()
                values = row.values.tolist()
                if [x for _, x in sorted(zip(values, column_names))] != sorted_columns:
                    sorted_values = sorted(values)
                    for i, col in enumerate(column_names):
                        synthetic_dataset.loc[index, col] = sorted_values[i]

    @abc.abstractmethod
    def generate(self, has_header: bool = None):
        """
        Abstract method to generate new synthetic data.

        Args:
            has_header (bool, optional): Whether the generated data should include headers. Defaults to None.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
