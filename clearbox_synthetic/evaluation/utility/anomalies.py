"""
This module provides functionality to detect anomalies in tabular data using
an engine that computes reconstruction errors, a dataset for the input data, 
and a preprocessor for transforming the data. The main class `Anomalies` 
provides methods to detect anomalies and to calculate feature-wise anomaly 
probabilities.
"""

import scipy.sparse

import numpy as np
import pandas as pd

from clearbox_synthetic.utils.dataset.dataset import Dataset
from clearbox_synthetic.utils.preprocessor.preprocessor import Preprocessor
from clearbox_synthetic.generation import TabularEngine


class Anomalies:
    """Class to detect anomalies in tabular data using a provided engine, dataset, and preprocessor.

    Attributes:
        dataset (Dataset): The dataset object containing the data.
        preprocessor (Preprocessor): The preprocessor object used to transform the data.
        engine (TabularEngine): The engine object used to compute reconstruction errors.
    """
    dataset: Dataset
    preprocessor: Preprocessor
    engine: TabularEngine

    def __init__(
        self, dataset: Dataset, engine: TabularEngine, preprocessor: Preprocessor = None
    ):
        """Initializes the Anomalies class with a dataset, engine, and optional preprocessor.

        Args:
            dataset (Dataset): The dataset object containing the data.
            engine (TabularEngine): The engine object used to compute reconstruction errors.
            preprocessor (Preprocessor, optional): The preprocessor object to transform data. If not provided,
                a default preprocessor is created using the dataset.
        """
        self.dataset = dataset
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor(dataset)
        )
        self.engine = engine

    def detect(self, n: int = 10):
        """Detects anomalies in the dataset based on reconstruction error.

        Args:
            n (int, optional): The number of top anomalies to detect. Defaults to 10.

        Returns:
            dict: A dictionary containing:
                - "values": A list of feature values for each anomaly.
                - "anomaly_probabilities": A list of anomaly probabilities for each feature.
        """
        preprocessed_data = self.preprocessor.transform(self.dataset.get_x())
        reconstruction_error = self.engine.reconstruction_error(preprocessed_data)

        anomaly_instances = np.argsort(reconstruction_error)[::-1][:n]

        anomaly_features = self.get_anomaly_features(
            self.dataset.get_x().iloc[anomaly_instances]
        )

        features_values = []
        for anomaly_index in anomaly_instances:
            features = []
            for value in self.dataset.get_x().iloc[anomaly_index].values.tolist():
                features.append(
                    "NaN"
                    if pd.isnull(value)
                    else value
                    if isinstance(value, str)
                    else str(value)
                    if isinstance(value, bool)
                    else float(value)
                    if isinstance(value, float)
                    else int(value)
                )
            features_values.append(features)

        anomalies = {
            "values": features_values,
            "anomaly_probabilities": anomaly_features,
        }

        return anomalies

    def get_anomaly_features(self, X: pd.DataFrame) -> list:
        """Calculates anomaly features for the given data.

        Args:
            X (pd.DataFrame): The data for which anomaly features need to be computed.

        Returns:
            list: A list of anomaly feature values for each instance.
        """
        preprocessed_data = self.preprocessor.transform(X)
        recon_x, _, _ = self.engine.apply(preprocessed_data)

        n_numerical_features = (
            self.preprocessor.get_features_sizes()[0][0]
            if self.preprocessor.get_features_sizes()[0]
            else 0
        )
        categorical_features_sizes = self.preprocessor.get_features_sizes()[1]

        numerical_anomaly_features = np.zeros(
            (preprocessed_data.shape[0], n_numerical_features)
        )

        for i in range(n_numerical_features):
            if isinstance(preprocessed_data[:, i], scipy.sparse.csr_matrix):
                converted_input = preprocessed_data[:, i].toarray().reshape(1, -1)[0]
            else:
                converted_input = preprocessed_data[:, i]
            numerical_anomaly_features[:, i] = np.exp(
                -(((converted_input - recon_x[:, i]) / 0.1) ** 2) / 2.0
            )

        categorical_anomaly_features = np.zeros(
            (
                preprocessed_data.shape[0],
                preprocessed_data.shape[1] - n_numerical_features,
            )
        )
        view_decoded = recon_x[:, n_numerical_features:]

        for i in range(preprocessed_data.shape[0]):
            w2 = 0  # index categorical label in preprocessed space
            w3 = 0  # index categorical feature
            features = preprocessed_data[i, n_numerical_features:] > 0
            if isinstance(features, scipy.sparse.csr_matrix):
                features = features.toarray().reshape(1, -1)[0]

            for w in categorical_features_sizes:
                if (features[w2 : w2 + w]).sum() == 0:
                    # it means that there's a NaN or an unknown
                    categorical_anomaly_features[i, w3] = 0.0
                else:
                    categorical_anomaly_features[i, w3] = view_decoded[
                        i, w2 + features[w2 : w2 + w].argmax()
                    ]
                w2 += w
                w3 += 1

        anomaly_features = pd.DataFrame()
        categorical_features = []
        if categorical_features_sizes:
            categorical_features = self.preprocessor.transformer.transformers[-1][2]
        numerical_index = 0
        categorical_index = 0
        discarded_columns = [i[0] for i in self.preprocessor.discarded[0]]
        for i, value in enumerate(self.dataset.x_columns()):
            if value not in discarded_columns:
                if value not in categorical_features:
                    anomaly_features[value] = np.asarray(
                        [
                            np.format_float_positional(value, precision=4)
                            for value in numerical_anomaly_features[:, numerical_index]
                        ]
                    )
                    numerical_index += 1
                else:
                    anomaly_features[value] = np.asarray(
                        [
                            np.format_float_positional(value, precision=4)
                            for value in categorical_anomaly_features[
                                :, categorical_index
                            ]
                        ]
                    )
                    categorical_index += 1

        return anomaly_features.values.tolist()
