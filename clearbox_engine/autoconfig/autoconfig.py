"""
autoconfig_module.py

This module provides functionality for automatically configuring and searching
optimal parameters for a tabular data engine. The main class, `Autoconfig`, is
used to perform grid search over various model architectures and batch sizes 
to find the best configuration based on reconstruction loss.

Functions:
    - learning_rule: Determines the learning rate, number of epochs, and batch
      size based on the size of the training data.

Classes:
    - Autoconfig: Handles data splitting and performs a grid search to determine
      the optimal configuration for a tabular model.

Dependencies:
    - math
    - threading
    - numpy
    - clearbox_engine.TabularEngine
"""

import math
import threading
from typing import List

import numpy as np

from clearbox_engine import TabularEngine


def learning_rule(training_rows_size: int):
    """Determines the learning rate, number of epochs, and batch size based on the
    size of the training data.

    Args:
        training_rows_size (int): The number of rows in the training dataset.

    Returns:
        tuple: A tuple containing (learning_rate, epochs, batch_size).
    """
    if training_rows_size < 1000:
        model_epochs = 1000
        model_batch_size = 16
    elif training_rows_size < 10000:
        model_epochs = 500
        model_batch_size = 32
    elif training_rows_size < 50000:
        model_epochs = 300
        model_batch_size = 128
    else:
        model_epochs = 100
        model_batch_size = 256

    model_learning_rate = 0.001

    return model_learning_rate, model_epochs, model_batch_size


class Autoconfig:
    """Class for automatically configuring and searching optimal parameters for a tabular engine.

    Attributes:
        train_ds (np.ndarray): The training dataset.
        y_train_ds (np.ndarray, optional): The target values for the training dataset.
        numerical_features_sizes (int): The size of ordinal features.
        categorical_features_sizes (List): The sizes of categorical features.
    """

    def __init__(
        self,
        train_ds: np.ndarray,
        numerical_features_sizes: int,
        categorical_features_sizes: List,
        y_train_ds: np.ndarray = None,
    ):
        """Initializes the Autoconfig class, splits the data into training and test sets,
        and sets feature sizes.

        Args:
            train_ds (np.ndarray): The complete dataset for training.
            numerical_features_sizes (int): The size of ordinal features.
            categorical_features_sizes (List): The sizes of categorical features.
            y_train_ds (np.ndarray, optional): The target values for the training dataset.
                Defaults to None.
        """
        splitted_train_ds = np.split(
            train_ds, [math.ceil(train_ds.shape[0] * 0.8)], axis=0
        )
        self.train_ds = splitted_train_ds[0]
        self.test_ds = splitted_train_ds[1]
        if y_train_ds is not None:
            splitted_y_train_ds = np.split(
                y_train_ds, [math.ceil(y_train_ds.shape[0] * 0.8)], axis=0
            )
            self.y_train_ds = splitted_y_train_ds[0]
            self.y_test_ds = splitted_y_train_ds[1]
        else:
            self.y_train_ds = None
            self.y_test_ds = None
        self.numerical_features_sizes = numerical_features_sizes
        self.categorical_features_sizes = categorical_features_sizes

    def grid_search(self):
        """Performs a grid search to find the optimal model configuration.

        The grid search iterates over different model architectures and batch
        sizes, fitting the model using multiple threads, and evaluates each model
        to determine the configuration with the lowest mean reconstruction loss.

        Returns:
            list: The optimal configuration (architecture and batch size) based on
            the evaluation loss.
        """
        features_size = self.train_ds.shape[1]
        rows_number = self.train_ds.shape[0]

        if features_size < 16:
            architectures = [
                [min(2, math.ceil(features_size / 4))],
            ]
        elif features_size < 64:
            architectures = [
                [math.ceil(features_size / 2), 4],
            ]
        else:
            architectures = [
                [math.ceil(features_size / 2), 8],
            ]

        batch_sizes = [128, 256]
        if rows_number < 512:
            batch_sizes = [16]

        grid_search = []
        for architecture in architectures:
            for batch in batch_sizes:
                grid_search.append([architecture, batch])

        processes = []
        engines = []
        losses = []

        for i, (architecture, batch_size) in enumerate(grid_search):
            engines.append(
                TabularEngine(
                    layers_size=architecture,
                    x_shape=self.train_ds[0].shape,
                    y_shape=self.y_train_ds[0].shape
                    if self.y_train_ds is not None
                    else [0],
                    numerical_feature_sizes=self.numerical_features_sizes,
                    categorical_feature_sizes=self.categorical_features_sizes,
                )
            )

            p = threading.Thread(
                target=engines[i].fit,
                args=(
                    self.train_ds,
                    self.y_train_ds if self.y_train_ds is not None else None,
                    5,
                    batch_size,
                    1e-2,
                ),
            )

            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for i, engine in enumerate(engines):
            losses.append(
                engine.evaluate(
                    self.test_ds, self.y_test_ds if self.y_test_ds is not None else None
                )["mean_reconstruction_loss"]
            )

        del engines

        _, idx = min((val, idx) for (idx, val) in enumerate(losses))

        return grid_search[idx]
