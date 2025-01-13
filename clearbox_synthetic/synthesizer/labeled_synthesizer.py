"""
This module defines the LabeledSynthesizer class, which extends the Synthesizer class to generate
new labeled instances from an existing dataset using various techniques, including shuffling and
reconstruction. The class provides methods for generating synthetic samples while considering
specific constraints, such as hybrid columns and discarded features.
"""

import os
from typing import List
import numpy as np
import pandas as pd
from clearbox_synthetic.preprocessor.preprocessor import create_series
from .synthesizer import Synthesizer


class LabeledSynthesizer(Synthesizer):
    """
    LabeledSynthesizer generates synthetic labeled instances from an existing dataset using
    sampling and shuffling techniques. It supports hybrid column handling, latent space noise
    injection, and column-specific rules for sample generation.

    Attributes:
        sampled_indexes (np.ndarray): The indexes of the sampled instances from the original dataset.
    """

    def _generate_instance(
        self,
        new_samples: pd.DataFrame,
        encoded: np.ndarray,
        X: np.ndarray,
        reshuffle_indexes: np.ndarray,
        index: int,
        sampled_index: int,
        n_sampling_points: int = 5,
        hybrid_columns: List = [],
    ):
        """
        Generates a single synthetic instance by shuffling selected columns based on distances
        in the latent space.

        Args:
            new_samples (pd.DataFrame): DataFrame to store generated samples.
            encoded (np.ndarray): Encoded representation of the dataset.
            X (np.ndarray): Original dataset features.
            reshuffle_indexes (np.ndarray): Indexes used to reshuffle columns.
            index (int): Index of the current instance being generated.
            sampled_index (int): Index of the sampled instance from the original dataset.
            n_sampling_points (int, optional): Number of sampling points for shuffling. Defaults to 5.
            hybrid_columns (List, optional): List of hybrid columns. Defaults to an empty list.
        """
        encoded_instance = encoded[sampled_index, :]
        distances = ((encoded - encoded_instance) ** 2).sum(axis=1) ** 0.5

        idx1 = np.argpartition(np.array(distances), n_sampling_points)[0:n_sampling_points]

        discarded_columns = [i for i in self.preprocessor.discarded[0]]
        columns_to_shuffle = [
            col
            for col in self.dataset.data.columns.tolist()
            if (
                col != self.dataset.target_column
                and col not in discarded_columns + hybrid_columns
            )
        ]

        for i, col in enumerate(columns_to_shuffle):
            new_samples[col].at[index] = X[col].at[idx1[reshuffle_indexes[index, i]]]

    def generate(
        self,
        has_header=None,
        points=None,
        n_sampling_points=5,
        hybrid_columns=[],
        latent_noise=0.0,
        Y=None,
    ):
        """
        Generates synthetic samples based on the original dataset and given constraints.

        Args:
            has_header (bool, optional): If the generated samples should have a header. Defaults to None.
            points (list, optional): Specific points to use for sampling. Defaults to None.
            n_sampling_points (int, optional): Number of points for sampling. Defaults to 5.
            hybrid_columns (list, optional): List of hybrid columns to consider. Defaults to an empty list.
            latent_noise (float, optional): Noise to add to the latent space representation. Defaults to 0.0.
            Y (np.ndarray, optional): Target values. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the generated synthetic samples.
        """
        X = self.dataset.get_x()

        if Y is None:
            if self.dataset.regression:
                Y = self.dataset.get_normalized_y()
            else:
                Y = self.dataset.get_one_hot_encoded_y()

        if len(hybrid_columns) == 0:
            hybrid_columns = list(X.columns)

        if points is None:
            n_samples = min(1500000, self.dataset.data.shape[0])
            sampled_indexes = np.random.choice(range(X.shape[0]), n_samples, replace=False)
        else:
            n_samples = len(points)
            sampled_indexes = np.random.choice(points, n_samples, replace=False)

        self.sampled_indexes = sampled_indexes
        data = self.dataset.get_x()
        preprocessed_data = self.preprocessor.transform(data)

        _, encoded, _ = self.engine.apply(preprocessed_data, Y)
        encoded = encoded + latent_noise * np.random.randn(*encoded.shape)

        new_samples = pd.DataFrame(
            index=range(n_samples),
            columns=list(data.columns) + [self.dataset.target_column],
        )

        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))

        discarded_columns = [i for i in self.preprocessor.discarded[0]]
        columns_to_shuffle = [
            col
            for col in self.dataset.data.columns.tolist()
            if col != self.dataset.target_column and col not in discarded_columns
        ]

        reshuffle_indexes = np.zeros((n_samples, len(columns_to_shuffle)), dtype=int)
        for i in range(len(columns_to_shuffle)):
            reshuffle_indexes[:, i] = np.random.choice(
                np.arange(0, n_sampling_points), reshuffle_indexes.shape[0]
            )

        columns_to_shuffle = [
            col
            for col in self.dataset.data.columns.tolist()
            if col != self.dataset.target_column and col not in discarded_columns + hybrid_columns
        ]

        if columns_to_shuffle:
            for i in range(n_samples):
                self._generate_instance(
                    new_samples,
                    encoded,
                    X,
                    reshuffle_indexes,
                    i,
                    sampled_indexes[i],
                    n_sampling_points,
                    hybrid_columns,
                )

        if hybrid_columns:
            recon_x = self.engine.decode(encoded, Y)
            recon_x = np.asarray(recon_x)
            out = self._sample_vae(data.iloc[sampled_indexes, :], recon_x[sampled_indexes, :])
            vaedf = out
            vaedf.index = new_samples.index

            to_fill = [
                i for i in hybrid_columns if i not in discarded_columns + list(self.preprocessor.rules.keys())
            ]
            for i in to_fill:
                new_samples[i] = vaedf[i]

        for w in [
            i for i in self.preprocessor.rules.keys() if "embed_category" in self.preprocessor.rules[i][0]
        ]:
            column = create_series(
                self.preprocessor.emb_rules[w][0],
                self.preprocessor.emb_rules[w][1],
                self.preprocessor.emb_rules[w][2],
                new_samples,
            )
            new_samples[w] = column

        for w in [i for i in self.preprocessor.rules.keys() if "sum" in self.preprocessor.rules[i][0]]:
            column = pd.Series(np.zeros(new_samples.shape[0]))
            for w1 in self.preprocessor.rules[w][1]:
                column += w1[0] * new_samples[w1[1]]
            new_samples[w] = column

        if self.dataset.get_y() is not None:
            new_samples[self.dataset.target_column] = self.dataset.get_y().iloc[sampled_indexes].values

        for i in discarded_columns:
            new_samples[i] = X[i].iloc[sampled_indexes].values

        for i in self.preprocessor.discarded[0]:
            if X[i].iloc[sampled_indexes].nunique() > 1:
                new_samples[i] = "*"

        dtypes = self.dataset.data.dtypes.to_dict()
        for i in dtypes:
            if dtypes[i] != "bool":
                new_samples[i] = new_samples[i].astype(dtypes[i])

        cat_cols = new_samples.columns[new_samples.dtypes == "object"]
        for i in cat_cols:
            new_samples[i] = new_samples[i].replace("nan", np.nan)

        if self.engine.privacy_budget == 0.0:
            num_cols = new_samples.columns[new_samples.dtypes != "object"]
            for i in num_cols:
                a = new_samples[i].dropna().unique()
                a.sort()
                dt = np.absolute(a[1:] - a[:-1]).min() if a.shape[0] > 1 else 0.0
                new_samples[i] += dt * np.random.choice([-2, -1, 0, 1, 2], new_samples.shape[0])

        if not has_header:
            new_samples = new_samples.rename(columns=new_samples.iloc[0]).drop(new_samples.index[0])
            return new_samples

        return new_samples.reindex(columns=self.dataset.columns())
