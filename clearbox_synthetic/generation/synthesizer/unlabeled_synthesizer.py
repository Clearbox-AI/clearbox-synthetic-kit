"""
This module defines the UnlabeledSynthesizer class, which generates synthetic unlabeled data
based on an encoded representation of a dataset.
"""

from .synthesizer import Synthesizer
import os
from typing import List
import numpy as np
import pandas as pd
from ...utils.preprocessor.preprocessor import create_series


class UnlabeledSynthesizer(Synthesizer):
    """
    UnlabeledSynthesizer is a class that generates synthetic unlabeled data
    using a pre-trained engine and a specified dataset.

    Attributes:
        sampled_indexes (np.ndarray): Indices of sampled instances from the dataset.
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
        Generate a single instance of synthetic data by reshuffling columns based on similarity.

        Args:
            new_samples (pd.DataFrame): DataFrame to store generated samples.
            encoded (np.ndarray): Encoded representation of the dataset.
            X (np.ndarray): Original dataset.
            reshuffle_indexes (np.ndarray): Indexes used for reshuffling columns.
            index (int): Index of the sample to generate.
            sampled_index (int): Index of the sampled instance from the dataset.
            n_sampling_points (int): Number of sampling points for similarity calculation.
            hybrid_columns (List): Columns treated as hybrid features.
        """
        encoded_instance = encoded[sampled_index, :]
        distances = np.sqrt(((encoded - encoded_instance) ** 2).sum(axis=1))

        idx1 = np.argpartition(np.array(distances), n_sampling_points)[:n_sampling_points]
        discarded_columns = [i for i in self.preprocessor.discarded[0]]

        columns_to_shuffle = [
            col for col in self.dataset.x_columns() if col not in discarded_columns + hybrid_columns
        ]

        for i, j in enumerate(columns_to_shuffle):
            new_samples[j].at[index] = X[j].at[idx1[reshuffle_indexes[index, i]]]

    def generate(
        self,
        has_header=None,
        points=None,
        n_sampling_points=5,
        hybrid_columns=[],
        latent_noise=0.0,
    ):
        """
        Generate synthetic data based on the encoded representation of the dataset.

        Args:
            has_header (bool, optional): Whether the generated data should include headers. Defaults to None.
            points (list, optional): List of points to sample from. Defaults to None.
            n_sampling_points (int, optional): Number of sampling points for similarity calculation. Defaults to 5.
            hybrid_columns (list, optional): List of columns to treat as hybrid features. Defaults to an empty list.
            latent_noise (float, optional): Amount of noise to add to the latent space. Defaults to 0.0.

        Returns:
            pd.DataFrame: Generated synthetic data.
        """
        X = self.dataset.get_x()

        if not hybrid_columns:
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

        _, encoded, _ = self.engine.apply(preprocessed_data)
        encoded = encoded + latent_noise * np.random.randn(*encoded.shape)

        new_samples = pd.DataFrame(index=range(n_samples), columns=list(data.columns))
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))

        discarded_columns = [i for i in self.preprocessor.discarded[0]]
        columns_to_shuffle = [
            col for col in self.dataset.data.columns.tolist()
            if col != self.dataset.target_column and col not in discarded_columns
        ]

        reshuffle_indexes = np.zeros((n_samples, len(columns_to_shuffle)))
        for i in range(len(columns_to_shuffle)):
            reshuffle_indexes[:, i] = np.random.choice(
                np.arange(0, n_sampling_points), reshuffle_indexes.shape[0]
            )
        reshuffle_indexes = reshuffle_indexes.astype(int)

        columns_to_shuffle = [
            col for col in self.dataset.data.columns.tolist()
            if col != self.dataset.target_column and col not in discarded_columns + hybrid_columns
        ]
        if columns_to_shuffle:
            for i in range(n_samples):
                self._generate_instance(
                    new_samples, encoded, X, reshuffle_indexes, i, sampled_indexes[i],
                    n_sampling_points, hybrid_columns
                )

        if hybrid_columns:
            recon_x = self.engine.decode(encoded)
            recon_x = np.asarray(recon_x)
            out = self._sample_vae(data.iloc[sampled_indexes, :], recon_x[sampled_indexes, :])
            vaedf = out
            vaedf.index = new_samples.index
            to_fill = [
                i for i in hybrid_columns
                if i not in discarded_columns + list(self.preprocessor.rules.keys())
            ]
            for i in to_fill:
                new_samples[i] = vaedf[i]

        for w in [
            i for i in self.preprocessor.rules.keys()
            if 'embed_category' in self.preprocessor.rules[i][0]
        ]:
            column = create_series(
                self.preprocessor.emb_rules[w][0],
                self.preprocessor.emb_rules[w][1],
                self.preprocessor.emb_rules[w][2],
                new_samples
            )
            new_samples[w] = column

        for w in [
            i for i in self.preprocessor.rules.keys()
            if 'sum' in self.preprocessor.rules[i][0]
        ]:
            column = pd.Series(np.zeros(new_samples.shape[0]))
            for w1 in self.preprocessor.rules[w][1]:
                column += w1[0] * new_samples[w1[1]]
            new_samples[w] = column

        for i in discarded_columns:
            new_samples[i] = X[i].iloc[sampled_indexes].values

        for i in self.preprocessor.discarded[0]:
            if X[i].iloc[sampled_indexes].nunique() > 1:
                new_samples[i] = "*"

        dtypes = self.dataset.data.dtypes.to_dict()
        for i in dtypes:
            if dtypes[i] != 'bool':
                new_samples[i] = new_samples[i].astype(dtypes[i])

        cat_cols = new_samples.columns[new_samples.dtypes == 'object']
        for i in cat_cols:
            new_samples[i] = new_samples[i].replace('nan', np.nan)

        if self.engine.privacy_budget == 0.0:
            num_cols = new_samples.columns[new_samples.dtypes != 'object']
            for i in num_cols:
                a = new_samples[i].dropna().unique()
                a.sort()
                dt = np.absolute(a[1:] - a[:-1]).min() if a.shape[0] > 1 else 0.0
                new_samples[i] = new_samples[i] + dt * np.random.choice(
                    [-2, -1, 0, 1, 2], new_samples.shape[0]
                )

        if not has_header:
            new_samples = new_samples.rename(columns=new_samples.iloc[0]).drop(new_samples.index[0])

        return new_samples
