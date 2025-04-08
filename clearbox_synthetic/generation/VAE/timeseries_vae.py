"""
This module implements a Variational Autoencoder (VAE) specifically designed for time series data using Flax and JAX.
It includes the custom Encoder and Decoder classes for sequence-based feature extraction and reconstruction.
"""

import json
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from flax import linen as nn
from typing import Sequence, Tuple, Dict
from .vae import VAEInterface

class Encoder(nn.Module):
    """
    Encoder network for the Time Series VAE.

    Attributes:
        features (Sequence[int]): Sequence of integers specifying the size of each dense layer.
        feature_sizes (int): Size of each feature in the input data.
        max_sequence_length (int): Maximum length of the input sequence.
        num_heads (int): Number of attention heads.
    """
    features: Sequence[int]
    feature_sizes: int
    max_sequence_length: int
    num_heads: int

    @nn.compact
    def __call__(self, x: np.ndarray, y: np.ndarray = None) -> Tuple:
        """
        Forward pass through the Encoder.

        Args:
            x (np.ndarray): Input features.
            y (np.ndarray, optional): Additional features.

        Returns:
            Tuple: Mean and log variance of the latent space distribution.
        """
        x = x.reshape(-1, self.max_sequence_length, self.feature_sizes) 
        x = x + (np.arange(self.max_sequence_length) / self.max_sequence_length).reshape(1, self.max_sequence_length, 1)
        x = nn.LayerNorm()(x)        
        x = x + nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=int(x.shape[1]/self.num_heads))(x, x)

        x = x.reshape(-1, x.shape[-1] * x.shape[-2])
        x_res = x
        for j,i in enumerate(range(len(self.features))):
            x = nn.Dense(x.shape[1], name=f"einter{j}", kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)(x)
            x = nn.gelu(x)
            
        x = x + x_res
        mean_x = nn.Dense(self.features[-1], name="layers_mean", kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)(x)
        log_var_x = nn.Dense(self.features[-1], name="layers_logvar", kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)(x)
        return mean_x, log_var_x


class Decoder(nn.Module):
    """
    Decoder network for the Time Series VAE.

    Attributes:
        features (Sequence[int]): Sequence of integers specifying the size of each dense layer.
        feature_sizes (int): Size of each feature in the output data.
        max_sequence_length (int): Maximum length of the output sequence.
    """
    features: Sequence[int]
    feature_sizes: int
    max_sequence_length: int
    num_heads: int

    @nn.compact
    def __call__(self, z: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Forward pass through the Decoder.

        Args:
            z (np.ndarray): Latent representation.
            y (np.ndarray, optional): Additional features.

        Returns:
            np.ndarray: Reconstructed features.
        """
        if y is not None:
            y = y.reshape(z.shape[0], y.shape[-1])
            y = nn.Dense(z.shape[1], name="embed_y_dec")(y)
            z = z + y
        z = nn.LayerNorm()(z)     
        z_res = z
        for j,i in enumerate(self.features):
            z = nn.Dense(z.shape[1], 
                         name=f"layers_{j}", 
                         kernel_init=nn.initializers.xavier_uniform(), 
                         bias_init=nn.initializers.zeros)(z)
            z = nn.gelu(z)
        z = z + z_res
        z = nn.Dense(int(self.feature_sizes * self.max_sequence_length), name=f"layers_intermediate")(z)
        z = nn.gelu(z)
        zis = []
        for j in range(self.feature_sizes):
            zis.append(nn.Dense(int(self.max_sequence_length), name=f"layers_e{j}")(z))

        z = jnp.concatenate(zis, axis=1)
        return z


class TimeSeriesVAE(VAEInterface, nn.Module):
    """
    Time Series Variational Autoencoder.

    Attributes:
        encoder_widths (Sequence[int]): Sequence specifying the encoder layer sizes.
        decoder_widths (Sequence[int]): Sequence specifying the decoder layer sizes.
        y_shape (Sequence[int]): Shape of the additional input features.
        feature_sizes (int): Size of each feature.
        max_sequence_length (int): Maximum length of the input sequence.
        num_heads (int): Number of attention heads.
        search_params (Dict): Dictionary of parameters for the model.
    """
    encoder_widths: Sequence[int]
    decoder_widths: Sequence[int]
    y_shape: Sequence[int]
    feature_sizes: int
    max_sequence_length: int
    num_heads: int
    search_params: Dict

    def setup(self):
        """Sets up the encoder and decoder components."""
        input_dim = np.prod(self.feature_sizes * self.max_sequence_length) + np.prod(self.y_shape)
        self.encoder = Encoder(self.encoder_widths, self.feature_sizes, self.max_sequence_length, self.num_heads)
        self.decoder = Decoder(self.decoder_widths + (input_dim,), self.feature_sizes, self.max_sequence_length, self.num_heads)

    def __call__(self, x: np.ndarray, y: np.ndarray = None) -> Tuple:
        """
        Forward pass through the VAE.

        Args:
            x (np.ndarray): Input features.
            y (np.ndarray, optional): Additional features.

        Returns:
            Tuple: Reconstructed input, mean, and log variance of the latent space.
        """
        mean, logvar = self.encoder(x, y)
        latent = latent_space_sampling(mean, logvar)
        recon_x = self.decoder(latent, y)
        return recon_x, mean, logvar

    def encode(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Encodes input features into the latent space."""
        assert x.shape[1:] == self.feature_sizes * self.max_sequence_length
        x = jnp.reshape(x, (x.shape[0], -1))
        if y is not None:
            y = jnp.reshape(y, (y.shape[0], -1))
        return self.encoder(x, y)

    def decode(self, z: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Decodes the latent representation into the original feature space."""
        return self.decoder(z, y)


def latent_space_sampling(mean, log_variance):
    """
    Samples from the latent space using the reparameterization trick.

    Args:
        mean (np.ndarray): Mean of the latent distribution.
        log_variance (np.ndarray): Log variance of the latent distribution.

    Returns:
        np.ndarray: Sampled latent vector.
    """
    standard_deviation = jnp.exp(log_variance / 2.0)
    epsilon = np.random.randn(*standard_deviation.shape)
    return mean + epsilon * standard_deviation


def compute_kernel(X, Y):
    """
    Computes the kernel matrix for MMD.

    Args:
        X (np.ndarray): First input matrix.
        Y (np.ndarray): Second input matrix.

    Returns:
        np.ndarray: Kernel matrix.
    """
    X_size = X.shape[0]
    Y_size = Y.shape[0]
    dim = X.shape[1]
    X = jnp.expand_dims(X, axis=1)
    Y = jnp.expand_dims(Y, axis=0)
    tiled_X = jnp.broadcast_to(X, (X_size, Y_size, dim))
    tiled_Y = jnp.broadcast_to(Y, (X_size, Y_size, dim))
    kernel_input = jnp.power(tiled_X - tiled_Y, 2).mean(2) / float(dim)
    return jnp.exp(-kernel_input)


def compute_mmd(X, Y):
    """Computes the Maximum Mean Discrepancy (MMD) between two distributions."""
    XX = compute_kernel(X, X)
    YY = compute_kernel(Y, Y)
    XY = compute_kernel(X, Y)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def compute_metrics(architecture, recon_x, x, mean, logvar, search_params):
    """
    Computes the loss metrics for the VAE.

    Args:
        architecture (Dict): Model architecture details.
        recon_x (np.ndarray): Reconstructed input features.
        x (np.ndarray): Original input features.
        mean (np.ndarray): Mean of the latent distribution.
        logvar (np.ndarray): Log variance of the latent distribution.
        search_params (Dict): Hyperparameters for loss computation.

    Returns:
        Dict: Dictionary containing various loss metrics.
    """
    gauss_sigmas = jnp.zeros(sum([architecture["feature_sizes"] * architecture["max_sequence_length"]])) + jnp.array([search_params["gauss_s"]])
    loss_ordinal = jnp.square(recon_x - x)
    loss_ordinal = jnp.sum(jnp.mean(jnp.divide(loss_ordinal, gauss_sigmas), 0))
    normal_samples = np.random.randn(*mean.shape)
    mmd_regularizer = compute_mmd(normal_samples, mean)
    loss_kld = -0.5 * jnp.mean(1 + logvar - mean**2 - jnp.exp(logvar))
    loss = jnp.mean(loss_ordinal) + search_params["alpha"] * loss_kld + search_params["beta"] * mmd_regularizer
    return {
        "loss": loss,
        "mean_reconstruction_loss": jnp.mean(loss_ordinal),
        "reconstruction_loss": loss_ordinal,
        "loss_ordinal": loss_ordinal,
    }


@partial(jax.jit, static_argnums=(0,))
def train_step(hashed_architecture, state, batch, search_params):
    """
    Performs a single training step.

    Args:
        hashed_architecture (str): JSON string of the model architecture.
        state (TrainState): Model state containing parameters and optimizer.
        batch (np.ndarray): Training batch.
        search_params (Dict): Hyperparameters for the training step.

    Returns:
        TrainState: Updated model state.
    """
    architecture = json.loads(hashed_architecture)
    y_batch = None
    if architecture["y_shape"] != [0]:
        y_batch = jnp.split(batch, np.cumsum([architecture["feature_sizes"] * architecture["max_sequence_length"]] + architecture["y_shape"]), axis=(batch.ndim - 1))[1]
        x_batch = jnp.split(batch, np.cumsum([architecture["feature_sizes"] * architecture["max_sequence_length"]] + architecture["y_shape"]), axis=(batch.ndim - 1))[0]
    else:
        x_batch = batch

    def loss_fn(params):
        VAE = TimeSeriesVAE(
            encoder_widths=architecture["layers_size"],
            decoder_widths=architecture["layers_size"][::-1],
            y_shape=architecture["y_shape"],
            feature_sizes=architecture["feature_sizes"],
            max_sequence_length=architecture["max_sequence_length"],
            num_heads=architecture["num_heads"],
            search_params=search_params,
        )
        recon_x, mean, logvar = VAE.apply({"params": params}, x_batch, y_batch)
        gauss_sigmas = jnp.zeros(sum([architecture["feature_sizes"] * architecture["max_sequence_length"]])) + jnp.array([2.0 * search_params["gauss_s"] * search_params["gauss_s"]])
        loss_ordinal = jnp.square(recon_x - x_batch)
        loss_ordinal = jnp.sum(jnp.divide(loss_ordinal, gauss_sigmas), 1)
        weight_penalty_params = jax.tree_util.tree_leaves(params)
        weight_l2 = sum([jnp.sum(x**2) for x in weight_penalty_params])
        weight_penalty = search_params["l2_reg"] * 0.5 * weight_l2
        loss_kld = -0.5 * jnp.mean(1 + logvar - mean**2 - jnp.exp(logvar))
        return jnp.mean(loss_ordinal) + weight_penalty + search_params["alpha"] * loss_kld
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


@partial(jax.jit, static_argnums=(0,))
def eval(hashed_architecture, model, eval_ds, search_params, n_samples=1000):
    """
    Evaluates the model on a given dataset.

    Args:
        hashed_architecture (str): JSON string of the model architecture.
        model (TrainState): Model state containing parameters.
        eval_ds (np.ndarray): Evaluation dataset.
        search_params (Dict): Hyperparameters for evaluation.
        n_samples (int): Number of samples for evaluation.

    Returns:
        Dict: Evaluation metrics.
    """
    architecture = json.loads(hashed_architecture)
    y_batch = None
    points = np.random.choice(np.arange(eval_ds.shape[0]), n_samples)
    if architecture["y_shape"] != [0]:
        y_batch = jnp.split(eval_ds, np.cumsum([architecture["feature_sizes"] * architecture["max_sequence_length"]] + architecture["y_shape"]), axis=(eval_ds.ndim - 1))[1]
        eval_ds = jnp.split(eval_ds, np.cumsum([architecture["feature_sizes"] * architecture["max_sequence_length"]] + architecture["y_shape"]), axis=(eval_ds.ndim - 1))[0]
        y_batch = y_batch[points, :]
        eval_ds = eval_ds[points, :]
    else:
        eval_ds = eval_ds[points, :]
        
    VAE = TimeSeriesVAE(
        encoder_widths=architecture["layers_size"],
        decoder_widths=architecture["layers_size"][::-1],
        y_shape=architecture["y_shape"],
        feature_sizes=architecture["feature_sizes"],
        max_sequence_length=architecture["max_sequence_length"],
        num_heads=architecture["num_heads"],
        search_params=search_params,
    )
    recon_xs, mean, logvar = VAE.apply({"params": model}, eval_ds, y_batch)
    return compute_metrics(architecture, recon_xs, eval_ds, mean, logvar, search_params)
