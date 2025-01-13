"""
This module contains the implementation of a Variational Autoencoder (VAE) for tabular data using Flax and JAX.
It includes custom Encoder and Decoder classes, as well as the main TabularVAE class that ties everything together. - .vae.VAEInterface
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
    Encoder network for the VAE.

    Attributes:
        features (Sequence[int]): Sequence of integers representing the number of neurons in each layer.
    """
    features: Sequence[int]

    @nn.compact
    def __call__(self, x: np.ndarray, y: np.ndarray = None) -> Tuple:
        """
        Forward pass through the Encoder.

        Args:
            x (np.ndarray): Input data.
            y (np.ndarray, optional): Additional input data to be concatenated with x.

        Returns:
            Tuple: A tuple containing the mean and log variance of the latent space distribution.
        """
        if y is not None:
            x = jnp.hstack([x, y])

        for i, feat in enumerate(self.features[:-1]):
            x = nn.sigmoid(nn.Dense(feat, name=f"layers_{i}")(x))
        mean_x = nn.Dense(self.features[-1], name="layers_mean")(x)
        log_var_x = nn.Dense(self.features[-1], name="layers_logvar")(x)

        return mean_x, log_var_x


class Decoder(nn.Module):
    """
    Decoder network for the VAE.

    Attributes:
        features (Sequence[int]): Sequence of integers representing the number of neurons in each layer.
        numerical_feature_sizes (Sequence[int]): Sizes of ordinal features.
        categorical_feature_sizes (Sequence[int]): Sizes of categorical features.
    """
    features: Sequence[int]
    numerical_feature_sizes: Sequence[int]
    categorical_feature_sizes: Sequence[int]

    @nn.compact
    def __call__(self, z: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Forward pass through the Decoder.

        Args:
            z (np.ndarray): Latent space representation.
            y (np.ndarray, optional): Additional input data to be concatenated with z.

        Returns:
            np.ndarray: Reconstructed output data.
        """
        if y is not None:
            z = jnp.hstack([z, y])

        for i, feat in enumerate(self.features[1:-1]):
            z = nn.sigmoid(nn.Dense(feat, name=f"layers_{i}")(z))
        z = nn.Dense(self.features[-1], name=f"layers_{len(self.features)-1}")(z)

        features_splitting_points = np.cumsum(
            self.numerical_feature_sizes + self.categorical_feature_sizes
        )
        splitted_z = np.split(z, features_splitting_points, axis=(z.ndim - 1))[:-1]

        activations = []
        if len(self.numerical_feature_sizes) > 0:
            activations.append(splitted_z[0])

        for categorical_tensor in splitted_z[len(self.numerical_feature_sizes):]:
            activations.append(nn.softmax(categorical_tensor, axis=(z.ndim - 1)))

        return jnp.hstack(activations)


class TabularVAE(VAEInterface, nn.Module):
    """
    Variational Autoencoder for tabular data.

    Attributes:
        encoder_widths (Sequence[int]): Sequence of integers for encoder layer sizes.
        decoder_widths (Sequence[int]): Sequence of integers for decoder layer sizes.
        x_shape (Sequence[int]): Shape of the input features.
        y_shape (Sequence[int]): Shape of the additional input features (if any).
        numerical_feature_sizes (Sequence[int]): Sizes of ordinal features.
        categorical_feature_sizes (Sequence[int]): Sizes of categorical features.
        search_params (Dict): Dictionary of parameters used for searching the best model.
    """
    encoder_widths: Sequence[int]
    decoder_widths: Sequence[int]
    x_shape: Sequence[int]
    y_shape: Sequence[int]
    numerical_feature_sizes: Sequence[int]
    categorical_feature_sizes: Sequence[int]
    search_params: Dict

    def setup(self):
        """Initializes the encoder and decoder networks."""
        input_dim = np.prod(self.x_shape) + np.prod(self.y_shape)
        self.encoder = Encoder(self.encoder_widths)
        self.decoder = Decoder(
            self.decoder_widths + (input_dim,),
            self.numerical_feature_sizes,
            self.categorical_feature_sizes,
        )

    def __call__(self, x: np.ndarray, y: np.ndarray = None) -> Tuple:
        """
        Forward pass through the VAE.

        Args:
            x (np.ndarray): Input features.
            y (np.ndarray, optional): Additional input features.

        Returns:
            Tuple: Reconstructed input, mean, and log variance of the latent space.
        """
        mean, logvar = self.encoder(x, y)
        latent = latent_space_sampling(mean, logvar)
        recon_x = self.decoder(latent, y)
        return recon_x, mean, logvar

    def encode(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Encodes the input features into the latent space.

        Args:
            x (np.ndarray): Input features.
            y (np.ndarray, optional): Additional input features.

        Returns:
            np.ndarray: Encoded mean and log variance.
        """
        assert x.shape[1:] == self.x_shape
        x = jnp.reshape(x, (x.shape[0], -1))
        if y is not None:
            y = jnp.reshape(y, (y.shape[0], -1))
        return self.encoder(x, y)

    def decode(self, z: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Decodes the latent space representation into the original feature space.

        Args:
            z (np.ndarray): Latent space representation.
            y (np.ndarray, optional): Additional input features.

        Returns:
            np.ndarray: Reconstructed features.
        """
        x = self.decoder(z, y)
        return x


def latent_space_sampling(mean, log_variance):
    """
    Samples from the latent space using the reparameterization trick.

    Args:
        mean (np.ndarray): Mean of the latent space distribution.
        log_variance (np.ndarray): Log variance of the latent space distribution.

    Returns:
        np.ndarray: Sampled latent space representation.
    """
    standard_deviation = jnp.exp(log_variance / 2.0)
    epsilon = np.random.randn(*standard_deviation.shape)
    return mean + epsilon * standard_deviation


def compute_kernel(X, Y):
    """
    Computes the kernel matrix for Maximum Mean Discrepancy (MMD).

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
    """
    Computes the Maximum Mean Discrepancy (MMD) between two distributions.

    Args:
        X (np.ndarray): First input matrix.
        Y (np.ndarray): Second input matrix.

    Returns:
        float: MMD value.
    """
    XX = compute_kernel(X, X)
    YY = compute_kernel(Y, Y)
    XY = compute_kernel(X, Y)
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd


@jax.vmap
def cross_entropy_loss(logs, targets):
    """
    Computes the cross-entropy loss between predicted logits and target labels.

    Args:
        logs (np.ndarray): Predicted logits.
        targets (np.ndarray): Target labels.

    Returns:
        float: Cross-entropy loss value.
    """
    nll = jnp.take_along_axis(logs, jnp.expand_dims(targets, axis=0), axis=0)
    ce = -jnp.mean(nll)
    return ce


def compute_metrics(architecture, recon_x, x, mean, logvar, search_params):
    """
    Computes various loss metrics for the VAE, including reconstruction loss, KLD, and MMD.

    Args:
        architecture (Dict): Model architecture information.
        recon_x (np.ndarray): Reconstructed input features.
        x (np.ndarray): Original input features.
        mean (np.ndarray): Mean of the latent space distribution.
        logvar (np.ndarray): Log variance of the latent space distribution.
        search_params (Dict): Dictionary of hyperparameters for model training.

    Returns:
        Dict: Dictionary of computed loss metrics.
    """
    features_splitting_points = np.cumsum(
        architecture["numerical_feature_sizes"]
        + architecture["categorical_feature_sizes"]
    )

    span_numerical = len(architecture["numerical_feature_sizes"])
    splitted_z = jnp.split(recon_x, features_splitting_points, axis=1)[:-1]
    splitted_x = jnp.split(x, features_splitting_points, axis=1)[:-1]

    if len(architecture["numerical_feature_sizes"]) > 0:
        gauss_sigmas = jnp.zeros(
            sum(architecture["numerical_feature_sizes"])
        ) + jnp.array([search_params["gauss_s"]])
        loss_ordinal = jnp.square(splitted_z[0] - splitted_x[0])
        loss_ordinal = jnp.sum(jnp.mean(jnp.divide(loss_ordinal, gauss_sigmas), 0))
    else:
        loss_ordinal = 0

    CE_categorical = 0
    if len(architecture["categorical_feature_sizes"]) > 0:
        for original_categorical, reconstructed_categorical in zip(
            splitted_x[span_numerical:], splitted_z[span_numerical:]
        ):
            targets = jnp.argmax(original_categorical, axis=1)
            CE_categorical += cross_entropy_loss(
                jnp.log(reconstructed_categorical + 1e-6), targets
            )

    normal_samples = np.random.randn(*mean.shape)
    mmd_regularizer = compute_mmd(normal_samples, mean)
    loss_kld = -0.5 * jnp.mean(1 + logvar - mean**2 - jnp.exp(logvar))

    loss = (
        jnp.mean(loss_ordinal + CE_categorical)
        + search_params["alpha"] * loss_kld
        + search_params["beta"] * mmd_regularizer
    )

    reconstruction_loss = loss_ordinal + CE_categorical

    return {
        "loss": loss,
        "mean_reconstruction_loss": jnp.mean(reconstruction_loss),
        "reconstruction_loss": reconstruction_loss,
        "loss_ordinal": loss_ordinal,
        "loss_cat": CE_categorical,
    }


@partial(jax.jit, static_argnums=(0,))
def train_step(hashed_architecture, state, batch, search_params):
    """
    Performs a single training step, computing gradients and updating model parameters.

    Args:
        hashed_architecture (str): JSON string of the model architecture.
        state (flax.training.TrainState): The state of the model, including parameters and optimizer.
        batch (np.ndarray): Batch of training data.
        search_params (Dict): Dictionary of hyperparameters for model training.

    Returns:
        flax.training.TrainState: Updated model state.
    """
    architecture = json.loads(hashed_architecture)

    y_batch = None
    if architecture["y_shape"] != [0]:
        y_batch = jnp.split(
            batch,
            np.cumsum(architecture["x_shape"] + architecture["y_shape"]),
            axis=(batch.ndim - 1),
        )[1]
        x_batch = jnp.split(
            batch,
            np.cumsum(architecture["x_shape"] + architecture["y_shape"]),
            axis=(batch.ndim - 1),
        )[0]
    else:
        x_batch = batch

    def loss_fn(params):
        VAE = TabularVAE(
            encoder_widths=architecture["layers_size"],
            decoder_widths=architecture["layers_size"][::-1],
            x_shape=architecture["x_shape"],
            y_shape=architecture["y_shape"],
            numerical_feature_sizes=architecture["numerical_feature_sizes"],
            categorical_feature_sizes=architecture["categorical_feature_sizes"],
            search_params=search_params,
        )
        recon_x, mean, logvar = VAE.apply({"params": params}, x_batch, y_batch)
        span_numerical = len(architecture["numerical_feature_sizes"])

        features_splitting_points = np.cumsum(
            architecture["numerical_feature_sizes"]
            + architecture["categorical_feature_sizes"]
        )

        splitted_z = jnp.split(recon_x, features_splitting_points, axis=1)[:-1]
        splitted_x = jnp.split(x_batch, features_splitting_points, axis=1)[:-1]

        if len(architecture["numerical_feature_sizes"]) > 0:
            gauss_sigmas = jnp.zeros(
                sum(architecture["numerical_feature_sizes"])
            ) + jnp.array([2.0 * search_params["gauss_s"] * search_params["gauss_s"]])
            loss_ordinal = jnp.square(splitted_z[0] - splitted_x[0])
            loss_ordinal = jnp.sum(jnp.divide(loss_ordinal, gauss_sigmas), 1)
        else:
            loss_ordinal = 0

        CE_categorical = 0
        if len(architecture["categorical_feature_sizes"]) > 0:
            for original_categorical, reconstructed_categorical in zip(
                splitted_x[span_numerical:], splitted_z[span_numerical:]
            ):
                gauss_sigmas = jnp.zeros(
                    reconstructed_categorical.shape[1]
                ) + jnp.array([2.0 * search_params["gauss_s_c"] * search_params["gauss_s_c"]])
                loss_cat = jnp.square(reconstructed_categorical - original_categorical)
                loss_cat = jnp.sum(jnp.mean(jnp.divide(loss_cat, gauss_sigmas), 0))
                loss_cat_noise = jnp.sum(
                    jnp.clip(reconstructed_categorical - search_params["prob_clip"], 0.)
                )
                CE_categorical += loss_cat + loss_cat_noise

        weight_penalty_params = jax.tree_util.tree_leaves(params)
        weight_l2 = sum([jnp.sum(x**2) for x in weight_penalty_params])
        weight_penalty = search_params["l2_reg"] * 0.5 * weight_l2
        loss_kld = -0.5 * jnp.mean(1 + logvar - mean**2 - jnp.exp(logvar))

        loss = (
            jnp.mean(loss_ordinal + CE_categorical)
            + weight_penalty
            + search_params["alpha"] * loss_kld
        )

        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


@partial(jax.jit, static_argnums=(0,))
def eval(hashed_architecture, model, eval_ds, search_params, n_samples=1000):
    """
    Evaluates the model on the given evaluation dataset.

    Args:
        hashed_architecture (str): JSON string of the model architecture.
        model (flax.training.TrainState): Model parameters.
        eval_ds (np.ndarray): Evaluation dataset.
        search_params (Dict): Dictionary of hyperparameters for model evaluation.
        n_samples (int, optional): Number of samples to evaluate. Defaults to 1000.

    Returns:
        Dict: Evaluation metrics.
    """
    architecture = json.loads(hashed_architecture)
    y_batch = None
    points = np.random.choice(np.arange(eval_ds.shape[0]), n_samples)

    if architecture["y_shape"] != [0]:
        y_batch = jnp.split(
            eval_ds,
            np.cumsum(architecture["x_shape"] + architecture["y_shape"]),
            axis=(eval_ds.ndim - 1),
        )[1]
        eval_ds = jnp.split(
            eval_ds,
            np.cumsum(architecture["x_shape"] + architecture["y_shape"]),
            axis=(eval_ds.ndim - 1),
        )[0]
        y_batch = y_batch[points, :]
        eval_ds = eval_ds[points, :]
    else:
        eval_ds = eval_ds[points, :]

    VAE = TabularVAE(
        encoder_widths=architecture["layers_size"],
        decoder_widths=architecture["layers_size"][::-1],
        x_shape=architecture["x_shape"],
        y_shape=architecture["y_shape"],
        numerical_feature_sizes=architecture["numerical_feature_sizes"],
        categorical_feature_sizes=architecture["categorical_feature_sizes"],
        search_params=search_params,
    )

    recon_xs, mean, logvar = VAE.apply({"params": model}, eval_ds, y_batch)
    return compute_metrics(architecture, recon_xs, eval_ds, mean, logvar, search_params)
