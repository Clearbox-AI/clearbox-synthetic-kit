"""
This module implements a denoising diffusion model using JAX, Equinox, and Flax libraries.
It includes an MLP-based denoising model and a TabularDiffusion class for training and sampling
from the diffusion model.

This implementation is based on the score-based diffusion example from Equinox:
https://docs.kidger.site/equinox/examples/score_based_diffusion/

Reference:
@inproceedings{song2021scorebased,
    title={Score-Based Generative Modeling through Stochastic Differential Equations},
    author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and
            Abhishek Kumar and Stefano Ermon and Ben Poole},
    booktitle={International Conference on Learning Representations},
    year={2021},
}
"""

import json
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from flax import linen as nn
from typing import Sequence, Tuple, Dict

from .diffusion import DiffusionInterface
import functools as ft

import diffrax as dfx
import equinox as eqx
import jax.random as jr
import optax
from loguru import logger


class MLPDenoising(eqx.Module):
    """A Multilayer Perceptron (MLP) model for denoising in the diffusion framework.
    
    Based on the implementation from Equinox's score-based diffusion example:
    https://docs.kidger.site/equinox/examples/score_based_diffusion/

    Attributes:
        mlp (eqx.nn.MLP): The MLP model for denoising.
        t1 (float): A normalization factor for the time variable.
        input_size (int): The size of the input data.
    """

    mlp: eqx.nn.MLP
    t1: float
    input_size: int

    def __init__(self, input_size: int, hidden_size: int, depth: int, t1: float, *, key):
        """Initializes the MLPDenoising model.

        Args:
            input_size (int): The size of the input data.
            hidden_size (int): The size of the hidden layers.
            depth (int): The depth (number of layers) of the MLP.
            t1 (float): A normalization factor for the time variable.
            key: A JAX random key for initializing the MLP.
        """
        self.mlp = eqx.nn.MLP(input_size + 1, input_size, hidden_size, depth, key=key)
        self.t1 = t1
        self.input_size = input_size

    def __call__(self, t: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Performs a forward pass of the MLPDenoising model.

        Args:
            t (float): The time variable.
            y (jnp.ndarray): The input data.

        Returns:
            jnp.ndarray: The denoised output.
        """
        t = jnp.asarray(t, dtype=jnp.float32)
        t = t / self.t1

        if t.ndim == 0:
            t = jnp.expand_dims(t, axis=0)
        elif t.ndim > 1:
            t = t.reshape(-1)

        if y.ndim == 1:
            y = y.reshape(1, -1)
        elif y.ndim > 2:
            raise ValueError(f"Unexpected y.ndim: {y.ndim}")

        t = t.reshape(-1, 1)
        inputs = jnp.concatenate([y, t], axis=-1)
        outputs = jax.vmap(self.mlp)(inputs)
        return outputs[0] if outputs.shape[0] == 1 else outputs


class TabularDiffusion(DiffusionInterface):
    """A class for training and sampling from a diffusion model on tabular data.
    
    Implementation inspired by Equinox's score-based diffusion example:
    https://docs.kidger.site/equinox/examples/score_based_diffusion/

    Attributes:
        key: A JAX random key used for various operations.
        hidden_size (int): The size of the hidden layers in the MLP.
        depth (int): The depth (number of layers) of the MLP.
        t1 (float): The end time for the diffusion process.
        dt0 (float): The time step size for the diffusion solver.
        model (MLPDenoising, optional): The MLP-based denoising model.
        int_beta (callable, optional): The integral of the beta function over time.
        input_size (int, optional): The size of the input data.
    """

    def __init__(self, seed: int = 42, hidden_size: int = 100, depth: int = 2, t1: float = 10.0, dt0: float = 0.1):
        """Initializes the TabularDiffusion class.

        Args:
            seed (int): The random seed for reproducibility.
            hidden_size (int): The size of the hidden layers in the MLP.
            depth (int): The depth (number of layers) of the MLP.
            t1 (float): The end time for the diffusion process.
            dt0 (float): The time step size for the diffusion solver.
        """
        self.key = jr.PRNGKey(seed)
        self.model_key, self.train_key, self.loader_key, self.sample_key, self.data_key = jr.split(self.key, 5)
        self.hidden_size = hidden_size
        self.depth = depth
        self.t1 = t1
        self.dt0 = dt0
        self.model = None
        self.int_beta = None
        self.input_size = None

    def batch_loss_fn_diff(self, model, weight, int_beta, data, t1, key):
        """Computes the diffusion model's batch loss function.

        Args:
            model: The MLP model for denoising.
            weight (callable): A weighting function for the loss.
            int_beta (callable): The integral of the beta function.
            data (jnp.ndarray): The input data.
            t1 (float): The end time for the diffusion process.
            key: A JAX random key for generating noise.

        Returns:
            float: The computed batch loss.
        """
        batch_size = data.shape[0]
        tkey, losskey = jr.split(key)

        t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1)
        int_beta_t = int_beta(t)
        exp_neg_half_int_beta_t = jnp.exp(-0.5 * int_beta_t)
        mean = data * exp_neg_half_int_beta_t[:, None]
        var = jnp.maximum(1 - jnp.exp(-int_beta_t), 1e-5)
        std = jnp.sqrt(var)[:, None]
        noise = jr.normal(losskey, data.shape)
        y = mean + std * noise

        pred = model(t, y)
        weight_t = weight(t)
        loss = weight_t * jnp.mean((pred + noise / std) ** 2, axis=1)
        return jnp.mean(loss)

    @eqx.filter_jit
    def make_step(self, model, weight, int_beta, data, t1, key, opt_state, opt_update):
        """Performs a single training step of the diffusion model.

        Args:
            model: The MLP model for denoising.
            weight (callable): A weighting function for the loss.
            int_beta (callable): The integral of the beta function.
            data (jnp.ndarray): The input data.
            t1 (float): The end time for the diffusion process.
            key: A JAX random key for generating noise.
            opt_state: The optimizer state.
            opt_update: The optimizer update function.

        Returns:
            Tuple[float, Any, Any, Any]: The loss, updated model, new key, and updated optimizer state.
        """
        loss_fn_diff = eqx.filter_value_and_grad(self.batch_loss_fn_diff)
        loss, grads = loss_fn_diff(model, weight, int_beta, data, t1, key)
        updates, opt_state = opt_update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        key = jr.split(key, 1)[0]
        return loss, model, key, opt_state

    @eqx.filter_jit
    def single_sample_fn(self, model, int_beta, data_shape, dt0, t1, key):
        """Generates a single sample from the diffusion model.

        Args:
            model: The MLP model for denoising.
            int_beta (callable): The integral of the beta function.
            data_shape (Tuple[int]): The shape of the data to sample.
            dt0 (float): The time step size for the diffusion solver.
            t1 (float): The end time for the diffusion process.
            key: A JAX random key for generating noise.

        Returns:
            jnp.ndarray: The generated sample.
        """
        def drift(t, y, args):
            if jnp.isscalar(t) or t.ndim == 0:
                _, beta = jax.jvp(int_beta, (jnp.array(t, dtype=float),), (jnp.ones_like(jnp.array(t, dtype=float)),))
            else:
                beta = jnp.ones_like(t)
            return -0.5 * beta * (y + model(t, y))

        term = dfx.ODETerm(drift)
        solver = dfx.Tsit5()
        t0 = 0.0
        y1 = jr.normal(key, data_shape)
        sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, y1)
        return sol.ys[0]

    def dataloader(self, data: jnp.ndarray, batch_size: int, *, key):
        """Yields batches of data for training.

        Args:
            data (jnp.ndarray): The input data.
            batch_size (int): The size of each batch.
            key: A JAX random key for shuffling data.

        Yields:
            jnp.ndarray: A batch of data.
        """
        dataset_size = data.shape[0]
        indices = jnp.arange(dataset_size)
        while True:
            key, subkey = jr.split(key, 2)
            perm = jr.permutation(subkey, indices)
            start = 0
            end = batch_size
            while end <= dataset_size:
                yield data[perm[start:end]]
                start = end
                end = start + batch_size
            if start < dataset_size:
                yield data[perm[start:dataset_size]]

    def fit(self, data: jnp.ndarray, num_steps: int = 30000, lr: float = 3e-4, batch_size: int = 256, print_freq: int = 1000):
        """Trains the diffusion model on the provided data.

        Args:
            data (jnp.ndarray): The input data for training.
            num_steps (int): The number of training steps.
            lr (float): The learning rate.
            batch_size (int): The size of each training batch.
            print_freq (int): The frequency of logging the training loss.
        """
        self.input_size = data.shape[1]
        self.model = MLPDenoising(data.shape[1], self.hidden_size, self.depth, self.t1, key=self.model_key)
        
        # Define int_beta to properly handle both scalar and array inputs
        def safe_int_beta(t):
            t_array = jnp.asarray(t, dtype=jnp.float32)
            # Handle array inputs by applying the operation elementwise
            if t_array.ndim > 0:
                return jnp.ones_like(t_array)  # Return ones with same shape instead of converting to int
            else:
                # Only convert to int if it's a scalar (ndim=0)
                return jnp.asarray(1.0, dtype=jnp.float32)
        
        self.int_beta = safe_int_beta
        weight = lambda t: 1 - jnp.exp(-safe_int_beta(t))

        opt = optax.adabelief(lr)
        opt_state = opt.init(eqx.filter(self.model, eqx.is_inexact_array))

        total_value = 0
        total_size = 0
        data_loader = self.dataloader(data, batch_size, key=self.loader_key)
        for step in range(num_steps):
            data_batch = next(data_loader)
            value, self.model, self.train_key, opt_state = self.make_step(
                self.model, weight, self.int_beta, data_batch, self.t1, self.train_key, opt_state, opt.update
            )
            total_value += value.item()
            total_size += 1
            if (step % print_freq) == 0 or step == num_steps - 1:
                logger.info(f"Diffusion Training Step={step} Loss={total_value / total_size}")
                total_value = 0
                total_size = 0

    def sample(self, n_samples: int) -> jnp.ndarray:
        """Generates samples from the trained diffusion model.

        Args:
            n_samples (int): The number of samples to generate.

        Returns:
            jnp.ndarray: The generated samples.
        """
        logger.info("Generating samples")
        sample_keys = jr.split(self.sample_key, n_samples)
        sample_fn = ft.partial(self.single_sample_fn, self.model, self.int_beta, (self.input_size,), self.dt0, self.t1)
        samples = jax.vmap(sample_fn)(sample_keys)
        return samples
