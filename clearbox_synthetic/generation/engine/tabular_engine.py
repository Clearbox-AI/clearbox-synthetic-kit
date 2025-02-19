import os
import json
import optax
import numpy as np
import equinox as eqx
from typing import Sequence, Callable, Dict, List, Tuple, Literal
import jax
from jax import random
from flax.core.frozen_dict import FrozenDict
from flax import serialization
from flax.training import train_state
from tqdm import trange
from loguru import logger

from clearbox_preprocessor import Preprocessor
from ...utils import Dataset
from ..VAE.tabular_vae import TabularVAE, train_step, eval
from ..diffusion.tabular_diffusion import TabularDiffusion
from .engine import EngineInterface

class TabularEngine(EngineInterface):
    """
    This class integrates the ``TabularVAE`` and ``TabularDiffusion`` models to enable training, 
    evaluation, and inference for tabular datasets.

    Attributes
    ----------
    model : TabularVAE
        The Variational Autoencoder model.
    diffusion_model : TabularDiffusion
        The Diffusion Model for additional training.
    params : FrozenDict
        The model parameters.
    search_params : Dict
        Training parameters.
    architecture : Dict
        The architecture configuration of the model.
    hashed_architecture : str
        A hashed string representation of the architecture.
    """
    X: Dataset
    model: TabularVAE
    diffusion_model: TabularDiffusion
    params: FrozenDict
    search_params: Dict
    architecture: Dict
    hashed_architecture: str

    def __init__(
        self,
        dataset: Dataset,
        layers_size: Sequence[int] = 50,
        params: FrozenDict = None,
        train_params: Dict = None,
        privacy_budget: float = 1.0,
        model_type: str = 'VAE',
        discarding_threshold: float = 0.9, 
        get_discarded_info: bool = False,
        excluded_col: List = [],
        time: str = None,
        missing_values_threshold: float = 0.999,
        n_bins: int = 0,
        scaling: Literal["none", "normalize", "standardize", "quantile"] = "none", 
        num_fill_null : Literal["interpolate","forward", "backward", "min", "max", "mean", "zero", "one"] = "mean",
        cat_labels_threshold: float = 0.01,
        unseen_labels = 'ignore',
    ):
        """
        Initializes the TabularEngine.

        Parameters
        ----------
        layers_size : Sequence[int]
            The sizes of the hidden layers.
        params : FrozenDict, optional
            Model parameters. Defaults to None.
        train_params : Dict, optional
            Training parameters. Defaults to None.
        privacy_budget : float, optional
            The privacy budget. Defaults to 1.0.
        model_type : str, optional
            Type of model ('VAE' or 'Diffusion'). Defaults to 'VAE'.
        """
        self._enforce_cpu_if_no_gpu()
        
        self.model_type = model_type

        rng = random.PRNGKey(0)
        rng, key = random.split(rng)

        X, Y = dataset.get_x_y()
        if Y:
            y_shape=Y[0].shape  
        else:
            y_shape = [0]

        self.preprocessor = Preprocessor(
            X,
            discarding_threshold,
            get_discarded_info,
            excluded_col,
            time,
            missing_values_threshold,
            n_bins,
            scaling,
            num_fill_null,
            cat_labels_threshold,
            unseen_labels,
        ) 
        X_train = self.preprocessor.transform(X)

        x_shape = X_train[0].shape   

        numerical_feature_sizes, categorical_feature_sizes = self.preprocessor.get_features_sizes()

        if model_type != 'VAE':
            layers_size = [int(x_shape[0])]
        if train_params is None:
            if model_type == 'VAE':
                beta = 0
                alpha = 0.1
            elif model_type =='Diffusion':
                beta = 0
                alpha = 0
            train_params = {
                "l2_reg": 0.000,
                "beta": beta,
                "alpha": alpha,
                "gauss_s": 0.01,
                "gauss_s_c": 0.1,
                "weight_decay": 0.000,
                "prob_clip": 0.99,
                "hidden_size": 100,
                "depth": 2,
                "t1": 10.0,
                "dt0": 0.1, 
            }

        self.privacy_budget = privacy_budget
        self.search_params = train_params
        self.model = TabularVAE(
            encoder_widths=layers_size,
            decoder_widths=layers_size[::-1],
            x_shape=x_shape,
            y_shape=y_shape,
            numerical_feature_sizes=numerical_feature_sizes,
            categorical_feature_sizes=categorical_feature_sizes,
            search_params=train_params,
        )

        if model_type == 'Diffusion':
            self.diffusion_model = TabularDiffusion(seed=42, 
                                                    hidden_size=train_params["hidden_size"], 
                                                    depth=train_params["depth"],
                                                    t1=train_params["t1"], 
                                                    dt0=train_params["dt0"])

        x = random.uniform(key, [np.prod(x_shape)])
        y = random.uniform(key, [np.prod(y_shape)]) if y_shape != [0] else None

        self.params = params if params else self.model.init(rng, x, y)["params"]
        self.architecture = {
            "layers_size": layers_size,
            "x_shape": x_shape,
            "y_shape": y_shape,
            "numerical_feature_sizes": numerical_feature_sizes,
            "categorical_feature_sizes": categorical_feature_sizes,
        }
        self.hashed_architecture = json.dumps(self.architecture)

    def _enforce_cpu_if_no_gpu(self):
        try:
            import jax
            import os 
            os.environ['JAX_PLATFORMS'] = 'cpu'

            # Check if all devices are CPU
            all_cpu = all(device.platform == 'cpu' for device in jax.devices())
            
            if all_cpu:
                os.environ['JAX_PLATFORMS'] = 'cpu'
                print("No GPU detected. JAX is set to CPU mode.")
            else:
                print("🚀 GPU detected. JAX will utilize GPU devices.")
                    
        except Exception as e:
            # In case of any errors with JAX initialization, fall back to CPU
            os.environ['JAX_PLATFORMS'] = 'cpu'
            print(f"⚠️ An error occurred: {e}. \nDefaulting to CPU.")

    def apply(self, x: np.ndarray, y: np.ndarray = None) -> Tuple:
        """Applies the model to the input data.

        Args:
            x (np.ndarray): The input data.
            y (np.ndarray, optional): The target data. Defaults to None.

        Returns:
            Tuple: The model's output.
        """
        return self.model.apply({"params": self.params}, x, y)

    def encode(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Encodes the input data into the latent space.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        y : np.ndarray, optional
            The target data. Defaults to None.

        Returns
        -------
        np.ndarray
            The encoded representation.
        """
        return self.model.apply({"params": self.params}, x, y, method=self.model.encode)

    def decode(self, z: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Decodes the latent representation back into the original space.

        Parameters
        ----------
        z : np.ndarray
            The latent representation.
        y : np.ndarray, optional
            The target data. Defaults to None.

        Returns
        -------
        np.ndarray
            The decoded data.
        """
        return self.model.apply({"params": self.params}, z, y, method=self.model.decode)

    def fit(
        self,
        dataset: Dataset,
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-2,
        val_ds: np.ndarray = None,
        y_val_ds: np.ndarray = None,
        patience: int = 4,
    ):
        """
        Trains the model on the provided dataset.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.
        epochs : int, optional
            The number of training epochs. Defaults to 20.
        batch_size : int, optional
            The batch size for training. Defaults to 128.
        learning_rate : float, optional
            The learning rate for the optimizer. Defaults to 1e-2.
        val_ds : np.ndarray, optional
            The validation dataset. Defaults to None.
        y_val_ds : np.ndarray, optional
            The target values for the validation dataset. Defaults to None.
        patience : int, optional
            The number of epochs to wait for improvement before stopping early. Defaults to 4.
        """
        weight_decay = self.search_params["weight_decay"]
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        )

        X, y_train_ds = dataset.get_x_y()
        train_ds = self.preprocessor.transform(X)
        
        train_loader = np.hstack([train_ds, y_train_ds]) if y_train_ds is not None else np.hstack([train_ds])
        metrics_train = None
        metrics_val = None
        type_training = 'Engine' if self.model_type == 'VAE' else 'Preprocessor'
        splits = np.arange(batch_size, train_loader.shape[0], batch_size)
        loop_range = trange(epochs, desc=f"{type_training} fitting in progress", unit="epoch", leave=True)
        total_loss = 1e5
        no_improvement = 0

        for i in loop_range:
            for batch in np.array_split(train_loader, splits, axis=0):
                state = train_step(self.hashed_architecture, state, batch, self.search_params)

            if i % 25 == 0:
                self.params = state.params
                metrics_train = self.evaluate(train_ds, y_train_ds)
                if val_ds is not None:
                    metrics_val = self.evaluate(val_ds, y_val_ds)
                    loop_range.set_postfix({
                        'Train loss:': metrics_train['mean_reconstruction_loss'],
                        'Val loss:': metrics_val['mean_reconstruction_loss']
                    }, refresh=True)
                    loss_i = metrics_val['loss']
                else:
                    loop_range.set_postfix({
                        'Train loss:': metrics_train['mean_reconstruction_loss']
                    }, refresh=True)
                    loss_i = metrics_train['loss']

                if loss_i < total_loss:
                    total_loss = loss_i
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement >= patience:
                    print(f"No improvement in the last {25 * patience} iterations, stopping early.")
                    break

        self.train_loss = metrics_train
        self.val_loss = metrics_val
        self.params = state.params

        if self.model_type == 'Diffusion':
            _, diff_train_data, _ = self.model.apply({"params": self.params}, train_ds, y_train_ds)
            self.diffusion_model.fit(diff_train_data, 
                                     num_steps=epochs, 
                                     lr = learning_rate, 
                                     batch_size = batch_size)

    def evaluate(
        self, 
        dataset: Dataset,
    ) -> Dict:
        """
        Evaluates the model on the test dataset.

        Parameters
        ----------
        dataset : Dataset
            The test dataset.

        Returns
        -------
        Dict
            Evaluation metrics.
        """
        X, y_test_ds = dataset.get_x_y()
        test_ds = self.preprocessor.transform(X)

        test_loader = np.hstack([test_ds, y_test_ds]) if y_test_ds is not None else np.hstack([test_ds])
        metrics = eval(self.hashed_architecture, self.params, test_loader, self.search_params)
        return metrics

    def reconstruction_error(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Computes the reconstruction error for the input data.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        y : np.ndarray, optional
            The target data. Defaults to None.

        Returns
        -------
        np.ndarray
            The reconstruction error for each instance.
        """
        instances = np.hstack([x, y]) if y is not None else x
        reconstruction_error = []

        for batch in np.array_split(instances, min(256, instances.shape[0]), axis=0):
            batch_reconstruction_error = eval(self.hashed_architecture, self.params, batch, self.search_params)["reconstruction_loss"]
            reconstruction_error.extend(batch_reconstruction_error)

        return np.array(reconstruction_error)

    def sample_from_latent_space(
        self, x: np.ndarray, ds: np.ndarray, y: np.ndarray = None, y_ds: np.ndarray = None, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples from the latent space around the given data point.

        Parameters
        ----------
        x : np.ndarray
            The data point to sample around.
        ds : np.ndarray
            The dataset to sample from.
        y : np.ndarray, optional
            The target values for `x`. Defaults to None.
        y_ds : np.ndarray, optional
            The target values for `ds`. Defaults to None.
        n_samples : int, optional
            The number of samples to generate. Defaults to 100.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The sampled data and the corresponding indices.
        """
        n_samples = min(n_samples, ds.shape[0] - 1)
        encoded_ds = self.encode(ds, y_ds)[0]
        encoded_x = self.encode(x, y)[0]
        distances = np.linalg.norm(encoded_ds - encoded_x, axis=1)
        idx = np.argpartition(distances, n_samples)[:n_samples]
        encoded_samples = encoded_ds[idx]
        return encoded_samples, idx

    def save(self, architecture_filename: str, sd_filename: str):
        """
        Saves the model architecture and parameters to files.

        Parameters
        ----------
        architecture_filename : str
            The file path to save the model architecture.
        sd_filename : str
            The file path to save the model parameters.
        """
        state_dict = serialization.to_state_dict(self.params)
        np.save(sd_filename, state_dict)
        if self.model_type == 'Diffusion':
            eqx.tree_serialise_leaves(f"{sd_filename}_diffusion.eqx", self.diffusion_model.model)
        with open(architecture_filename, "w") as f:
            json.dump(self.architecture, f)
