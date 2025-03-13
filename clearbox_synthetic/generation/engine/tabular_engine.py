import os
import json
import optax
import numpy as np
import pandas as pd
import scipy
import equinox as eqx
from typing import Sequence, Callable, Dict, List, Tuple, Literal
import jax
from jax import random
from flax.core.frozen_dict import FrozenDict
from flax import serialization
from flax.training import train_state
from tqdm import tqdm, trange
from loguru import logger

import os

####################
# # UNCOMMENT FOR DEBUGGING
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
# preprocessor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../preprocessor/clearbox-preprocessor"))
# sys.path.append(preprocessor_path)
# from clearbox_preprocessor import Preprocessor

# from clearbox_synthetic.utils import Dataset
# from clearbox_synthetic.generation.VAE.tabular_vae import TabularVAE, train_step, eval
# from clearbox_synthetic.generation.diffusion.tabular_diffusion import TabularDiffusion
# from clearbox_synthetic.generation.engine.engine import EngineInterface
####################

from clearbox_preprocessor import Preprocessor
from ...utils import Dataset
from ..VAE.tabular_vae import TabularVAE, train_step, eval
from ..diffusion.tabular_diffusion import TabularDiffusion
from .engine import EngineInterface

def _process_categorical(df: pd.DataFrame, input_feats: List, target: List) -> Tuple[Dict, Dict, pd.Index]:
    """
    Preprocesses categorical features, generates embedding rules, and bins numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        input_feats (List): List of input feature column names.
        target (List): List of target feature column names.

    Returns:
        Tuple: A dictionary of preprocessing rules, numerical binning information, and processed column names.
    """
    input_cat = [i for i in input_feats if df[i].dtype == 'object' or df[i].nunique() < 4]
    input_num = [i for i in input_feats if df[i].dtype != 'object' and df[i].nunique() >= 4]
    sub = df[input_cat + target]
    bins_num = {}
    for i in input_num:
        low = np.quantile(df[i].fillna(0), 0.01)
        high = np.quantile(df[i].fillna(0), 0.99)
        newco = pd.cut(np.clip(df[i], low, high), 3)
        bins_num[i] = newco.cat.categories
        sub = pd.concat([newco, sub], axis=1)
    
    for i in input_cat:
        sub[i] = sub[i].fillna('NaN')
        
    sub = sub.sample(n=min(5000, df.shape[0]), replace=False, random_state=42)
    dupli = sub.drop_duplicates(subset=input_feats)
    prepro_dict = {}
    print('Processing embedding rules:')
    for i in tqdm(range(dupli.shape[0])):
        mask = (sub.iloc[:, :-1] == dupli.iloc[i, :-1]).sum(axis=1) == len(input_feats)
        prepro_dict[dupli.iloc[i, :-1].to_string(index=False)] = sub[mask][target[0]].value_counts(dropna=False)
    
    return prepro_dict, bins_num, dupli.columns

class TabularEngine(EngineInterface):
    """
    This class integrates the ``TabularVAE`` and ``TabularDiffusion`` models to enable training, 
    evaluation, and inference for tabular datasets.

    Parameters
    ----------
    dataset : Dataset
        The dataset used to initialize the generative engine.

    layers_size : Sequence[int], optional, default=[50]
        The sizes of the hidden layers.

    params : FrozenDict, optional, default=None
        Model parameters. 

    train_params : Dict, optional, default=None
        Training parameters.

    privacy_budget : float, optional, default=1.0
        The privacy budget. 

    model_type : str, optional, default='VAE'
        Type of model ('VAE' or 'Diffusion').

    rules : Dict, optional, default={}
        Rules for embedding and transformations. 
    
    cat_labels_threshold : float, optional, default=0.02
        A float value between 0 and 1 that sets the threshold for discarding categorical features.
        It defines a minimum frequency threshold for keeping a label as a separate category. If a label appears 
        in less than :code:`cat_labels_threshold * 100%` of the total occurrences in a categorical column, it is grouped 
        into a generic ``"other"`` category. 

        For instance, if ``cat_labels_threshold=0.02`` and a label appears less than 2% in the dataset, that label will be converted to `"other"`.

    get_discarded_info : bool, optional, default=False
        If set to ``True``, the preprocessor will feature the method ``get_discarded_features_reason``,
        which provides information on which columns were discarded and the reason for discarding.
        Note that enabling this option may significantly slow down the processing operation.
        The list of discarded columns is available even when `get_discarded_info=False`, so consider
        setting this flag to ``True`` only if you need to know why a column was discarded or, in the case
        of columns containing only one unique value, what that value was.

    excluded_col : List, optional, default=[]
        A list of column names to be excluded from processing. These columns will be returned in the
        final DataFrame without being modified.

    scaling : str, default="none"
        The method used to scale numerical features:

        - "none"        : No scaling is applied   
        - "normalize"   : Normalizes numerical features to the [0, 1] range.
        - "standardize" : Standardizes numerical features to have a mean of 0 and a standard deviation of 1.
        - "quantile"    : Transforms numerical features using quantiles information.
        - "kbins"       : Converts continuous numerical data into discrete bins. The number of bins is defined by the parameter n_bin

    num_fill_null : FillNullStrategy or str, default="mean"
        Strategy or value used to fill null values in numerical features:

        - "mean"        : Fills null values with the mean of the column.
        - "interpolate" : Fills null values using interpolation.
        - "forward"     : Fills null values using the previous non-null value.
        - "backward"    : Fills null values using the next non-null value.
        - "min"         : Fills null values with the minimum value of the column.
        - "max"         : Fills null values with the maximum value of the column.
        - "zero"        : Fills null values with zeros.
        - "one"         : Fills null values with ones.
        - value         : Fills null values with the specified value.

    n_bins : int, default=0
        Number of bins to discretize numerical features. If set to a value greater than 0 and if scaling=="kbins",
        numerical features are discretized into the specified number of bins using quantile-based
        binning.

    unseen_labels : str, default="ignore"
        - "ignore"        : If new data contains labels unseen during fit one hot encoding contains 0 in every column.
        - "error"         : Raise an error if new data contains labels unseen during fit.

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
        layers_size: Sequence[int] = [50],
        params: FrozenDict = None,
        train_params: Dict = None,
        privacy_budget: float = 1.0,
        model_type: str = 'VAE',
        rules: Dict = {},
        cat_labels_threshold: float = 0.02,
        get_discarded_info: bool = False,
        excluded_col: List = [],
        missing_values_threshold: float = 0.999,
        n_bins: int = 0,
        scaling: Literal["none", "normalize", "standardize", "quantile"] = "quantile", 
        num_fill_null : Literal["interpolate","forward", "backward", "min", "max", "mean", "zero", "one"] = "mean",
        unseen_labels = 'ignore',
    ):
        self._enforce_cpu_if_no_gpu()
        
        # Save all preprocessor arguments as class attributes
        self.cat_labels_threshold = cat_labels_threshold
        self.excluded_col = excluded_col
        self.missing_values_threshold = missing_values_threshold
        self.n_bins = n_bins
        self.scaling = scaling
        self.num_fill_null = num_fill_null
        self.unseen_labels = unseen_labels
        
        self.model_type = model_type
        self.rules = rules
        self.emb_rules = {}
            
        rng = random.PRNGKey(0)
        rng, key = random.split(rng)

        X, Y = dataset.get_x_y()
        if Y is not None:
            y_shape=Y[0].shape  
        else:
            y_shape = [0]

        for w in [i for i in rules.keys() if 'embed_category' in rules[i][0]]:
            self.emb_rules[w] = _process_categorical(X, rules[w][1], rules[w][2])
            X = X.drop(w, axis=1)

        for w in [i for i in rules.keys() if 'sum' in rules[i][0]]:
            X = X.drop(w, axis=1)

        self.preprocessor = Preprocessor(
            X,
            cat_labels_threshold     = self.cat_labels_threshold,
            excluded_col             = self.excluded_col,
            missing_values_threshold = self.missing_values_threshold,
            n_bins                   = self.n_bins,
            scaling                  = self.scaling,
            num_fill_null            = self.num_fill_null,
            unseen_labels            = self.unseen_labels,
        ) 
        X_train = self.preprocessor.transform(X)

        x_shape = X_train.to_numpy()[0].shape

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
        self.val_loss   = metrics_val
        self.params     = state.params

        if self.model_type == 'Diffusion':
            _, diff_train_data, _ = self.model.apply({"params": self.params}, train_ds.to_numpy(), y_train_ds)
            self.diffusion_model.fit(diff_train_data, 
                                     num_steps=epochs, 
                                     lr = learning_rate, 
                                     batch_size = batch_size)

    def evaluate(self, test_ds: np.ndarray, y_test_ds: np.ndarray = None) -> Dict:
        """
        Evaluates the model on the test dataset.

        Parameters
        ----------
        test_ds : np.ndarray
            The test dataset.
        y_test_ds : np.ndarray, optional
            The target values for the test dataset. Defaults to None.

        Returns
        -------
        Dict
            Evaluation metrics.
        """
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

    def _sample_vae(self, x, recon_x):
            """
            Sample data from a Variational Autoencoder (VAE) using the original and reconstructed data.

            Args:
                x (np.ndarray): Original input data.
                recon_x (np.ndarray): Reconstructed data from the VAE.

            Returns:
                pd.DataFrame: The inverse-transformed synthetic data.
            """
            
            # preprocessed_x = self.preprocessor.transform(x).to_numpy()

            n_numerical_features = (
                self.preprocessor.get_features_sizes()[0][0] if self.preprocessor.get_features_sizes()[0] else 0
            )
            categorical_features_sizes = self.preprocessor.get_features_sizes()[1]

            numerical_features_sampled = np.zeros((x.shape[0], n_numerical_features))
            
            for i in range(n_numerical_features):
                numerical_features_sampled[:, i] = (
                    recon_x[:, i] + self.search_params["gauss_s"] * np.random.randn(recon_x.shape[0])
                )

            categorical_features_sampled = np.zeros(
                (x.shape[0], x.shape[1] - n_numerical_features)
            )
            view_decoded = recon_x[:, n_numerical_features:]

            for i in range(x.shape[0]):
                w2 = 0  # index categorical label in preprocessed space
                w3 = 0  # index categorical feature
                features = x[i, n_numerical_features:] > 0
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
            return e

    def generate(
            self, 
            dataset: Dataset,
            n_samples: int = 100, 
            noise: float = 0.0, 
            random_state: int = 42
        ) -> np.ndarray:
        """
        Generates synthetic data from the model.

        Parameters
        ----------
        dataset : Dataset
            The input data to condition the generation on. If None, random samples will be generated.
        n_samples : int, optional
            The number of samples to generate. Defaults to 100.
        noise : float, optional
            The amount of noise to add to the latent space. Defaults to 0.0.
        random_state : int, optional
            The random seed for reproducibility. Defaults to 42.

        Returns
        -------
        np.ndarray
            The generated synthetic data.
        """
        rng = random.PRNGKey(random_state)
        
        if dataset is None:
            y = None
            # Generate completely random samples if no conditioning data is provided
            if self.model_type == 'Diffusion':
                samples = self.diffusion_model.sample(n_samples, rng)

                # Decode the samples back to the original space
                return self.model.apply({"params": self.params}, samples, y, method=self.model.decode)
            else:
                # Generate random latent vectors
                rng, latent_key = random.split(rng)
                latent_dim = self.architecture["layers_size"][-1]
                z = random.normal(latent_key, (n_samples, latent_dim))

                # Decode the latent vectors to generate synthetic data
                generated_np = self.model.apply({"params": self.params}, z, y, method=self.model.decode)
                return self.preprocessor.inverse_transform(pd.DataFrame(generated_np))
        else:
            x, y = dataset.get_x_y()
            x = self.preprocessor.transform(x)
            if self.model_type == 'Diffusion':
                # Use the VAE to encode the input data first
                z_mean, z_logvar = self.model.apply({"params": self.params}, x.to_numpy(), y, method=self.model.encode)

                # Add noise to the latent representation if specified
                if noise > 0:
                    rng, noise_key = random.split(rng)
                    z_noise = random.normal(noise_key, z_mean.shape) * noise
                    z_mean = z_mean + z_noise

                # Use the diffusion model to generate samples conditioned on the latent representation
                samples = self.diffusion_model.sample(n_samples, rng, condition=z_mean)

                # Decode the samples back to the original space
                generated_np = self.model.apply({"params": self.params}, samples, y, method=self.model.decode)
                generated_df = self.preprocessor.inverse_transform(pd.DataFrame(generated_np,columns=x.columns))
            else:
                # Encode the input data to get latent representations
                if y is not None:
                    recon_x = self.apply(x.to_numpy(), y)[0]
                else:
                    recon_x = self.apply(x.to_numpy())[0]

                generated_np = self._sample_vae(x.to_numpy(), recon_x)
                generated_df = self.preprocessor.inverse_transform(pd.DataFrame(generated_np,columns=x.columns))

            # Add the target column on which the generation was conditioned
            generated_df[dataset.target_column] = dataset.data[dataset.target_column]
            return generated_df

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

