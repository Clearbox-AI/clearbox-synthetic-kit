import json
import optax
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Sequence, Tuple, Dict
from jax import random
from flax.core.frozen_dict import FrozenDict
from flax import serialization
from flax.training import train_state
from tqdm import trange
from loguru import logger

from clearbox_preprocessor import Preprocessor

from clearbox_synthetic.utils import Dataset
from clearbox_synthetic.generation.engine.engine import EngineInterface
from clearbox_synthetic.generation.VAE.timeseries_vae import TimeSeriesVAE, train_step, eval

class TimeSeriesEngine(EngineInterface):
    """
    Manages training and evaluation of a time series model using a Variational Autoencoder (VAE). 
    It handles model initialization, training, evaluation, and saving functionalities.

    Attributes
    ----------
    model : TimeSeriesVAE
        The Variational Autoencoder model for time series data.
    params : FrozenDict
        The parameters of the model.
    search_params : Dict
        The training parameters.
    architecture : Dict
        The architecture details of the model.
    hashed_architecture : str
        A hashed string representation of the model architecture.
    """

    def __init__(
        self,
        dataset: Dataset,
        time_id: str,
        layers_size: Sequence[int] = [40],
        params: FrozenDict = None,
        train_params: Dict = None,
        num_heads: int = 4,
    ):
        """
        Initializes the TimeSeriesEngine with the given parameters.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing time series data.
        time_id : str
            The time series index that identifies single events.
        layers_size : Sequence[int]
            List of sizes for the hidden layers.
        params : FrozenDict, optional
            Model parameters. Defaults to None.
        train_params : Dict, optional
            Training parameters. Defaults to None.
        num_heads : int
            Number of self-attention heads.
        """

        self.meta_columns = [i for i in dataset.data.columns if (dataset.data[i].dtype in ['object','category']) and (i not in [time_id])]
        self.time_id = time_id        
        X = dataset.data
        self.time_columns = [i for i in dataset.data.columns if i not in [time_id]+self.meta_columns]
        self.time_mean= dataset.data[self.time_columns].mean(axis=0).values
        self.time_std= dataset.data[self.time_columns].std(axis=0).values
        self.n_time_features = len(self.time_columns)
        n =  X[[self.time_id,self.time_columns[0]]].groupby(self.time_id).count().max().values[0]
        self.max_sequence_length = n if n % 2 == 0 else n + 1
        self.prepro = Preprocessor(dataset.data[self.meta_columns])
        self.num_heads = num_heads
        rng = random.PRNGKey(0)
        rng, key = random.split(rng)

        if train_params is None:
            train_params = {
                "l2_reg": 0.000,
                "beta": 0,
                "alpha": 0.1,
                "gauss_s": 0.01,
                "gauss_s_c": 0.1,
                "weight_decay": 0.000,
                "prob_clip": 0.99,
            }

        self.search_params = train_params
        self.x_shape = self.n_time_features * self.max_sequence_length
        self.y_shape = self.prepro.transform(dataset.data[self.meta_columns]).shape[1]
        self.model = TimeSeriesVAE(
            encoder_widths=layers_size,
            decoder_widths=layers_size[::-1],
            y_shape=self.y_shape,
            feature_sizes=self.n_time_features,
            max_sequence_length=self.max_sequence_length,
            num_heads=self.num_heads,
            search_params=train_params,
        )

        x = random.uniform(key, [np.prod(self.x_shape)])

        if self.y_shape != 0:
            y = random.uniform(key, [np.prod(self.y_shape)])
        else:
            y = None

        self.params = params or self.model.init(rng, x, y)["params"]
        self.search_params = train_params

        self.architecture = {
            "layers_size": layers_size,
            "y_shape": [int(self.y_shape)],
            "feature_sizes": int(self.n_time_features),
            "max_sequence_length": int(self.max_sequence_length),
            "num_heads": int(self.num_heads)}

        self.hashed_architecture = json.dumps(self.architecture)

    def apply(self, x: np.ndarray, y: np.ndarray = None) -> Tuple:
        """
        Applies the model to the input data.

        Parameters
        ----------
        x : np.ndarray
            Input data.
        y : np.ndarray, optional
            Target data. Defaults to None.

        Returns
        -------
        Tuple
            The model's output.
        """
        return self.model.apply({"params": self.params}, x, y)

    def encode(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Encodes the input data into the latent space.

        Parameters
        ----------
        x : np.ndarray
            Input data.
        y : np.ndarray, optional
            Target data. Defaults to None.

        Returns
        -------
        np.ndarray
            Encoded data.
        """
        return self.model.apply({"params": self.params}, x, y, method=self.model.encode)

    def decode(self, z: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Decodes the latent space representation into the original space.

        Parameters
        ----------
        z : np.ndarray
            Latent space representation.
        y : np.ndarray, optional
            Target data. Defaults to None.

        Returns
        -------
        np.ndarray
            Decoded data.
        """
        return self.model.apply({"params": self.params}, z, y, method=self.model.decode)

    def fit(
        self,
        dataset: Dataset,
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-2,
        val_dataset: Dataset = None,
        patience: int = 4,
    ):
        """
        Trains the model on the provided dataset.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.
        epochs : int, optional
            Number of training epochs. Defaults to 20.
        batch_size : int, optional
            Size of each training batch. Defaults to 128.
        learning_rate : float, optional
            Learning rate for the optimizer. Defaults to 1e-2.
        val_ds : np.ndarray, optional
            Validation dataset. Defaults to None.
        y_val_ds : np.ndarray, optional
            Target values for the validation dataset. Defaults to None.
        patience : int, optional
            Number of epochs to wait before early stopping. Defaults to 4.
        """
        ntrain = dataset.data[self.time_id].nunique()

        X_proc = np.zeros((ntrain, self.max_sequence_length*self.n_time_features))
        X = dataset.data
        meta_data = dataset.data.groupby(self.time_id).first()[self.meta_columns]
        y_train_ds = self.prepro.transform(meta_data)
        logger.info('Preprocessing training time series')
        for j,i in tqdm(enumerate(X[self.time_id].unique())):
            dt = (X[X[self.time_id]==i][self.time_columns].values-self.time_mean)/self.time_std
            L = self.max_sequence_length-dt.T.shape[1]
            dt = np.hstack([dt.T,np.zeros((self.n_time_features,L))])
            X_proc[j,:] = dt.reshape(self.max_sequence_length*self.n_time_features,-1).T
        
        
        if val_dataset:
            X_proc_val = np.zeros((ntrain, self.max_sequence_length*self.n_time_features))
            X_val = val_dataset.data            
            meta_data_val = val_dataset.data.groupby(self.time_id).first()[self.meta_columns]
            y_val_ds = self.prepro.transform(meta_data_val)
            logger.info('Preprocessing validation time series')
            for j,i in tqdm(enumerate(X_val[self.time_id].unique())):
                dt = (X_val[X_val[self.time_id]==i][self.time_columns].values-self.time_mean)/self.time_std
                L = self.max_sequence_length-dt.T.shape[1]
                dt = np.hstack([dt.T,np.zeros((self.n_time_features,L))])
                X_proc_val[j,:] = dt.reshape(self.max_sequence_length*self.n_time_features,-1).T
                
        train_loader = np.hstack([X_proc, y_train_ds]) if y_train_ds is not None else np.hstack([X_proc])
            
        weight_decay = self.search_params["weight_decay"]

        state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            params   = self.params,
            tx       = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        )
                

        metrics_train = None
        metrics_val = None
        splits = np.arange(batch_size, train_loader.shape[0], batch_size)
        loop_range = trange(epochs, desc="Engine fitting in progress", unit="epoch", leave=True)

        total_loss = 1e5
        no_improvement = 0

        for i in loop_range:
            for batch in np.array_split(train_loader, splits, axis=0):
                state = train_step(self.hashed_architecture, state, batch, self.search_params)

            if i % 25 == 0:  # Update progress every 25 iterations
                self.params = state.params
                metrics_train = self.evaluate(X_proc, y_train_ds)
                if val_dataset is not None:
                    metrics_val = self.evaluate(X_proc_val, y_val_ds)
                    loop_range.set_postfix({
                        "Train loss": metrics_train["mean_reconstruction_loss"],
                        "Val loss": metrics_val["mean_reconstruction_loss"]
                    }, refresh=True)
                    loss_i = metrics_val["loss"]
                else:
                    loop_range.set_postfix({
                        "Train loss": metrics_train["mean_reconstruction_loss"]
                    }, refresh=True)
                    loss_i = metrics_train["loss"]

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

    def evaluate(self, test_ds: np.ndarray, y_test_ds: np.ndarray = None) -> Dict:
        """
        Evaluates the model on the test dataset.

        Args:
            test_ds (np.ndarray): Test dataset.
            y_test_ds (np.ndarray, optional): Target values for the test dataset. Defaults to None.

        Returns:
            Dict: Evaluation metrics.
        """
        if y_test_ds is not None:
            test_loader = np.hstack([test_ds, y_test_ds])
        else:
            test_loader = np.hstack([test_ds])

        metrics = eval(self.hashed_architecture, self.params, test_loader, self.search_params)
        return metrics

    def reconstruction_error(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Computes the reconstruction error for the input data.

        Parameters
        ----------
        x : np.ndarray
            Input data.
        y : np.ndarray, optional
            Target data. Defaults to None.

        Returns
        -------
        np.ndarray
            Reconstruction error for each instance.
        """
        instances = np.hstack([x, y]) if y is not None else x
        reconstruction_error = np.empty(shape=instances.shape[0])
        i = 0

        for batch in np.array_split(instances, min(256, instances.shape[0]), axis=0):
            batch_reconstruction_error = eval(
                self.hashed_architecture, self.params, batch, self.search_params
            )["reconstruction_loss"]
            for error in batch_reconstruction_error:
                reconstruction_error[i] = error
                i += 1

        return reconstruction_error

    def sample_from_latent_space(
        self, x: np.ndarray, ds: np.ndarray, y: np.ndarray = None, y_ds: np.ndarray = None, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples data from the latent space close to the given input.

        Parameters
        ----------
        x : np.ndarray
            Input data.
        ds : np.ndarray
            Dataset to sample from.
        y : np.ndarray, optional
            Target data for the input. Defaults to None.
        y_ds : np.ndarray, optional
            Target data for the dataset. Defaults to None.
        n_samples : int, optional
            Number of samples to generate. Defaults to 100.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Sampled data and their indices.
        """
        n_samples = min(n_samples, ds.shape[0] - 1)
        encoded_ds = self.encode(ds, y_ds)[0]
        encoded_x = self.encode(x, y)[0]
        distances = ((encoded_ds - encoded_x) ** 2).sum(axis=1) ** 0.5
        idx = np.argpartition(np.asarray(distances), n_samples)[:n_samples]
        encoded_samples = encoded_ds[idx]

        return encoded_samples, idx
    
    def generate(
            self, 
            dataset: Dataset,
            n_samples: int = 100, 
        ):
        """
        Generates synthetic time series data from the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 100.
        """
        synth_data = self.decode(np.random.randn(n_samples,
                                                 self.architecture['layers_size'][0]))  # b.shape[1] dimensione dato da generare
        indeces = dataset.data[self.time_id].values    
        dfs = []

        for i in tqdm(range(synth_data.shape[0])):
            x_i = synth_data[i,:]
            time_series = []
            for j,feat_name in enumerate(self.time_columns):
                ts =x_i[self.max_sequence_length*j:self.max_sequence_length*(j+1)]
                ts = ts * self.time_std[j] + self.time_mean[j]
                time_series.append(ts)
            df_i = pd.DataFrame(np.array(time_series).T)
            df_i.columns = self.time_columns
            df_i['id'] = i
            dfs.append(df_i)    
        
        return pd.concat(dfs, axis= 0)  

    def save(self, architecture_filename: str, sd_filename: str):
        """
        Saves the model architecture and parameters.

        Parameters
        ----------
        architecture_filename : str
            Filename to save the model architecture.
        sd_filename : str
            Filename to save the state dictionary.
        """
        state_dict = serialization.to_state_dict(self.params)
        np.save(sd_filename, state_dict)
        with open(architecture_filename, "w") as f:
            json.dump(self.architecture, f)