import json
import optax
import numpy as np
import equinox as eqx
import pandas as pd
from tqdm import tqdm
from typing import Sequence, Tuple, Callable, Dict, Literal, List
from jax import random
from flax.core.frozen_dict import FrozenDict
from flax import serialization
from flax.training import train_state
from tqdm import trange
from loguru import logger

# from ..VAE.timeseries_vae import TimeSeriesVAE, train_step, eval
# from .engine import EngineInterface
# from ...utils import Dataset
# from clearbox_preprocessor import Preprocessor

####################
# # UNCOMMENT FOR DEBUGGING
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
preprocessor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../preprocessor/clearbox-preprocessor"))
sys.path.append(preprocessor_path)
from clearbox_preprocessor import Preprocessor

from clearbox_synthetic.utils import Dataset
from clearbox_synthetic.generation.engine.engine import EngineInterface
from clearbox_synthetic.generation.VAE.timeseries_vae import TimeSeriesVAE, train_step, eval
####################
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
        # num_heads: Sequence[int] = 1,
        layers_size: Sequence[int] = [40],
        params: FrozenDict = None,
        train_params: Dict = None,
        privacy_budget: float = 1.0,
        cat_labels_threshold: float = 0.02,
        get_discarded_info: bool = False,
        excluded_col: List = [],
        missing_values_threshold: float = 0.999,
        n_bins: int = 0,
        scaling: Literal["none", "normalize", "standardize", "quantile"] = "normalize", 
        num_fill_null : Literal["interpolate","forward", "backward", "min", "max", "mean", "zero", "one"] = "interpolate",
        unseen_labels = 'ignore',
    ):
        """
        Initializes the TimeSeriesEngine with the given parameters and validates the license.

        Parameters
        ----------
        layers_size : Sequence[int]
            List of sizes for the hidden layers.
        time_id : str
            The time series index that identifies single events.
        max_sequence_length : Sequence[int]
            Maximum length of the input sequences.
        num_heads : Sequence[int]
            Number of self-attention heads.
        y_shape : Sequence[int], optional
            Shape of the target variable. Defaults to [0].
        params : FrozenDict, optional
            Model parameters. Defaults to None.
        train_params : Dict, optional
            Training parameters. Defaults to None.
        train_loss : Dict, optional
            Training loss information. Defaults to None.
        val_loss : Dict, optional
            Validation loss information. Defaults to None.
        privacy_budget : float, optional
            Privacy budget for the model. Defaults to 1.0.
        """
        num_heads = 1

        # Save all preprocessor arguments as class attributes
        self.time_id = time_id
        self.cat_labels_threshold = cat_labels_threshold
        self.get_discarded_info = get_discarded_info
        self.excluded_col = excluded_col
        self.missing_values_threshold = missing_values_threshold
        self.n_bins = n_bins
        self.scaling = scaling
        self.num_fill_null = num_fill_null
        self.unseen_labels = unseen_labels

        rng = random.PRNGKey(0)
        rng, key = random.split(rng)

        X, Y = dataset.get_x_y()
        if Y is not None:
            y_shape=Y[0].shape  
        else:
            y_shape = [0]

        # Determine the time series max length by taking the 95th percentile of the series lengths
        series_lengths = X.groupby(time_id).size()
        max_sequence_length = int(series_lengths.quantile(0.95))

        self.preprocessor = Preprocessor(
            X,
            time_id = self.time_id,
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
        layers_size = [int(x_shape[0])]

        feature_sizes = self.preprocessor.get_features_sizes()[0][0]

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

        self.privacy_budget = privacy_budget
        self.search_params = train_params
        x_shape = feature_sizes * max_sequence_length

        self.model = TimeSeriesVAE(
            encoder_widths=layers_size,
            decoder_widths=layers_size[::-1],
            y_shape=y_shape,
            feature_sizes=feature_sizes,
            max_sequence_length=max_sequence_length,
            num_heads=num_heads,
            search_params=train_params,
        )

        x = random.uniform(key, [np.prod(x_shape)])

        if y_shape != [0]:
            y = random.uniform(key, [np.prod(y_shape)])
        else:
            y = None

        self.params = params or self.model.init(rng, x, y)["params"]
        self.search_params = train_params

        self.architecture = {
            "layers_size": layers_size,
            "y_shape": y_shape,
            "feature_sizes": feature_sizes,
            "max_sequence_length": max_sequence_length,
            "num_heads": num_heads,
        }

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
        weight_decay = self.search_params["weight_decay"]

        state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            params   = self.params,
            tx       = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        )
        
        X, y_train_ds = dataset.get_x_y()
        train_ds = self.preprocessor.transform(X)

        train_loader = np.hstack([train_ds, y_train_ds]) if y_train_ds is not None else np.hstack([train_ds])

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
                metrics_train = self.evaluate(train_ds, y_train_ds)
                if val_ds is not None:
                    metrics_val = self.evaluate(val_ds, y_val_ds)
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
            n_samples: int = 100, 
        ):
        """
        Generates synthetic time series data from the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 100.
        """
        # Generate the synthetic time series by decoding the samples from a gaussian distribution
        synth_data = self.decode(np.random.randn(n_samples,self.architecture['layers_size'][0]))  # b.shape[1] dimension of data to be generated
        #indeces = train_dataset.data[prepro.time_index].sample(N, replace = False).values
        df =  self.preprocessor.inverse_transform(synth_data)
        dfs = []
        # Create a dataframe with the original schema
        for i in tqdm(range(df.shape[0])):
            x_i = df.iloc[i]
            time_series = []
            for feat_name in self.preprocessor.datetime_features:
                time_series.append(x_i[[j for j in df.columns if feat_name in j]].values)
            df_i = pd.DataFrame(np.array(time_series).T)
            df_i.columns = self.preprocessor.datetime_features
            
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

if __name__ == "__main__":
    import os
    import pandas as pd
    import polars as pl

    file_path = "https://raw.githubusercontent.com/Clearbox-AI/clearbox-synthetic-kit/main/tutorials/time_series/data/daily_delhi_climate"
    path=os.path.join(file_path, "DailyDelhiClimateTrain.csv")
    df = pd.read_csv(path)

    # Adding a time index column with month, as "mm" to be used as target column
    df['id'] =df['date'].apply(lambda x: ''.join(x.split('-')[0:2]))
    train_dataset = Dataset.from_dataframe(df)

    engine = TimeSeriesEngine(train_dataset, time_id = "id", scaling="normalize")
    engine.fit(train_dataset, epochs=1, batch_size=124, learning_rate=0.00001)
    # synthetic_df = engine.generate(n_samples=100)
    a = 1