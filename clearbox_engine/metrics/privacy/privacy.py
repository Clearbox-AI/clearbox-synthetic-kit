# import sys
import pyximport
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from loguru import logger
from typing import List, Union
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

from clearbox_engine import Preprocessor, Dataset

pyximport.install(setup_args={"include_dirs": np.get_include()})
from clearbox_engine.metrics.privacy.gower_matrix_c import gower_matrix_c


def sample_datasets(
    training_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    synthetic_set: pd.DataFrame,
    max_sample_size: int = 70000,
):
    """
    Given three dataframes (training, validation and synthetic set) with (potentially)
    three different sizes (= number of rows), sample and return the three sets with
    the same number of rows.

    IMPORTANT: It is assumed that synthetic_set.shape[0]<=training_set.shape[0] and
    validation_set.shape[0]<=training_set.shape[0]

    Parameters
    ----------
    training_set : pd.DataFrame
        The training set as a pandas DataFrame.
    validation_set : pd.DataFrame
        The validation set as a pandas DataFrame.
    synthetic_set : pd.DataFrame
        The synthetic set as a pandas DataFrame.
    max_sample_size : int, optional
        Max number of rows to sample, by default 100000

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Three pandas dataframes with the same number of rows.
    """

    training_size = training_set.shape[0]
    validation_size = validation_set.shape[0]
    synthetic_size = synthetic_set.shape[0]

    if synthetic_size > training_size:
        raise ValueError(
            "Size of synthetic set ({}) must not be larger than the training set one ({}).".format(
                synthetic_size, training_size
            )
        )

    if validation_size > training_size:
        raise ValueError(
            "Size of validation set ({}) must not be larger than the training set one ({}) ".format(
                validation_size, training_size
            )
        )

    sample_size = int(
        min(max_sample_size, training_size, synthetic_size, validation_size)
    )
    fixed_random_state = 42

    if sample_size == training_size:
        sampled_training_set = training_set
    else:
        _, sampled_training_set = train_test_split(
            training_set, test_size=sample_size, random_state=fixed_random_state
        )

    if sample_size == synthetic_size:
        sampled_synthetic_set = synthetic_set
    else:
        _, sampled_synthetic_set = train_test_split(
            synthetic_set, test_size=sample_size, random_state=fixed_random_state
        )

    if validation_size > sample_size:
        _, sampled_validation_set = train_test_split(
            validation_set, test_size=sample_size, random_state=fixed_random_state
        )
    else:
        sampled_validation_set = validation_set

    return sampled_training_set, sampled_validation_set, sampled_synthetic_set


def sample_datasets_for_dcr_test(
    training_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    synthetic_set: pd.DataFrame,
    max_sample_size: int = 10000,
):
    """
    Samples datasets for Distance to the Closest Record (DCR) tests, ensuring consistent sizes.

    Parameters
    ----------
        training_set : pd.DataFrame
            The training set as a pandas DataFrame.
        validation_set : pd.DataFrame
            The validation set as a pandas DataFrame.
        synthetic_set : pd.DataFrame
            The synthetic set as a pandas DataFrame.
        max_sample_size : int, optional
            Maximum sample size. Defaults to 10000.

    Returns
    -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Sampled training, validation, and synthetic sets.
    """
    training_size = training_set.shape[0]
    validation_size = validation_set.shape[0]
    synthetic_size = synthetic_set.shape[0]
    sample_size = int(
        min(max_sample_size, training_size, validation_size, synthetic_size)
    )

    fixed_random_state = 42
    if sample_size == training_size:
        sampled_training_set = training_set
    else:
        _, sampled_training_set = train_test_split(
            training_set, test_size=sample_size, random_state=fixed_random_state
        )

    if sample_size == synthetic_size:
        sampled_synthetic_set = synthetic_set
    else:
        _, sampled_synthetic_set = train_test_split(
            synthetic_set, test_size=sample_size, random_state=fixed_random_state
        )

    if validation_size > sample_size:
        _, sampled_validation_set = train_test_split(
            validation_set, test_size=sample_size, random_state=fixed_random_state
        )
    else:
        sampled_validation_set = validation_set

    return sampled_training_set, sampled_validation_set, sampled_synthetic_set


def preprocess_datasets(
    training_set: pd.DataFrame,
    synthetic_set: pd.DataFrame,
    categorical_features: Union[List[bool], np.ndarray],
    validation_set: pd.DataFrame = None,
):
    """
    Given three dataframes (training, validation and synthetic set), cast categorical
    columns into pd.Categorical and ordinal columns into pd.Numeric. Any columns not
    numeric is considered categorical.

    Every categorical features is processed and transformed into pandas categorical
    codes (integers).

    The three datasets are merged/concatenated together and preprocessing is performed
    on the new dataset then they are splitted again. This is to prevent the presence/absence
    of a categorical feature in one or some of the sets from leading to a mismatch between the
    categorical codes assigned to the same feature in different sets.

    Args:
        training_set : pd.DataFrame
            The training set as a pandas DataFrame.
        synthetic_set : pd.DataFrame
            The synthetic set as a pandas DataFrame.
        categorical_features : List[bool]
            A list of boolean: categorical_features[i]==True if feature i is categorical,
            False otherwise.
        validation_set : pd.DataFrame, optional
            The validation set as a pandas DataFrame., by default None

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, (pd.DataFrame)]
            The DataFrames given as parameters preprocessed with the right dtypes.
    """

    if len(list(training_set.columns)) != len(categorical_features):
        raise ValueError(
            "'categorical_features' size ({}) doesn't match the number of columns of the 'training_set' ({}).".format(
                len(list(training_set.columns)), len(categorical_features)
            )
        )

    if list(training_set.columns) != list(synthetic_set.columns) or (
        validation_set is not None
        and list(training_set.columns) != list(validation_set.columns)
    ):
        raise ValueError(
            "Columns mismatch error, datasets given as parameters must have the same columns."
        )

    training_numerics = [
        True if is_numeric_dtype(column_dtype) else False
        for column_dtype in training_set.dtypes
    ]
    synthetic_numerics = [
        True if is_numeric_dtype(column_dtype) else False
        for column_dtype in synthetic_set.dtypes
    ]

    if validation_set is not None:
        validation_numerics = [
            True if is_numeric_dtype(column_dtype) else False
            for column_dtype in validation_set.dtypes
        ]

    if training_numerics != synthetic_numerics or (
        validation_set is not None and training_numerics != validation_numerics
    ):
        raise ValueError(
            "Columns types mismatch error, datasets given as parameters must have the same dtypes."
        )

    preprocessed_training_set = training_set.copy(deep=True)
    training_set_rows = len(preprocessed_training_set)
    preprocessed_synthetic_set = synthetic_set.copy(deep=True)
    synthetic_set_rows = len(preprocessed_synthetic_set)

    whole_dataset = pd.concat(
        objs=[preprocessed_training_set, preprocessed_synthetic_set], axis=0
    )

    if validation_set is not None:
        preprocessed_validation_set = validation_set.copy(deep=True)
        whole_dataset = pd.concat(
            objs=[whole_dataset, preprocessed_validation_set], axis=0
        )

    columns = list(whole_dataset.columns)

    for column_name, is_categorical in zip(columns, categorical_features):
        if is_categorical:
            whole_dataset[column_name] = pd.Categorical(whole_dataset[column_name])
            whole_dataset[column_name] = whole_dataset[column_name].cat.codes
            # If column contains Nans fill with empty string and add a category for it
            if whole_dataset.loc[:, column_name].isnull().sum() != 0:
                whole_dataset.loc[:, column_name] = (
                    whole_dataset.loc[:, column_name].cat.add_categories("").fillna("")
                )
        else:
            try:
                whole_dataset.loc[:, column_name] = pd.to_numeric(
                    whole_dataset[column_name]
                )
            except ValueError as e:
                if "Unable to parse string" in str(e):
                    raise ValueError(
                        "Unable to process column '{}'. You marked it as numeric, but it is actually categorical. Check the categorical_features parameter.".format(
                            column_name
                        )
                    )
                else:
                    raise e

    if validation_set is not None:
        preprocessed_training_set = whole_dataset[:training_set_rows]
        preprocessed_synthetic_set = whole_dataset[
            training_set_rows : training_set_rows + synthetic_set_rows
        ]
        preprocessed_validation_set = whole_dataset[
            training_set_rows + synthetic_set_rows :
        ]

        return (
            preprocessed_training_set,
            preprocessed_synthetic_set,
            preprocessed_validation_set,
        )
    else:
        preprocessed_training_set = whole_dataset[:training_set_rows]
        preprocessed_synthetic_set = whole_dataset[training_set_rows:]
        return preprocessed_training_set, preprocessed_synthetic_set


def adversary_dataset(
    training_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    original_dataset_sample_fraction: float = 0.2,
):
    """
    Create an adversary dataset (K) for the Membership Inference Test given a training
    and validation set. The validation set must be smaller than the training set.

    The size of the resulting adversary dataset is a fraction of the sum of the training
    set size and the validation set size.

    It takes half of the final rows from the training set and the other half from the
    validation set. It adds a column to mark which rows was sampled from the training set.

    Parameters
    ----------
    training_set : pd.DataFrame
        The training set as a pandas DataFrame.
    validation_set : pd.DataFrame
        The validation set as a pandas DataFrame.
    original_dataset_sample_fraction : float, optional
        How many rows (a fraction from 0  to 1) to sample from the concatenation of the
        training and validation set, by default 0.2

    Returns
    -------
    pd.DataFrame
        A new pandas DataFrame in which half of the rows come from the training set and
        the other half come from the validation set.
    """
    sample_number_of_rows = (
        training_set.shape[0] + validation_set.shape[0]
    ) * original_dataset_sample_fraction

    # if the validation set is very small, we'll set the number of rows to sample equal to
    # the number of rows of the validation set, that is every row of the validation set
    # is going into the adversary set.
    sample_number_of_rows = min(int(sample_number_of_rows / 2), validation_set.shape[0])

    sampled_from_training = training_set.sample(
        sample_number_of_rows, replace=False, random_state=42
    )
    sampled_from_training["privacy_test_is_training"] = True

    sampled_from_validation = validation_set.sample(
        sample_number_of_rows, replace=False, random_state=42
    )
    sampled_from_validation["privacy_test_is_training"] = False

    adversary_dataset = pd.concat(
        [sampled_from_training, sampled_from_validation], ignore_index=True
    )
    adversary_dataset = adversary_dataset.sample(frac=1).reset_index(drop=True)
    return adversary_dataset


def gower_matrix(
    X_categorical: np.ndarray,
    X_numerical: np.ndarray,
    Y_categorical: np.ndarray,
    Y_numerical: np.ndarray,
    numericals_ranges: np.ndarray,
    features_weight_sum: float,
    fill_diagonal: bool,
    first_index: int = -1,
):
    """
    _summary_

    Parameters
    ----------
    X_categorical : np.ndarray
        2D array containing only the categorical features of the X dataframe as uint8 values, shape (x_rows, cat_features).
    X_numerical : np.ndarray
        2D array containing only the numerical features of the X dataframe as float32 values, shape (x_rows, num_features).
    Y_categorical : np.ndarray
        2D array containing only the categorical features of the Y dataframe as uint8 values, shape (y_rows, cat_features).
    Y_numerical : np.ndarray
        2D array containing only the numerical features of the Y dataframe as float32 values, shape (y_rows, num_features).
    numericals_ranges : np.ndarray
        1D array containing the range (max-min) of each numerical feature as float32 values, shap (num_features,).
    features_weight_sum : float
        Sum of the feature weights used for the final average computation (usually it's just the number of features, each
        feature has a weigth of 1).
    fill_diagonal : bool
       Whether to fill the matrix diagonal values with a value larger than 1 (5.0). It must be True to get correct values
       if your computing the matrix just for one dataset (comparing a dataset with itself), otherwise you will get DCR==0
       for each row because on the diagonal you will compare a pair of identical instances.
    first_index : int, optioanl
        This is required only in case of parallel computation: the computation will occur batch by batch so ther original
        diagonal values will no longer be on the diagonal on each batch. We use this index to fill correctly the diagonal
        values. If -1 it's assumed there's no parallel computation, by default -1

    Returns
    -------
    np.ndarray
        1D array containing the Distance to the Closest Record for each row of x_dataframe shape (x_dataframe rows, )
    """
    return gower_matrix_c(
        X_categorical,
        X_numerical,
        Y_categorical,
        Y_numerical,
        numericals_ranges,
        features_weight_sum,
        fill_diagonal,
        first_index,
    )


def distances_to_closest_record(
    x_dataframe: pd.DataFrame,
    categorical_features: Union[List, np.ndarray],
    y_dataframe: pd.DataFrame = None,
    feature_weights: Union[List, np.ndarray] = None,
    parallel: bool = True,
):
    """
    Compute distance matrix between two dataframes containing mixed datatypes
    (numerical and categorical) using a modified version of the Gower's distance.

    Paper references:
    * A General Coefficient of Similarity and Some of Its Properties, J. C. Gower
    * Dimensionality Invariant Similarity Measure, Ahmad Basheer Hassanat

    Parameters
    ----------
    x_dataframe : pd.DataFrame
        A dataset containing numerical and categorical data.
    categorical_features : List
        List of booleans that indicates which features are categorical.
        If categoricals_features[i] is True, feature i is categorical.
        Must have same length of x_dataframe.columns.
    y_dataframe : pd.DataFrame, optional
        Another dataset containing numerical and categorical data, by default None.
        It must contains the same columns of x_dataframe.
        If None, the distance matrix is computed between x_dataframe and x_dataframe
    feature_weights : List, optional
        List of features weights to use computing distances, by default None.
        If None, each feature weight is 1.0
    parallel : Boolean, optional
        Whether to enable the parallelization to compute Gower matrix, by default True


    Returns
    -------
    np.ndarray
        1D array containing the Distance to the Closest Record for each row of x_dataframe
        shape (x_dataframe rows, )

    Raises
    ------
    TypeError
        If X and Y don't have the same (number of) columns.
    """
    X = x_dataframe
    # se c'è un secondo dataframe le distanze vengono calcolate con esso, altrimente X con sè stesso
    if y_dataframe is None:
        Y = x_dataframe
        fill_diagonal = True
    else:
        Y = y_dataframe
        fill_diagonal = False

    if not isinstance(X, np.ndarray):
        if not np.array_equal(X.columns, Y.columns):
            raise TypeError("X and Y dataframes have different columns.")
    else:
        if not X.shape[1] == Y.shape[1]:
            raise TypeError("X and Y arrays have different number of columns.")
    categorical_features = np.array(categorical_features)

    # Entrambi i dataframe vengono trasformati in array/matrice numpy
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    if feature_weights is None:
        # se non ho passato pesi specifici, tutti i pesi sono 1
        feature_weights = np.ones(X.shape[1])
    else:
        feature_weights = np.array(feature_weights)

    # La somma dei pesi è necessaria per fare la media alla fine (divisione)
    weight_sum = feature_weights.sum().astype("float32")
    # Matrice delle feature categoriche di X (num_rows_X x num_cat_feat)
    X_categorical = X[:, categorical_features].astype("uint8")
    # Matrice delle feature numeriche di X (num_rows_X x num_num_feat)
    X_numerical = X[:, np.logical_not(categorical_features)].astype("float32")
    # Matrice delle feature categoriche di Y (num_rows_Y x num_cat_feat)
    Y_categorical = Y[:, categorical_features].astype("uint8")
    # Matrice delle feature numeriche di Y (num_rows_Y x num_num_feat)
    Y_numerical = Y[:, np.logical_not(categorical_features)].astype("float32")
    # Il rango delle numeriche è necessario per il modo in cui sono calcolate le distanze Gower
    # Trovo il minimo e il massimo per ogni numerica concatenando X e Y quindi il rango sottraendo
    # tutti i minimi dai massimi.
    numericals_mins = np.amin(np.concatenate((X_numerical, Y_numerical)), axis=0)
    numericals_maxs = np.amax(np.concatenate((X_numerical, Y_numerical)), axis=0)
    numericals_ranges = numericals_maxs - numericals_mins

    X_rows = X_categorical.shape[0]

    """
    Parallel computation: I use all available CPUs except one, dividing the total number 
    of rows in X by the number of utilized CPUs to determine the chunk size for parallel processing. 
    The for loop with the index variable performs the actual computation, and the index is used to 
    correctly assign values for the distance between identical instances when calculating on a 
    single dataframe (when fill_diagonal == True).
    """
    if parallel:
        result_objs = []
        number_of_cpus = os.cpu_count() - 1
        chunk_size = int(X_rows / number_of_cpus)
        chunk_size = chunk_size if chunk_size > 0 else 1
        with Pool(processes=number_of_cpus) as pool:
            for index in range(0, X_rows, chunk_size):
                result = pool.apply_async(
                    gower_matrix,
                    (
                        X_categorical[index : index + chunk_size],
                        X_numerical[index : index + chunk_size],
                        Y_categorical,
                        Y_numerical,
                        numericals_ranges,
                        weight_sum,
                        fill_diagonal,
                        index,
                    ),
                )
                result_objs.append(result)
            results = [result.get() for result in result_objs]
        return np.concatenate(results)
    else:
        return gower_matrix(
            X_categorical,
            X_numerical,
            Y_categorical,
            Y_numerical,
            numericals_ranges,
            weight_sum,
            fill_diagonal,
        )


def dcr_stats(distances_to_closest_record: np.ndarray):
    """
    Return distribution stats for an array containing DCR computed previously.

    Parameters
    ----------
    distances_to_closest_record : np.ndarray
        A 1D-array containing the Distance to the Closest Record for each row of a dataframe
        shape (dataframe rows, )

    Returns
    -------
    Dict
        A dictionary containing mean and percentiles of the given DCR array.
    """
    dcr_mean = np.mean(distances_to_closest_record)
    dcr_percentiles = np.percentile(distances_to_closest_record, [0, 25, 50, 75, 100])
    return {
        "mean": dcr_mean.item(),
        "min": dcr_percentiles[0].item(),
        "25%": dcr_percentiles[1].item(),
        "median": dcr_percentiles[2].item(),
        "75%": dcr_percentiles[3].item(),
        "max": dcr_percentiles[4].item(),
    }


def number_of_dcr_equals_to_zero(distances_to_closest_record: np.ndarray):
    """
    Return the number of 0s in the given DCR array, that is the number of duplicates/clones detected.

    Parameters
    ----------
    distances_to_closest_record : np.ndarray
        A 1D-array containing the Distance to the Closest Record for each row of a dataframe
        shape (dataframe rows, )

    Returns
    -------
    int
        The number of 0s in the given DCR array.
    """
    zero_values_mask = distances_to_closest_record == 0.0
    return zero_values_mask.sum()


def dcr_histogram(
    distances_to_closest_record: np.ndarray,
    bins: Union[int, np.int64] = 20,
    scale_to_100: bool = True,
):
    """
    Compute the histogram of a DCR array: the DCR values equal to 0 are extracted before the
    histogram computation so that the first bar represent only the 0 (duplicates/clones)
    and the following bars represent the standard bins (with edge) of an histogram.

    Parameters
    ----------
    distances_to_closest_record : np.ndarray
        A 1D-array containing the Distance to the Closest Record for each row of a dataframe
        shape (dataframe rows, )
    bins : int, optional
        _description_, by default 20
    scale_to_100 : bool, optional
        Wheter to scale the histogram bins between 0 and 100 (instead of 0 and 1), by default True

    Returns
    -------
    Dict
        A dict containing the following items:
            * bins, histogram bins detected as string labels.
              The first bin/label is 0 (duplicates/clones), then the format is [inf_edge, sup_edge).
            * count, histogram values for each bin in bins
            * bins_edge_without_zero, the bin edges as returned by the np.histogram function without 0.
    """
    range_bins_with_zero = ["0.0"]
    number_of_dcr_zeros = number_of_dcr_equals_to_zero(distances_to_closest_record)
    dcr_non_zeros = distances_to_closest_record[distances_to_closest_record > 0]
    counts_without_zero, bins_without_zero = np.histogram(
        dcr_non_zeros, bins=bins, range=(0.0, 1.0), density=False
    )
    if scale_to_100:
        scaled_bins_without_zero = bins_without_zero * 100
    else:
        scaled_bins_without_zero = bins_without_zero
    range_bins_with_zero.append("(0.0-{:.2f})".format(scaled_bins_without_zero[1]))
    for i, left_edge in enumerate(scaled_bins_without_zero[1:-2]):
        range_bins_with_zero.append(
            "[{:.2f}-{:.2f})".format(left_edge, scaled_bins_without_zero[i + 2])
        )
    range_bins_with_zero.append(
        "[{:.2f}-{:.2f}]".format(
            scaled_bins_without_zero[-2], scaled_bins_without_zero[-1]
        )
    )
    counts_with_zero = np.insert(counts_without_zero, 0, number_of_dcr_zeros)
    return {
        "bins": range_bins_with_zero,
        "counts": counts_with_zero.tolist(),
        "bins_edge_without_zero": bins_without_zero.tolist(),
    }


def validation_dcr_test(dcr_synth_train: np.ndarray, dcr_synth_validation: np.ndarray):
    """
    _summary_

    Parameters
    ----------
    dcr_synth_train : np.ndarray
        A 1D-array containing the Distance to the Closest Record for each row of the synthetic
        dataset wrt the training dataset, shape (synthetic rows, )
    dcr_synth_validation : np.ndarray
        A 1D-array containing the Distance to the Closest Record for each row of the synthetic
        dataset wrt the validation dataset, shape (synthetic rows, )

    Returns
    -------
    float
        The percentage of synthetic rows closer to the training dataset than to the validation dataset.

    Raises
    ------
    ValueError
        If the two DCR array given as parameters have different shapes.
    """
    if dcr_synth_train.shape != dcr_synth_validation.shape:
        raise ValueError("Dcr arrays have different shapes.")

    warnings = ""
    percentage = 0.0

    if dcr_synth_train.sum() == 0:
        percentage = 100.0
        warnings = (
            "The synthetic dataset is an exact copy/clone of the training dataset."
        )
    elif (dcr_synth_train == dcr_synth_validation).all():
        percentage = 0.0
        warnings = (
            "The validation dataset is an exact copy/clone of the training dataset."
        )
    else:
        if dcr_synth_validation.sum() == 0:
            warnings = "The synthetic dataset is an exact copy/clone of the validation dataset."

        number_of_rows = dcr_synth_train.shape[0]
        synth_dcr_smaller_than_holdout_dcr_mask = dcr_synth_train < dcr_synth_validation
        synth_dcr_smaller_than_holdout_dcr_sum = (
            synth_dcr_smaller_than_holdout_dcr_mask.sum()
        )
        percentage = synth_dcr_smaller_than_holdout_dcr_sum / number_of_rows * 100

    return {"percentage": percentage, "warnings": warnings}


def training_metrics(
    processed_training_dataset: pd.DataFrame,
    categorical_features: Union[List, np.ndarray],
    full_analysis: bool = False,
    parallel: bool = True,
):
    """
    Computes metrics for the training dataset to evaluate data quality, including
    duplicate counts and, if requested, Distance to the Closest Record (DCR) analysis.

    Parameters
    ----------
    processed_training_dataset : pd.DataFrame
        The training dataset that has been processed and standardized.
    categorical_features : Union[List, np.ndarray]
        A list or array indicating which features are categorical. If `categorical_features[i]`
        is True, the i-th feature is categorical.
    full_analysis : bool, optional
        If True, performs a comprehensive analysis by calculating the DCR values and
        generating related statistics and histograms, by default False.
    parallel : bool, optional
        If True, enables parallel computation to speed up distance calculations, by default True.

    Returns
    -------
    dict
        A dictionary containing metrics for the training dataset:
        - "training_duplicates": The number of duplicate rows in the training dataset.
        - "training_duplicates_percentage": The percentage of duplicate rows in the training dataset.
        - "training_unique_duplicates": The number of unique duplicate rows in the training dataset.
        - "training_unique_duplicates_percentage": The percentage of unique duplicate rows in the training dataset.
        - "dcr_train_train_stats" (optional): Statistical summary of the DCR values if `full_analysis` is True.
        - "dcr_train_train_hist" (optional): Histogram of the DCR values if `full_analysis` is True.

    Notes
    -----
    The function analyzes the training dataset by calculating duplicates and, if requested,
    performing DCR analysis to measure the similarity between records.
    """
    if full_analysis:
        dcr_train_train = distances_to_closest_record(
            processed_training_dataset,
            categorical_features,
            parallel=parallel,
        )
        dcr_train_train_stats = dcr_stats(dcr_train_train)

        number_of_bins = 100
        dcr_train_train_hist = dcr_histogram(dcr_train_train, bins=number_of_bins)

    training_rows = processed_training_dataset.shape[0]
    training_duplicates = processed_training_dataset.duplicated(keep=False).sum().item()
    training_unique_duplicates = processed_training_dataset.duplicated().sum().item()

    training_duplicates_percentage = training_duplicates / training_rows * 100
    training_unique_duplicates_percentage = (
        training_unique_duplicates / training_rows * 100
    )

    training_metrics_dict = {
        "training_duplicates": training_duplicates,
        "training_duplicates_percentage": training_duplicates_percentage,
        "training_unique_duplicates": training_unique_duplicates,
        "training_unique_duplicates_percentage": training_unique_duplicates_percentage,
    }

    if full_analysis:
        training_metrics_dict["dcr_train_train_stats"] = dcr_train_train_stats
        training_metrics_dict["dcr_train_train_hist"] = dcr_train_train_hist

    return training_metrics_dict


def synthetic_metrics(
    processed_synthetic_dataset: pd.DataFrame,
    categorical_features: Union[List, np.ndarray],
    full_analysis: bool = False,
    parallel: bool = True,
):
    """
    Computes various metrics to evaluate the quality and uniqueness of a synthetic dataset,
    including duplicate counts and Distance to the Closest Record (DCR) statistics.

    Parameters
    ----------
    processed_synthetic_dataset : pd.DataFrame
        The synthetic dataset that has been processed and standardized.
    categorical_features : Union[List, np.ndarray]
        A list or array indicating which features are categorical. If `categorical_features[i]`
        is True, the i-th feature is categorical.
    full_analysis : bool, optional
        If True, performs a comprehensive analysis by calculating the DCR values and
        generating related statistics and histograms, by default False.
    parallel : bool, optional
        If True, enables parallel computation to speed up distance calculations, by default True.

    Returns
    -------
    dict
        A dictionary containing metrics for the synthetic dataset:
        - "synthetic_duplicates": The number of duplicate rows in the synthetic dataset.
        - "synthetic_duplicates_percentage": The percentage of duplicate rows in the synthetic dataset.
        - "synthetic_unique_duplicates": The number of unique duplicate rows in the synthetic dataset.
        - "synthetic_unique_duplicates_percentage": The percentage of unique duplicate rows in the synthetic dataset.
        - "dcr_synth_synth_stats" (optional): Statistical summary of the DCR values if `full_analysis` is True.
        - "dcr_synth_synth_hist" (optional): Histogram of the DCR values if `full_analysis` is True.

    Notes
    -----
    The function evaluates the synthetic dataset's quality by checking for duplicates and,
    if requested, computing DCR statistics. DCR measures the similarity between records
    within the synthetic dataset.
    """
    if full_analysis:
        dcr_synth_synth = distances_to_closest_record(
            processed_synthetic_dataset,
            categorical_features,
            parallel=parallel,
        )
        dcr_synth_synth_stats = dcr_stats(dcr_synth_synth)

        number_of_bins = 100
        dcr_synth_synth_hist = dcr_histogram(dcr_synth_synth, bins=number_of_bins)

    synthetic_rows = processed_synthetic_dataset.shape[0]
    synthetic_duplicates = (
        processed_synthetic_dataset.duplicated(keep=False).sum().item()
    )
    synthetic_unique_duplicates = processed_synthetic_dataset.duplicated().sum().item()

    synthetic_duplicates_percentage = synthetic_duplicates / synthetic_rows * 100
    synthetic_unique_duplicates_percentage = (
        synthetic_unique_duplicates / synthetic_rows * 100
    )

    synthetic_metrics_dict = {
        "synthetic_duplicates": synthetic_duplicates,
        "synthetic_duplicates_percentage": synthetic_duplicates_percentage,
        "synthetic_unique_duplicates": synthetic_unique_duplicates,
        "synthetic_unique_duplicates_percentage": synthetic_unique_duplicates_percentage,
    }

    if full_analysis:
        synthetic_metrics_dict["dcr_synth_synth_stats"] = dcr_synth_synth_stats
        synthetic_metrics_dict["dcr_synth_synth_hist"] = dcr_synth_synth_hist

    return synthetic_metrics_dict


def synthetic_training_metrics(
    processed_synthetic_dataset: pd.DataFrame,
    processed_training_dataset: pd.DataFrame,
    categorical_features: Union[List, np.ndarray],
    training_hist_bins: np.ndarray = None,
    parallel: bool = True,
):
    """
    Computes privacy metrics to assess the similarity between a synthetic dataset and
    a training dataset using Distance to the Closest Record (DCR).

    Parameters
    ----------
    processed_synthetic_dataset : pd.DataFrame
        The synthetic dataset that has been processed and standardized.
    processed_training_dataset : pd.DataFrame
        The training dataset that has been processed and standardized.
    categorical_features : Union[List, np.ndarray]
        A list or array indicating which features are categorical. If `categorical_features[i]` 
        is True, the i-th feature is categorical.
    training_hist_bins : np.ndarray, optional
        Precomputed histogram bin edges from the training DCR distribution. If None, they will 
        be generated, by default None.
    parallel : bool, optional
        If True, enables parallel computation to speed up distance calculations, by default True.

    Returns
    -------
    tuple
        A tuple containing:
        - dcr_synth_train : np.ndarray
            The Distance to the Closest Record (DCR) between the synthetic and training datasets.
        - synth_train_metrics_dict : dict
            A dictionary containing:
            - "synth_train_clones": The number of synthetic records that are identical to training records.
            - "synth_train_clones_percentage": The percentage of synthetic records that are clones of training records.
            - "dcr_synth_train_stats": Statistical summary of the DCR values.
            - "dcr_synth_train_hist": Histogram of the DCR values.

    Notes
    -----
    DCR is used to measure the similarity between datasets by calculating the distance
    between each synthetic record and its closest record in the training dataset.
    The function also provides a histogram of DCR values and the percentage of records
    that are identical (clones).
    """
    dcr_synth_train = distances_to_closest_record(
        processed_synthetic_dataset,
        categorical_features,
        processed_training_dataset,
        parallel=parallel,
    )

    dcr_synth_train_stats = dcr_stats(dcr_synth_train)

    if training_hist_bins is not None:
        dcr_synth_train_hist = dcr_histogram(dcr_synth_train, bins=training_hist_bins)
    else:
        number_of_bins = 100
        dcr_synth_train_hist = dcr_histogram(dcr_synth_train, bins=number_of_bins)

    synth_train_clones = dcr_synth_train_hist["counts"][0]

    synthetic_rows = processed_synthetic_dataset.shape[0]
    synth_train_clones_percentage = synth_train_clones / synthetic_rows * 100

    synth_train_metrics_dict = {
        "synth_train_clones": synth_train_clones,
        "synth_train_clones_percentage": synth_train_clones_percentage,
        "dcr_synth_train_stats": dcr_synth_train_stats,
        "dcr_synth_train_hist": dcr_synth_train_hist,
    }
    return dcr_synth_train, synth_train_metrics_dict


def synthetic_holdout_metrics(
    processed_synthetic_dataset: pd.DataFrame,
    processed_holdout_dataset: pd.DataFrame,
    categorical_features: Union[List, np.ndarray],
    dcr_synth_train: np.ndarray = None,
    processed_training_dataset: pd.DataFrame = None,
    training_hist_bins: np.ndarray = None,
    parallel: bool = True,
    test_sampling: bool = True,
):
    """
    Computes privacy-related metrics by analyzing the distances between a synthetic dataset
    and a holdout dataset. Optionally compares distances to a training dataset to test 
    whether synthetic data is more similar to training data than to holdout data.

    Parameters
    ----------
    processed_synthetic_dataset : pd.DataFrame
        The synthetic dataset that has been processed and standardized.
    processed_holdout_dataset : pd.DataFrame
        The holdout dataset that has been processed and standardized.
    categorical_features : Union[List, np.ndarray]
        A list or array indicating which features are categorical. If `categorical_features[i]` 
        is True, the i-th feature is categorical.
    dcr_synth_train : np.ndarray, optional
        Precomputed Distance to the Closest Record (DCR) between the synthetic and training 
        datasets. If None, it will be calculated, by default None.
    processed_training_dataset : pd.DataFrame, optional
        The training dataset that has been processed and standardized. Needed if test sampling 
        is enabled, by default None.
    training_hist_bins : np.ndarray, optional
        Precomputed histogram bin edges from the training DCR distribution. If None, they will 
        be generated, by default None.
    parallel : bool, optional
        If True, enables parallel computation to speed up distance calculations, by default True.
    test_sampling : bool, optional
        If True, uses sampling for the DCR computation to reduce computational cost, by default True.

    Returns
    -------
    dict
        A dictionary containing the following metrics:
        - "dcr_synth_holdout_stats": Statistics (mean, min, max, etc.) of DCR between synthetic 
          and holdout datasets.
        - "dcr_synth_holdout_hist": Histogram of DCR values between synthetic and holdout datasets.
        - "synth_holdout_test": Percentage of synthetic data that is closer to the training data 
          than to the holdout data.
        - "synth_holdout_test_warnings": Warnings related to the DCR comparison test.
        - "dcr_synth_train_stats": (Optional) Statistics of DCR between synthetic and training 
          datasets if test_sampling is enabled.
        - "dcr_synth_train_hist": (Optional) Histogram of DCR values between synthetic and training 
          datasets if test_sampling is enabled.

    Notes
    -----
    The function uses DCR (Distance to the Closest Record) to measure the similarity between 
    datasets. It can optionally perform sampling to reduce the computational cost when datasets 
    are large. The validation DCR test checks if synthetic data is closer to the training set 
    compared to the holdout set, which helps assess the risk of overfitting or data leakage.
    """
    if test_sampling and processed_training_dataset is not None:
        (
            sampled_training,
            sampled_holdout,
            sampled_synthetic,
        ) = sample_datasets_for_dcr_test(
            processed_training_dataset,
            processed_holdout_dataset,
            processed_synthetic_dataset,
            max_sample_size=10000,
        )

        dcr_synth_train = distances_to_closest_record(
            sampled_synthetic,
            categorical_features,
            sampled_training,
            parallel=parallel,
        )
        dcr_synth_train_stats = dcr_stats(dcr_synth_train)
        if training_hist_bins is not None:
            dcr_synth_train_hist = dcr_histogram(
                dcr_synth_train, bins=training_hist_bins
            )
        else:
            number_of_bins = 100
            dcr_synth_train_hist = dcr_histogram(dcr_synth_train, bins=number_of_bins)

        dcr_synth_holdout = distances_to_closest_record(
            sampled_synthetic,
            categorical_features,
            sampled_holdout,
            parallel=parallel,
        )
    else:
        dcr_synth_holdout = distances_to_closest_record(
            processed_synthetic_dataset,
            categorical_features,
            processed_holdout_dataset,
            parallel=parallel,
        )

    dcr_synth_holdout_stats = dcr_stats(dcr_synth_holdout)
    if training_hist_bins is not None:
        dcr_synth_holdout_hist = dcr_histogram(
            dcr_synth_holdout, bins=training_hist_bins
        )
    else:
        number_of_bins = 100
        dcr_synth_holdout_hist = dcr_histogram(dcr_synth_holdout, bins=number_of_bins)

    synth_holdout_test = validation_dcr_test(dcr_synth_train, dcr_synth_holdout)

    synth_holdout_metrics = {
        "dcr_synth_holdout_stats": dcr_synth_holdout_stats,
        "dcr_synth_holdout_hist": dcr_synth_holdout_hist,
        "synth_holdout_test": synth_holdout_test["percentage"].item(),
        "synth_holdout_test_warnings": synth_holdout_test["warnings"],
    }

    if test_sampling:
        synth_holdout_metrics["dcr_synth_train_stats"] = dcr_synth_train_stats
        synth_holdout_metrics["dcr_synth_train_hist"] = dcr_synth_train_hist

    return synth_holdout_metrics


def membership_inference_test(
    processed_adversary_dataset: pd.DataFrame,
    processed_synthetic_dataset: pd.DataFrame,
    categorical_features: Union[List, np.ndarray],
    adversary_guesses_ground_truth: np.ndarray,
    parallel: bool = True,
):
    """
    Conducts a Membership Inference Test to estimate the risk of revealing membership
    information based on the distance between the adversary dataset and the synthetic dataset.

    The function calculates the Distance to the Closest Record (DCR) between the adversary 
    dataset and the synthetic dataset, applies various distance thresholds, and computes 
    precision scores to evaluate the risk of membership inference.

    Parameters
    ----------
    processed_adversary_dataset : pd.DataFrame
        The processed adversary dataset containing mixed-type data.
    processed_synthetic_dataset : pd.DataFrame
        The processed synthetic dataset containing mixed-type data.
    categorical_features : Union[List, np.ndarray]
        A list or array of booleans indicating which features are categorical. If 
        `categorical_features[i]` is True, the i-th feature is categorical.
    adversary_guesses_ground_truth : np.ndarray
        Ground truth array indicating the true membership status of each instance in 
        the adversary dataset.
    parallel : bool, optional
        If True, enables parallel computation to speed up the DCR calculation, 
        by default True.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "adversary_distance_thresholds": List of distance thresholds used for 
          precision calculations.
        - "adversary_precisions": List of precision scores for each distance threshold.
        - "membership_inference_mean_risk_score": A float value representing the 
          mean risk score, ranging from 0 to 1, with higher values indicating 
          higher risk.

    Notes
    -----
    The `membership_inference_mean_risk_score` is calculated as twice the difference 
    between the mean adversary precision and 0.5, clipped to a minimum of 0. This score 
    provides an estimate of how well an adversary can differentiate between members and 
    non-members of the training data based on the synthetic data.
    """
    dcr_adversary_synth = distances_to_closest_record(
        processed_adversary_dataset,
        categorical_features,
        processed_synthetic_dataset,
        parallel=parallel,
    )
    adversary_precisions = []
    distance_thresholds = np.quantile(
        dcr_adversary_synth, [0.5, 0.25, 0.2, np.min(dcr_adversary_synth) + 0.01]
    )
    for distance_threshold in distance_thresholds:
        adversary_guesses = dcr_adversary_synth < distance_threshold
        adversary_precision = precision_score(
            adversary_guesses_ground_truth, adversary_guesses, zero_division=0
        )
        adversary_precisions.append(max(adversary_precision, 0.5))
    adversary_precision_mean = np.mean(adversary_precisions).item()
    membership_inference_mean_risk_score = max(
        (adversary_precision_mean - 0.5) * 2, 0.0
    )

    return {
        "adversary_distance_thresholds": distance_thresholds.tolist(),
        "adversary_precisions": adversary_precisions,
        "membership_inference_mean_risk_score": membership_inference_mean_risk_score,
    }


class PrivacyScore:
    original_dataset: pd.DataFrame
    synthetic_dataset: pd.DataFrame
    holdout_dataset: pd.DataFrame
    preprocessor: Preprocessor
    categorical_features: Union[List, np.ndarray]
    parallel: bool
    dcr_test_sampling: bool
    """
    Class to compute privacy metrics for original and synthetic datasets.

    Attributes:
        original_dataset (pd.DataFrame): Original dataset.
        synthetic_dataset (pd.DataFrame): Synthetic dataset.
        holdout_dataset (pd.DataFrame): Holdout dataset (optional).
        preprocessor (Preprocessor): Preprocessor instance.
        categorical_features (Union[List, np.ndarray]): List of categorical feature indicators.
        parallel (bool): Whether to use parallel computation.
        dcr_test_sampling (bool): Whether to use sampling for DCR tests.
    """
    def __init__(
        self,
        original_dataset: Dataset,
        synthetic_dataset: Dataset,
        holdout_dataset: Dataset = None,
        preprocessor: Preprocessor = None,
        max_sample_size: int = 70000,
        parallel: bool = True,
        dcr_test_sampling: bool = True,
    ):
        """
        Initializes the PrivacyScore class with datasets and configuration.

        Args:
            original_dataset (Dataset): Original dataset.
            synthetic_dataset (Dataset): Synthetic dataset.
            holdout_dataset (Dataset, optional): Holdout dataset. Defaults to None.
            preprocessor (Preprocessor, optional): Preprocessor instance. Defaults to None.
            max_sample_size (int, optional): Maximum sample size for analysis. Defaults to 70000.
            parallel (bool, optional): Enable parallel computation. Defaults to True.
            dcr_test_sampling (bool, optional): Enable sampling for DCR tests. Defaults to True.
        """
        self.max_sample_size = max_sample_size
        self.parallel = parallel
        self.dcr_test_sampling = dcr_test_sampling
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor(original_dataset)
        )

        categorical_features = {}
        discarded = [discarded for discarded in self.preprocessor.discarded[0]]
        for feature in original_dataset.columns():
            if feature not in discarded+list(self.preprocessor.rules.keys()):
                if feature in self.preprocessor.get_categorical_features():
                    categorical_features[feature] = True
                else:
                    categorical_features[feature] = False
        if (
            original_dataset.target_column is not None
            and original_dataset.regression is False
        ):
            categorical_features[original_dataset.target_column] = True
        self.categorical_features = list(categorical_features.values())

        original_df = original_dataset.data.copy()
        preprocessed_original_df = self.preprocessor.transform(original_df)
        original_df = self.preprocessor.reverse_transform(
            preprocessed_original_df
        ).fillna(0)
        if original_dataset.target_column is not None:
            original_df[original_dataset.target_column] = original_dataset.data[
                original_dataset.target_column
            ]
        original_df = original_df.reindex(
            columns=[col for col in original_dataset.columns() if col not in discarded+list(self.preprocessor.rules.keys())]
        )

        synthetic_df = synthetic_dataset.data.copy()
        preprocessed_synthetic_df = self.preprocessor.transform(synthetic_df)
        synthetic_df = self.preprocessor.reverse_transform(
            preprocessed_synthetic_df
        ).fillna(0)
        if original_dataset.target_column is not None:
            synthetic_df[original_dataset.target_column] = synthetic_dataset.data[
                original_dataset.target_column
            ]
        synthetic_df = synthetic_df.reindex(
            columns=[col for col in original_dataset.columns() if col not in discarded+list(self.preprocessor.rules.keys())]
        )

        if holdout_dataset is not None:
            holdout_df = holdout_dataset.data.copy()
            preprocessed_holdout_df = self.preprocessor.transform(holdout_df)
            holdout_df = self.preprocessor.reverse_transform(
                preprocessed_holdout_df
            ).fillna(0)
            if original_dataset.target_column is not None:
                holdout_df[original_dataset.target_column] = holdout_dataset.data[
                    original_dataset.target_column
                ]
            holdout_df = holdout_df.reindex(
                columns=[
                    col for col in original_dataset.columns() if col not in discarded+list(self.preprocessor.rules.keys())
                ]
            )

            (
                self.original_dataset,
                self.synthetic_dataset,
                self.holdout_dataset,
            ) = preprocess_datasets(
                original_df, synthetic_df, self.categorical_features, holdout_df
            )
        else:
            self.original_dataset, self.synthetic_dataset = preprocess_datasets(
                original_df, synthetic_df, self.categorical_features
            )
            self.holdout_dataset = None

    def get(self, verbose: bool = False):
        """
        Computes and returns privacy metrics for the datasets.

        Args:
            verbose (bool, optional): Whether to log detailed progress. Defaults to False.

        Returns:
            dict: Privacy metrics for the original and synthetic datasets.
        """
        sampled_training, sampled_validation, sampled_synthetic = sample_datasets(
            self.original_dataset,
            self.holdout_dataset,
            self.synthetic_dataset,
            self.max_sample_size,
        )

        metrics_dict = {}

        if self.original_dataset.shape[0] > self.max_sample_size:
            full_analysis = False
        else:
            full_analysis = True

        if verbose:
            logger.info(
                f"Computing training privacy metrics for Training Set {self.original_dataset.shape[0]}x{self.original_dataset.shape[1]}, sampling: {sampled_training.shape[0]}x{sampled_training.shape[1]}, fullAnalysis: {full_analysis}, parallel: {self.parallel}"
            )
        training_metrics_dict = training_metrics(
            sampled_training,
            self.categorical_features,
            full_analysis=full_analysis,
            parallel=self.parallel,
        )

        if verbose:
            logger.info(
                f"Computing synthetic privacy metrics for Synthetic Set {self.synthetic_dataset.shape[0]}x{self.synthetic_dataset.shape[1]}, sampling: {sampled_synthetic.shape[0]}x{sampled_synthetic.shape[1]}, parallel: {self.parallel}"
            )
        synthetic_metrics_dict = synthetic_metrics(
            sampled_synthetic, self.categorical_features, parallel=self.parallel
        )

        if verbose:
            logger.info(
                f"Computing synthetic-training privacy metrics, sampledSynthetic {sampled_synthetic.shape[0]}x{sampled_synthetic.shape[1]}, sampledTraining {sampled_training.shape[0]}x{sampled_training.shape[1]}, parallel: {self.parallel}"
            )
        dcr_synth_train, synthetic_training_metrics_dict = synthetic_training_metrics(
            sampled_synthetic,
            sampled_training,
            self.categorical_features,
            parallel=self.parallel,
        )

        metrics_dict = {
            "training_metrics": training_metrics_dict,
            "synthetic_metrics": synthetic_metrics_dict,
            "synthetic_training_metrics": synthetic_training_metrics_dict,
        }

        if self.holdout_dataset is not None:
            if verbose:
                logger.info(
                    f"Computing synthetic-holdout privacy metrics, sampledSynthetic {sampled_synthetic.shape[0]}x{sampled_synthetic.shape[1]}, sampledHoldout {sampled_validation.shape[0]}x{sampled_validation.shape[1]}, parallel: {self.parallel}, testSampling: {self.dcr_test_sampling}"
                )
            synthetic_holdout_metrics_dict = synthetic_holdout_metrics(
                sampled_synthetic,
                sampled_validation,
                self.categorical_features,
                dcr_synth_train=dcr_synth_train,
                processed_training_dataset=sampled_training,
                parallel=self.parallel,
                test_sampling=self.dcr_test_sampling,
            )

            adv_dataset = adversary_dataset(sampled_training, sampled_validation, 0.2)
            adversary_guesses_ground_truth = adv_dataset[
                "privacy_test_is_training"
            ].to_numpy()

            if verbose:
                logger.info(
                    f"Computing Membership Inference Test, with adversarySet {adv_dataset.shape[0]}x{adv_dataset.shape[1]}, parallel: {self.parallel}"
                )
            membership_inference_test_dict = membership_inference_test(
                adv_dataset.drop(columns="privacy_test_is_training"),
                sampled_synthetic,
                self.categorical_features,
                adversary_guesses_ground_truth,
                parallel=self.parallel,
            )
            metrics_dict["synthetic_holdout_metrics"] = synthetic_holdout_metrics_dict
            metrics_dict["membership_inference_test"] = membership_inference_test_dict

            return metrics_dict