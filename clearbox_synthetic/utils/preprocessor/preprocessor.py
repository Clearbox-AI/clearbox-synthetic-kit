"""
This module provides the Preprocessor class and related functions for preprocessing tabular data. It includes 
functionality for encoding categorical variables, handling missing values, and feature selection.

Functions:
    - process_categorical: Preprocesses categorical and numerical features for embedding.
    - jaccard_similarity: Computes Jaccard similarity between two strings.
    - attach_series: Generates a series based on Jaccard similarity and a dependency dictionary.
    - create_series: Creates a pandas Series using embedded columns from the dataset.

Classes:
    - Preprocessor: A class for transforming tabular data, handling categorical and numerical feature encoding, 
      and feature selection.
"""

import sklearn.pipeline
import sklearn.compose
import sklearn.preprocessing
import sklearn.impute
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Callable, Tuple
from loguru import logger
import copy
from jax import jit

from clearbox_synthetic.utils import Dataset
from ..transformers import (
    NumericalTransformer,
    CategoricalTransformer,
    DatetimeTransformer,
    )

def process_categorical(df: pd.DataFrame, input_feats: List, target: List) -> Tuple[Dict, Dict, pd.Index]:
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

def jaccard_similarity(str1: str, str2: str) -> float:
    """
    Computes the Jaccard similarity between two strings.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        float: Jaccard similarity score between the two strings.
    """
    set1 = set(str1)
    set2 = set(str2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def attach_series(dict_dep: Dict, list_str: np.array, keys: List) -> List:
    """
    Generates a series of values by comparing strings with Jaccard similarity.

    Args:
        dict_dep (Dict): Dictionary of dependencies for mapping values.
        list_str (np.array): Array of strings used for comparison.
        keys (List): List of keys to match.

    Returns:
        List: Generated series of values.
    """
    rows = []
    for i in tqdm(range(len(keys))):
        k = keys[i]
        try:
            choice = list(dict_dep[k].keys())
            p = dict_dep[k].values
            p = p / np.sum(p)
            rows.append(np.random.choice(choice, p=p))
        except KeyError:
            w = np.argmax([jaccard_similarity(k, item) for item in list_str])
            choice = list(dict_dep[list_str[w]].keys())
            p = dict_dep[list_str[w]].values
            p = p / np.sum(p)
            if len(choice) > 0:
                rows.append(np.random.choice(choice, p=p))
            else:
                rows.append(np.nan)
                print(list_str[w])
    return rows

def create_series(dict_dep: Dict, bins_dep: Dict, columns: List, df: pd.DataFrame) -> pd.Series:
    """
    Creates a pandas Series using embedded columns and dictionary dependencies.

    Args:
        dict_dep (Dict): Dictionary of dependencies for mapping values.
        bins_dep (Dict): Dictionary of binning information for numerical features.
        columns (List): List of columns for processing.
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: Generated series based on the dependencies.
    """
    list_str = np.random.choice(list(dict_dep.keys()), min(len(list(dict_dep.keys())), 1000))
    input_cat = [i for i in columns[:-1] if df[i].dtype == 'object' or df[i].nunique() < 4]
    
    x = copy.deepcopy(df)
    for i in input_cat:
        x[i] = x[i].fillna('NaN')
    for i in bins_dep.keys():
        x[i] = pd.cut(x[i], bins=bins_dep[i])
    
    keys = list(x[columns[:-1]].apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist())
    rows = attach_series(dict_dep, list_str, keys)
    
    return pd.Series(rows)

class Preprocessor:
    """
    Preprocessor class for handling and transforming tabular data.

    Attributes:
        transformer (ColumnTransformer): Sklearn ColumnTransformer for data preprocessing.
        not_fitted_transformer (ColumnTransformer): Unfitted version of the transformer.
        inverse_transformer (Callable): Function for reversing the transformations.
        discarded (Tuple): Tuple of columns discarded during feature selection.
        sorted_columns (List): List of sorted column names for transformation.
        numerical_features (List): List of ordinal feature names.
        categorical_features (List): List of categorical feature names.
        datetime_features (List): List of datetime feature names.
        emb_rules (Dict): Embedding rules for categorical features.
    """

    def __init__(
        self,
        dataset: Dataset,
        threshold: float = 0.02,
        n_numerical_bins: int = 0,
        num_transformer_type: str = "Quantile",
        na_fill_value: float = -0.001,
        rules: Dict = {},
        time_index = None,
        meta_columns: List = [],
    ):
        """
        Initializes the Preprocessor class.

        Args:
            dataset (Dataset): The input dataset.
            threshold (float, optional): Threshold for feature selection. Defaults to 0.02.
            n_numerical_bins (int, optional): Number of bins for ordinal encoding. Defaults to 0.
            num_transformer_type (str, optional): Type of numerical transformer. Defaults to "Quantile".
            na_fill_value (float, optional): Value to fill missing values. Defaults to -0.001.
            rules (Dict, optional): Rules for embedding and transformations. Defaults to {}.
        """
        X = dataset.get_x().copy()
        self.time_index = None
        self.meta_columns = meta_columns
        self.time_columns = []
        self.time_mean = None
        self.time_std = None
        self.n_time_features = None
        self.max_sequence_length = None
        if time_index is not None:
            self.time_index = time_index
            self.time_columns = [i for i in dataset.data.columns if i not in [time_index]+meta_columns]
            self.time_mean= dataset.data[self.time_columns].mean(axis=0).values
            self.time_std= dataset.data[self.time_columns].std(axis=0).values
            self.n_time_features = len(self.time_columns)
            n =  X[[self.time_index,self.time_columns[0]]].groupby(self.time_index).count().max().values[0]
            self.max_sequence_length = round(n / 32) * 32
            X = X[meta_columns]
            
        self.rules = rules
        self.emb_rules = {}
        for w in [i for i in rules.keys() if 'embed_category' in rules[i][0]]:
            self.emb_rules[w] = process_categorical(X, rules[w][1], rules[w][2])
            X = X.drop(w, axis=1)

        for w in [i for i in rules.keys() if 'sum' in rules[i][0]]:
            X = X.drop(w, axis=1)
            
        dataset_infer = Dataset(data=X.sample(min(X.shape[0], 1000), random_state=42))
        self.numerical_features, self.categorical_features, self.datetime_features = self._infer_feature_types(dataset_infer)
            
        X = self._feature_selection(X, self.categorical_features, self.numerical_features, threshold)
        X = X.sample(n=min(X.shape[0], int(1e4)), random_state=42)

        discarded = [discarded for discarded in self.discarded[0]]
        self.sorted_columns = [col for col in X.columns.values if col not in discarded]

        self.numerical_features = [i for i in self.numerical_features if i not in self.discarded[0]]
        self.categorical_features = [i for i in self.categorical_features if i not in self.discarded[0]]

        transformers_list = []
        if len(self.numerical_features) > 0:
            transformers_list.append(
                ("numerical_transformer", NumericalTransformer(n_numerical_bins, num_transformer_type, na_fill_value), self.numerical_features)
            )
        if len(self.datetime_features) > 0:
            transformers_list.append(
                ("datetime_transformer", DatetimeTransformer(), self.datetime_features)
            )
        if len(self.categorical_features) > 0:
            transformers_list.append(
                ("categorical_transformer", CategoricalTransformer(), self.categorical_features)
            )

        column_transformer = sklearn.compose.ColumnTransformer(transformers=transformers_list)
        self.transformer = column_transformer.fit(X)
        self.not_fitted_transformer = column_transformer
        self.inverse_transformer = self.inverse_preprocessor

    @staticmethod
    def _infer_feature_types(dataset: Dataset) -> Tuple[List[str], List[str], List[str]]:
        """
        Infers feature types (ordinal, categorical, datetime) from the dataset.

        Args:
            dataset (Dataset): The input dataset.

        Returns:
            Tuple: Lists of ordinal, categorical, and datetime feature names.
        """
        if dataset.column_types:
            numerical_features = [
                column for column in dataset.column_types.keys()
                if (dataset.column_types[column] in ["number", "boolean"])
                and column != dataset.target_column
                and column != dataset.sequence_index
                and column != dataset.group_by
            ]
            categorical_features = [
                column for column in dataset.column_types.keys()
                if dataset.column_types[column] == "string"
                and column != dataset.target_column
                and column != dataset.sequence_index
                and column != dataset.group_by
            ]
            datetime_features = [
                column for column in dataset.column_types.keys()
                if dataset.column_types[column] == "datetime"
                and column != dataset.target_column
                and column != dataset.sequence_index
                and column != dataset.group_by
            ]
        else:
            bool_features = dataset.x_columns(include=["bool"])
            dataset.data[bool_features] = dataset.data[bool_features].astype("category")

            datetime_features = dataset.x_columns(include=["datetime", "timedelta"])
            dataset.data[datetime_features] = dataset.data[datetime_features].astype("int64")
            datetime_features = []

            numerical_features = dataset.x_columns(include=["number", "datetime"])
            categorical_features = dataset.x_columns(include=["object", "category"])

        return numerical_features, categorical_features, datetime_features

    @staticmethod
    def _shrink_labels(instance: pd.DataFrame, too_much_info: dict) -> pd.DataFrame:
        """
        Shrinks labels in the dataset based on provided information.

        Args:
            instance (pd.DataFrame): The DataFrame instance to modify.
            too_much_info (dict): Dictionary of labels to be replaced.

        Returns:
            pd.DataFrame: Modified DataFrame with labels shrunk.
        """
        for column_name in too_much_info:
            if instance[column_name].dtype == "object":
                instance[column_name].replace(too_much_info[column_name], "*", inplace=True)
            else:
                instance[column_name].replace(too_much_info[column_name], -999999, inplace=True)
        return instance

    def _feature_selection(
        self,
        X: pd.DataFrame,
        categorical_features: List[str],
        numerical_features: List[str],
        threshold: float,
    ) -> pd.DataFrame:
        """
        Selects the most informative features from the DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame.
            categorical_features (List[str]): List of categorical feature names.
            numerical_features (List[str]): List of ordinal feature names.
            threshold (float): Threshold for feature selection.

        Returns:
            pd.DataFrame: DataFrame with selected features.
        """
        cat_features_stats = [
            (i, X[i].value_counts(), X[i].nunique(), X.columns.get_loc(i))
            for i in categorical_features
        ]

        ord_features_stats = [
            (i, X[i].value_counts(), X[i].unique(), X.columns.get_loc(i))
            for i in numerical_features
        ]

        no_info = []
        too_much_info = {}
        for column_stats in cat_features_stats:
            if (column_stats[1].shape[0] == 1) or (column_stats[1].shape[0] >= (X.shape[0] * 0.98)):
                no_info.append(column_stats[0])
            else:
                counts = column_stats[1].values / column_stats[1].values.sum()
                values_to_shrink_indices = np.where(counts < threshold)[0]
                if values_to_shrink_indices.shape[0] > 0 and column_stats[1].shape[0] > 2:
                    too_much_info[column_stats[0]] = column_stats[1].index[values_to_shrink_indices].to_list()

        for column_stats in ord_features_stats:
            if column_stats[1].shape[0] <= 1:
                no_info.append(column_stats[0])
        X = self._shrink_labels(X, too_much_info)
        self.discarded = (no_info, too_much_info)

        return X

    def inverse_preprocessor(self, encoded_matrix: np.ndarray) -> pd.DataFrame:
        """
        Reverses the transformation on the encoded data.

        Args:
            encoded_matrix (np.ndarray): Encoded data matrix.

        Returns:
            pd.DataFrame: DataFrame with the original features.
        """
        preprocessor_input_columns = []
        transformer_input_columns = {}
        preprocessor_output_columns = []
        inverse_preprocessors_map = {}

        for w, transformer in enumerate(self.transformer.transformers_):
            if transformer[0] != "remainder":
                preprocessor_input_columns += transformer[-1]
                inv_transform = transformer[1].inverse_transform
                feat_names = transformer[1].get_feature_names()
                partial_output_columns = feat_names.tolist() if len(feat_names) > 0 else transformer[-1]
                preprocessor_output_columns += partial_output_columns
                transformer_input_columns[w] = transformer[-1]
                inverse_preprocessors_map[
                    tuple([preprocessor_output_columns.index(i) for i in partial_output_columns])
                ] = inv_transform

        non_encoded_dataframe = pd.DataFrame(columns=preprocessor_input_columns)
        for j, (encoded_columns_indices, inverse_transform) in enumerate(inverse_preprocessors_map.items()):
            encoded_values = encoded_matrix[:, list(encoded_columns_indices)]
            decoded_values = inverse_transform(encoded_values)
            for i1, i2 in enumerate(transformer_input_columns[j]):
                non_encoded_dataframe[i2] = decoded_values[:, i1]

        return non_encoded_dataframe[preprocessor_input_columns]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms the input DataFrame using the fitted transformer.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            np.ndarray: Transformed data.
        """
        X = X.copy()
        X = self._shrink_labels(X, self.discarded[1])

        for transformer in self.transformer.transformers_:
            if "categorical_transformer" in transformer:
                categories = [cat for cat in transformer[2]]
                X[categories] = X[categories].astype(str)
        
        if self.time_index is None:
            x_batch_preprocessed = self.transformer.transform(X)
            return x_batch_preprocessed

        else:
            x_batch_preprocessed = self.transformer.transform(X[self.meta_columns])
            x_shape = X[self.time_index].nunique()
            X_proc = np.zeros((x_shape, self.max_sequence_length*self.n_time_features))
            for j,i in enumerate(X[self.time_index].unique()):
                dt = (X[X[self.time_index]==i][self.time_columns].values-self.time_mean)/self.time_std
                L = self.max_sequence_length-dt.T.shape[1]
                dt = np.hstack([dt.T,np.zeros((self.n_time_features,L))])
                X_proc[j,:] = dt.reshape(self.max_sequence_length*self.n_time_features,-1).T
            return X_proc, x_batch_preprocessed


    def reverse_transform(self, x: np.ndarray) -> pd.DataFrame:
        """
        Reverses the transformations on the data matrix to get the original DataFrame.

        Args:
            x (np.ndarray): Encoded data matrix.

        Returns:
            pd.DataFrame: Original DataFrame after reversing transformations.
        """
        if self.time_index is None:
            x = self.inverse_transformer(x).fillna(0)
            x = x[self.sorted_columns]
            for w in [i for i in self.rules.keys() if 'embed_category' in self.rules[i][0]]:
                column = create_series(self.emb_rules[w][0], self.emb_rules[w][1], self.emb_rules[w][2], x)
                x[w] = column

            for w in [i for i in self.rules.keys() if 'sum' in self.rules[i][0]]:
                column = pd.Series(np.zeros(x.shape[0]))
                for w1 in self.rules[w][1]:
                    column += w1[0] * x[w1[1]]
                x[w] = column
        else:
            cols = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
            out = []
            for j,i in enumerate(cols):
                oi = pd.DataFrame(x[:,0+self.max_sequence_length*j:self.max_sequence_length*(j+1)]*self.time_std[j]+self.time_mean[j])
                oi.columns = [i+'_'+str(w) for w in range(self.max_sequence_length)]
                out.append(oi)
            x = pd.concat(out, axis=1)
            
        return x

    def get_features_sizes(self) -> Tuple[List[int], List[int]]:
        """
        Gets the sizes of ordinal and categorical features after transformation.

        Returns:
            Tuple: Sizes of ordinal and categorical features.
        """
        numerical_sizes = []
        categorical_sizes = []
        for transformer in self.transformer.transformers_:
            if "numerical_transformer" in transformer:
                numerical_sizes.append(len(transformer[-1]))
            if "datetime_transformer" in transformer:
                if numerical_sizes:
                    numerical_sizes[0] += len(transformer[-1])
                else:
                    numerical_sizes.append(len(transformer[-1]))

            if "categorical_transformer" in transformer:
                one_hot_encoder = transformer[1].encoder
                categorical_sizes = [len(cat) for cat in one_hot_encoder.categories_]

        return numerical_sizes, categorical_sizes

    def get_numerical_features(self) -> List[str]:
        """
        Gets the list of ordinal features.

        Returns:
            List[str]: List of ordinal feature names.
        """
        return self.numerical_features.copy()

    def get_categorical_features(self) -> List[str]:
        """
        Gets the list of categorical features.

        Returns:
            List[str]: List of categorical feature names.
        """
        return self.categorical_features.copy()

    def get_datetime_features(self) -> List[str]:
        """
        Gets the list of datetime features.

        Returns:
            List[str]: List of datetime feature names.
        """
        return self.datetime_features.copy()