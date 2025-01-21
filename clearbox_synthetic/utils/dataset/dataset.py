"""
This module provides tools and classes for working with tabular datasets, including data manipulation, 
validation, preprocessing, and analysis.
It is designed for flexibility in machine learning workflows, supporting regression and classification tasks,
and ensuring dataset integrity through automated checks and validations. 
"""

import copy
import pickle
from jax.numpy import ndarray

import pandas as pd
from loguru import logger

from datetime import datetime
from typing import List, Dict, Set, Tuple, Union, Optional, IO, Callable, Any, ClassVar
from pydantic import BaseModel, field_validator,ConfigDict
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

DTYPES_MAP = {"b": bool, "i": int, "u": int, "f": float, "c": float, "O": str, "S": str}


def _infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer column types from the dataframe.
    """
    # return {col: infer_column_type(df[col]) for col in df.columns}
    pass


class Dataset(BaseModel):
    """
    A felxible class for tabular dataset manipulation.

    Attributes
    ----------
    data : pandas.DataFrame
        A tabular dataset, more than 1 row.
    timestamp : datetime, default=datetime.now()
        A datetime timestamp.
    name : str, optional
        A string name for the dataset.
    target_column : str or int or Tuple, optional
        The target column (y) name.
    bounds : dict of dict, optional
        A dictionary of allowed values for each column except the target one.
        For an numeric column use 'column': {'max': max_value, 'min': min_value}.
        For a categorical column use 'column': {allowed_value+}.
    regression : bool, default False
            Indicates whether the dataset is used or not for a regression problem.
    """

    data: pd.DataFrame
    timestamp: Optional[datetime]
    name: Optional[str]
    target_column: Optional[Union[int, str, tuple]]
    sequence_index: Optional[Union[int, str]] = None
    group_by: Optional[Union[int, str]] = None
    column_types: Optional[Dict[str, str]]
    bounds: Optional[Dict]
    regression: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # to_csv: ClassVar = pd.DataFrame.to_csv

    @field_validator("timestamp", mode="before", check_fields=True)
    @classmethod
    def set_timestamp_now(cls, v):
        return v or datetime.now()

    @field_validator("target_column", mode="before")
    @classmethod
    def validate_target_column(cls, v, values):
        if v is not None and isinstance(v, str) and v not in values["data"].columns:
            raise ValueError(f"'{v}' is not a column of the dataset.")
        if v is not None and isinstance(v, int) and v >= len(values["data"].columns):
            raise ValueError(f"'{v}' is not a valid index.")
        return v

    @field_validator("group_by", mode="before")
    @classmethod
    def validate_group_by(cls, v, values):
        if v is not None and isinstance(v, str) and v not in values["data"].columns:
            raise ValueError(f"'{v}' is not a column of the dataset.")
        if v is not None and isinstance(v, int) and v >= len(values["data"].columns):
            raise ValueError(f"'{v}' is not a valid index.")
        return v

    @field_validator("sequence_index", mode="before")
    @classmethod
    def validate_sequence_index(cls, v, values):
        if v is not None and isinstance(v, str) and v not in values["data"].columns:
            raise ValueError(f"'{v}' is not a column of the dataset.")
        if v is not None and isinstance(v, int) and v >= len(values["data"].columns):
            raise ValueError(f"'{v}' is not a valid index.")
        return v

    @field_validator("bounds", mode="before")
    @classmethod
    def validate_bounds(cls, v, values):
        if "data" not in values:
            raise ValueError("Data attribute is missing; cannot validate bounds.")
        if not v:
            numerical_cols = values["data"].select_dtypes(include=["number", "datetime"])
            categorical_cols = values["data"].select_dtypes(include=["object", "category"])
            bounds = {}
            for num in numerical_cols.columns:
                bounds[num] = {"min": values["data"][num].min(), "max": values["data"][num].max()}
            for cat in categorical_cols.columns:
                bounds[cat] = set(values["data"][cat].dropna().unique())
            return bounds
        return v

    @field_validator("column_types", mode="before")
    @classmethod
    def validate_column_types(cls, v, values):
        if v:
            if set(v.keys()) != set(values["data"].columns):
                raise ValueError("Column types must be defined for all columns.")
        return v

    @field_validator("regression", mode="before")
    @classmethod
    def validate_regression(cls, v):
        return v or False
    
    @classmethod
    def from_csv(
        cls,
        csv_file: Union[str, IO],
        timestamp: datetime = None,
        target_column: Union[int, str, Tuple] = None,
        sequence_index: Union[int, str] = None,
        group_by: Union[int, str] = None,
        column_types: Dict[str, str] = None,
        name: str = None,
        bounds: Dict = None,
        sep: str = ",",
        header: Union[str, int, List[int]] = "infer",
        cols_names: list = None,
        index_col: Union[int, str, List, bool] = None,
        usecols: List = None,
        dtype: Union[str, Dict] = None,
        converters: Dict = None,
        skiprows: int = None,
        nrows: int = None,
        na_values: Any = "?",
        skip_blank_lines: bool = True,
        dayfirst: bool = False,
        thousands: str = None,
        decimal: str = ".",
        regression: bool = False,
        drop_target_na_rows: bool = True,
    ) -> "Dataset":
        """
        Create a Dataset object loading the dataset from a csv file.

        Parameters
        ----------
        csv_file : string or file-like object
            The csv file path as a string or the csv file. By file-like object, we refer to objects with a read()
            method, such as a file handler (e.g. via builtin open function) or StringIO.
        timestamp : datetime, optional
            Timestamp assigned to the dataset.
        target_column : str or int or Tuple, optional
            The y column of the dataset (Supervised Machine Learning)
        column_types : dict, optional
            An optional dictionary that indicates for each column the data type.
        name : string, optional
            A string name for the dataset.
        bounds : dict of dict, optional
            A dictionary of allowed values. For an ordinal column use 'column': {'max': max_value, 'min': min_value}.
            For a categorical column use 'column': {allowed_value+}.
        sep : string, default ','
            Delimiter char/string to use.
        header : int, list of int, default ‘infer’
            Row number(s) to use as the column names, and the start of the data. Default behavior is to infer the column
            names: if no names are passed the behavior is identical to header=0 and column names are inferred from the
            first line of the file, if column names are passed explicitly then the behavior is identical to header=None.
            Explicitly pass header=0 to be able to replace existing names. The header can be a list of integers that
            specify row locations for a multi-index on the columns e.g. [0,1,3]. Intervening rows that are not specified
            will be skipped (e.g. 2 in this example is skipped).
        cols_names : list, optional
            List of column names to use. If file contains no header row, then you should explicitly pass header=None.
            Duplicates in this list are not allowed.
        index_col :  int, str, sequence of int / str, or False, optional
            Column(s) to use as the row labels of the DataFrame, either given as string name or column index. If a
            sequence of int / str is given, a MultiIndex is used.
        usecols : list-like or callable, optional
            Return a subset of the columns. If list-like, all elements must either be positional (i.e. integer indices
            into the document columns) or strings that correspond to column names provided either by the user in names
            or inferred from the document header row(s). For example, a valid list-like usecols parameter would be
            [0, 1, 2] or ['foo', 'bar', 'baz']. Element order is ignored, so usecols=[0, 1] is the same as [1, 0]. To
            instantiate a DataFrame from data with element order preserved use pd.read_csv(data, usecols=['foo', 'bar'])
            [['foo', 'bar']] for columns in ['foo', 'bar'] order or pd.read_csv(data, usecols=['foo', 'bar'])
            [['bar', 'foo']] for ['bar', 'foo'] order. If callable, the callable function will be evaluated against the
            column names, returning names where the callable function evaluates to True. An example of a valid callable
            argument would be lambda x: x.upper() in ['AAA', 'BBB', 'DDD']. Using this parameter results in much faster
            parsing time and lower memory usage.
        prefix : str, optional
            Prefix to add to column numbers when no header, e.g. ‘X’ for X0, X1, …
        dtype : Type name or dict of column -> type, optional
            Data type for data or columns. E.g. {‘a’: np.float64, ‘b’: np.int32, ‘c’: ‘Int64’}
            Use str or object together with suitable na_values settings to preserve and not interpret dtype.
            If converters are specified, they will be applied INSTEAD of dtype conversion.
        converters : dict, optional
            Dict of functions for converting values in certain columns. Keys can either be integers or column labels.
        skiprows : int, optional
            Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file.
        nrows : int, optional
            Number of rows of file to read. Useful for reading pieces of large files.
        na_values : scalar, string, list-like, or dict, default '?'
            Additional string to recognize as NA/NaN value.
        skip_blank_lines : bool, default True
            If True, skip over blank lines rather than interpreting as NaN values.
        parse_dates : bool or list of int or names or list of lists or dict, default False
            The behavior is as follows:
                * boolean. If True -> try parsing the index.
                * list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3 each as a separate date column.
                * list of lists. e.g. If [[1, 3]] -> combine columns 1 and 3 and parse as a single date column.
                * dict, e.g. {‘foo’ : [1, 3]} -> parse columns 1, 3 as date and call result ‘foo’.
            If a column or index cannot be represented as an array of datetimes, say because of an unparseable value or
            a mixture of timezones, the column or index will be returned unaltered as an object data type. For
            non-standard datetime parsing, use pd.to_datetime after pd.read_csv. To parse an index or column with a
            mixture of timezones, specify date_parser to be a partially-applied pandas.to_datetime() with utc=True.
        infer_datetime_format : bool, default False
            If True and parse_dates is enabled, pandas will attempt to infer the format of the datetime strings in the
            columns, and if it can be inferred, switch to a faster method of parsing them. In some cases this can
            increase the parsing speed by 5-10x.
        keep_date_col : bool, default False
            If True and parse_dates specifies combining multiple columns then keep the original columns.
        date_parser : function, optional
            Function to use for converting a sequence of string columns to an array of datetime instances. The default
            uses dateutil.parser.parser to do the conversion. Pandas will try to call date_parser in three different
            ways, advancing to the next if an exception occurs: 1) Pass one or more arrays (as defined by parse_dates)
            as arguments; 2) concatenate (row-wise) the string values from the columns defined by parse_dates into a
            single array and pass that; and 3) call date_parser once for each row using one or more strings
            (corresponding to the columns defined by parse_dates) as arguments.
        dayfirst : bool, default False
            DD/MM format dates, international and European format.
        thousands : str, optional
            Thousands separator.
        decimal : str, default ‘.’
            Character to recognize as decimal point (e.g. use ‘,’ for European data).
        regression : bool, default False
            Indicates whether the dataset is used or not for a regression problem.
        drop_target_na_rows : bool, default True
            If True and target_column is not None (Labeled Dataset), drop all rows containing na value in the target column

        Returns
        -------
        Dataset
            A new Dataset instance.
        """
        data = pd.read_csv(
            csv_file,
            sep=sep,
            header=header,
            names=cols_names,
            index_col=index_col,
            usecols=usecols,
            dtype=dtype,
            converters=converters,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            skip_blank_lines=skip_blank_lines,
            dayfirst=dayfirst,
            thousands=thousands,
            decimal=decimal,
            on_bad_lines="error",
        )

        if header is None and cols_names is None:
            cols_names = ["Column #{}".format(i) for i in data.columns]
            data.columns = cols_names
            if target_column:
                target_column = data.columns[target_column]

        if target_column is not None and target_column not in data:
            logger.warning(
                f"Target column '{target_column}' is not a column in the dataset, target_column set as None (Unlabeled Dataset) "
            )
            target_column = None

        if target_column and drop_target_na_rows:
            target_column_na_values = data[target_column].isnull().sum()
            if target_column_na_values > 0:
                logger.info(
                    f"There are {target_column_na_values} rows containing na value in the target column, they will be dropped."
                )
                data.dropna(subset=[target_column], inplace=True)
                if len(data.index) == 0:
                    raise ValueError(
                        "After removing the rows containing na value in the target column, the dataset is empty."
                    )

        return cls(
            timestamp=timestamp,
            data=data,
            target_column=target_column,
            sequence_index=sequence_index,
            group_by=group_by,
            column_types=column_types,
            name=name,
            bounds=bounds,
            regression=regression,
        )

    @classmethod
    def to_csv(self, path: str):
        """
        Generate and save a csv file starting from the dataset.

        Parameters
        ----------
        path : str
            The path where to save the generated csv file.
        """
        self.data.to_csv(
            path,
            index=False,
        )

    @classmethod
    def get_x(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Return all columns of the dataset except the target column (y).

        Returns
        -------
        pd.Dataframe or pd.Series
            All columns of the dataset except the target column (y) as a pandas Dataframe.
        """
        return self.subset(
            [column for column in self.columns() if column != self.target_column]
        )

    @classmethod
    def get_group_by(self) -> pd.Series:
        """
        Return the sequence index of the dataset.

        Returns
        -------
        pd.Series
            The sequence index of the dataset.
        """
        if self.group_by is not None:
            return self.data[self.group_by]
        else:
            return None

    @classmethod
    def get_y(self) -> pd.Series:
        """
        Return the target column of the dataset (y).

        Returns
        -------
        pd.Series
            The target column (y) of the dataset.
        """
        if self.target_column is not None:
            return self.data[self.target_column]
        else:
            return None

    @classmethod
    def get_y_std(self) -> float:
        """
        Return the std of target column of the dataset (y), if regression.

        Returns
        -------
        float
            The std of the target column (y) of the dataset.
        """
        if self.regression is True:
            std = self.data[self.target_column].values.std()
            if std >= 1e-3:
                return std
            else:
                return 1.0
        else:
            return None

    @classmethod
    def get_y_mean(self) -> float:
        """
        Return the mean of target column of the dataset (y), if regression.

        Returns
        -------
        float
            The mean of the target column (y) of the dataset.
        """
        if self.regression is True:
            return self.data[self.target_column].values.mean()
        else:
            return None

    @classmethod
    def get_normalized_y(self) -> ndarray:
        """
        Standardize the target column of the dataset (y), if regression is True

        Returns
        -------
        float
            The standardized target column (y)
        """
        if self.regression is True:
            y_mean = self.get_y_mean()
            y_std = self.get_y_std()
            return ((self.data[self.target_column].values - y_mean) / y_std).reshape(
                -1, 1
            )
        else:
            return None

    @classmethod
    def get_label_encoded_y(self) -> pd.Series:
        """
        Return the target column of the dataset (y), preprocessed with a Label Encoder
        and the relative labels.

        Returns
        -------
        pd.Series
            The target column (y) of the dataset.
        """
        if self.target_column is not None:
            if self.regression is True:
                return self.data[self.target_column]
            else:
                y_encoder = LabelEncoder()
                return (y_encoder.fit_transform(self.get_y()), y_encoder.classes_)
        else:
            return None

    @classmethod
    def get_one_hot_encoded_y(self) -> pd.Series:
        """
        Return the target column of the dataset (y), preprocessed with a One Hot Encoder.

        Returns
        -------
        pd.Series
            The one hot encoded target column (y) of the dataset.
        """
        if self.target_column is not None:
            if self.regression is True:
                return self.data[self.target_column]
            else:
                y_encoder = OneHotEncoder(handle_unknown="ignore")
                return y_encoder.fit_transform(
                    self.get_y().to_numpy().reshape(-1, 1)
                ).toarray()
        else:
            return None

    @classmethod
    def get_n_classes(self) -> int:
        """
        Return the number of unique values in the target column (y) of the dataset.

        Returns
        -------
        int
            The number of unique values in the target column (y) of the dataset.

        """
        if self.target_column is not None:
            if self.regression is True:
                return 1
            else:
                return len(self.get_y().unique())
        else:
            return 0

    @classmethod
    def rows_number(self) -> int:
        """
        Return the number of rows of the dataset.

        Returns
        -------
        int
            Number of rows of the dataset.
        """
        return self.data.shape[0]

    @classmethod
    def columns_number(self) -> int:
        """
        Return the number of columns/features of the dataset.

        Returns
        -------
        int
            Number of columns of the dataset.
        """
        return self.data.shape[1]

    @classmethod
    def columns(self, include: Union[int, str, List] = None) -> List[str]:
        """
        Return the list of column names of (a subset of) the dataset.

        Parameters
        ----------
        include : scalar or list-like, optional
            A selection of dtypes or strings to be included. To select all numeric types, use 'number'. To select
            strings you must use the 'object' dtype, but note that this will return all object dtype columns. To select
            Pandas categorical dtypes, use 'category'.

        Returns
        -------
        list
            Names of columns of (a subset of) the dataset.
        """
        return (
            list(self.data.select_dtypes(include=include).columns)
            if include
            else list(self.data.columns)
        )

    @classmethod
    def x_columns(self, include: Union[int, str, List] = None) -> List[str]:
        """
        Return the list of column names of the X subset of the dataset (no target column).

        Parameters
        ----------
        include : scalar or list-like, optional
            A selection of dtypes or strings to be included. To select all numeric types, use 'number'. To select
            strings you must use the 'object' dtype, but note that this will return all object dtype columns. To select
            Pandas categorical dtypes, use 'category'.

        Returns
        -------
        list
            Names of columns of the X subset of the dataset (no target column).

        """
        x_columns: List = self.columns()
        if self.target_column is not None:
            x_columns.remove(self.target_column)
        return (
            list(self.data[x_columns].select_dtypes(include=include).columns)
            if include
            else x_columns
        )
    
    @classmethod
    def columns_types(self) -> Dict:
        """
        Return a dict with the column name as key and the column dtype as value.

        Returns
        -------
        dict
            Columns types.
        """
        types = {}
        for column_name, column_dtype in zip(self.x_columns(), self.get_x().dtypes):
            types[column_name] = column_dtype
        return types

    @classmethod
    def column_bounds(self, column: Union[str, Tuple]) -> Union[Dict, Set]:
        """
        Return the bounds of a single column of the dataset.

        Parameters
        ----------
         column : str or tuple of str
            Name of a column.

        Returns
        -------
        dict
            Column bounds.
        """
        return self.bounds[column]

    @classmethod
    def subset(self, columns: List) -> Union[pd.DataFrame, pd.Series]:
        """
        Return a subset of the dataset given a list of column names.

        Parameters
        ----------
        columns : list
            List of column names as str or tuple in case of multi-level index.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            Subset Column(s) from the dataset.
        """
        return self.data[columns]

    @classmethod
    def subset_by_type(self, include: Union[int, str, List]) -> pd.DataFrame:
        """
        Return a subset of the dataset based on the column dtypes.

        Parameters
        ----------
        include : scalar or list-like
            A selection of dtypes or strings to be included. To select all numeric types, use 'number'. To select
            strings you must use the 'object' dtype, but note that this will return all object dtype columns. To select
            Pandas categorical dtypes, use 'category'.

        Returns
        -------
        pd.Dataframe
            A subset of the dataset including the dtypes in include.

        """
        return self.data.select_dtypes(include=include)

    @classmethod
    def row_by_index(self, idx: int) -> pd.Series:
        """
        Return a row of the dataset given an index.

        Parameters
        ----------
        idx : int
            A single row index value.

        Returns
        -------
        Pandas.series
            A single row of the dataset.
        """
        return self.data.loc[idx]

    @classmethod
    def pop_column(self, column: Union[str, Tuple]) -> Union[pd.Series, pd.DataFrame]:
        """
        Return a column and drop it from the dataset.

        Parameters
        ----------
        column : str or tuple of str
            Name of the column to be popped as a str or a tuple of str in case of multi-level index.

        Returns
        -------
        pd.Series
            The column popped out.
        """
        self.bounds.pop(column)
        return self.data.pop(column)

    @classmethod
    def drop_columns(self, columns: Union[str, List]) -> None:
        """
        Drop one or more columns of the dataset. This method transform the dataset in place.

        Parameters
        ----------
        columns : list
            List of column names to drop as str or tuple of str in case of multi-level index.
        """
        if isinstance(columns, str):
            self.bounds.pop(columns)
        else:
            for col in columns:
                self.bounds.pop(col)
        self.data.drop(columns, axis=1, inplace=True)

    @classmethod
    def info(self) -> None:
        """
        Display a concise summary of the dataset: information about the pd.DataFrame including the index dtype and
        columns dtypes, non-null values and memory usage.
        """
        self.data.info()

    @classmethod
    def head(self, num_rows: int = 5) -> pd.DataFrame:
        """
        Return the first num_rows rows of the dataset. It is useful for quickly testing if your object has the right
        type of data in it. If num_rows is not passed, display the first 5 rows.

        Parameters
        ----------
        num_rows : int, optional
            Number of rows to display.

        Returns
        -------
        pandas.DataFrame
            Return the first num_rows rows of the dataset.
        """
        return self.data.head(num_rows)

    @classmethod
    def describe(self, include: str = "all") -> pd.DataFrame:
        """
        Return descriptive statistics that summarize the central tendency, dispersion and shape of the dataset
        distribution, excluding NaN values. Analyzes both numeric and object series, as well as DataFrame columns sets
        of mixed data types.

        Parameters
        ----------
        include : str or list-like of dtypes or None, default 'all'
            By default all columns of the input will be included in the output. Using a list-like of dtypes limits the
            results to the provided data types. To limit the result to numeric types submit 'number'. To limit it
            to object columns submit 'object'.

        Returns
        -------
        pandas.DataFrame
            Return descriptive statistics of the dataset
        """
        return self.data.describe(include=include).transpose()

    @classmethod
    def unique_values(self, columns: List = None) -> Dict:
        """
        Return a dictionary of unique values of (a subset of) the dataset.

        Parameters
        ----------
        columns : list, optional
            List of column names as string or tuple in case of multi-level index.
            If None, return all the unique values for every column.

        Returns
        -------
        dict
            A dictionary {'column' -> [unique_value+]}.
        """
        feats: List = self.columns() if columns is None else list(columns)
        return {feat: self.data[feat].unique() for feat in feats}

    @classmethod
    def value_counts(self, column: Union[int, str, Tuple]) -> pd.DataFrame:
        """
        Given a target column, return a dataframe containing the number of samples and the frequency for each unique
        values of the column in the dataset. Useful to check if the dataset is balanced with respect to the y column.

        Parameters
        ----------
        column : str or tuple of str
            A string name of a single column or tuple of string for a multi-indexed column.

        Returns
        -------
        pd.Dataframe
            Number and frequency of samples in the dataset for each unique values of the column col.
        """
        return pd.DataFrame(
            {
                "count": self.data.loc[:, column].value_counts(dropna=False),
                "freq": (
                    self.data.loc[:, column].value_counts(dropna=False, normalize=True)
                    * 100
                ).round(2),
            }
        )

    @classmethod
    def target_balance(self) -> pd.DataFrame:
        """
        Return a dataframe containing the number of samples and the frequency for each unique values of the target
        column.

        Returns
        -------
        pd.Dataframe
            Number and frequency of samples in the dataset for each unique values of the target column
        """
        if self.target_column is not None:
            return self.value_counts(self.target_column)
        else:
            return None

    @classmethod
    def get_values(self):
        """
        Return the Dataset as a NumPy array/matrix.

        Returns
        -------
        nd.array
            Numpy version of the dataset, just the values, no more column names or indices.
        """
        return self.data.to_numpy()

    @classmethod
    def categorical_map(self) -> Dict:
        """
        Return a map of the categorical feature indices and corresponding values.

        Returns
        -------
        category_map : dict
            A dictionary with keys being the indices of the categorical columns and values being lists of unique values
            for that column.

        """
        features = list(self.columns())
        categorical_features = [f for f in features if self.data[f].dtype == "O"]
        category_map = dict()
        for f in categorical_features:
            category_map[features.index(f)] = list(self.data[f].unique().astype(str))
        return category_map

    @classmethod
    def types_map(self) -> Dict:
        """
        Return a map of the features and corresponding type. This is necessary to create a Pydantic model based on
        dataset features.

        Returns
        -------
        types_map : dict
            A dictionary with keys being the columns names and values being the type of that column.

        """
        features = list(self.x_columns())
        types_map = {f: (DTYPES_MAP[self.data[f].dtype.kind], ...) for f in features}
        return types_map

    @classmethod
    def pairwise_correlation(self) -> pd.DataFrame:
        """
        Compute pairwise correlation of columns, excluding NA/null values.

        Returns
        -------
        pandas.Dataframe
            Correlation matrix of the dataset.
        """
        return self.data.corr()

    @classmethod
    def column_correlation(self, column: Union[str, Tuple]) -> pd.Series:
        """
        Compute correlation between a single numeric column and each other columns in the dataset.

        Parameters
        ----------
        column : str or tuple of str
            A string name of a single numeric column or tuple of string for a multi-level index.
        Returns
        -------
        pandas.Series
            Correlation values sorted by descending order.
        """
        corr_mat = self.data.corr()
        return corr_mat.loc[:, column].sort_values(ascending=False)

    @classmethod
    def check_na_values(self) -> Union[pd.Series, None]:
        """
        Check for columns with missing values in the dataset.

        Returns
        -------
        pandas.Series or None
            A series with the number of missing values for each columns that has missing values or None if there are
            no missing values in the dataset.
        """
        na_series = self.data.isna().sum()
        na_series = na_series[na_series > 0]
        if len(na_series) > 0:
            return na_series
        else:
            return None

    @classmethod
    def drop_na_values(self, axis: int = 0, how: str = "any") -> None:
        """
        Drop all the missing values in the dataset. This method transform the dataset in place. Check also
        fill_na_values().

        Parameters
        ----------
        axis : {0, 1}
            Axis along which to fill missing values.
        how : {'any', 'all'}, default 'any'
            Determine if row or columns is removed from the dataset, when we have at least one NA or all NA.
        """
        self.data.dropna(axis=axis, how=how, inplace=True)
        self._update_all_bounds()
    
    @classmethod
    def fill_na_values(
        self, fill_with: Union[str, int, float, Dict], columns: List = None
    ) -> Union[str, int, float, Dict]:
        """
        Fill missing values in the dataset. You can choose which column(s) to fill and what value(s) use to
        fill it.

        Parameters
        ----------
        fill_with : {'mean', 'median'}, scalar or dict
            Value(s) to use to fill it the columns. You can choose from {mean, median}, if you want to fill the missing
            values in a numeric columns with its mean or median value. You can write a specific string or scalar, if you
            want to fill all the missing values in the selected columns with just that particular value. You can pass a
            dict with key==column_name -> value==fill_with (eg. {'country': 'italy', 'language': 'italian'}, if you want
            to specify what values to use for a subset of columns.
        columns : list, optional
            List of column names as string or tuple in case of multi-level index. If None, if fill_with is a dictionary
            the method fill the columns specified in the dictionary key, elif fill_with is in {mean, median} the method
            fill all the columns containing Nan with the relative mean/median (error if there is at least one object
            columns), else the method fill all Nan values in dataframe with the single specified value.

        Returns
        -------
        fill_with
            The value(s) used to fill the missing values, useful if you have choose median or mean because you have to
            fill the missing values in the test set with the same values.
        """
        if fill_with == "median":
            fill_with = self.data.median()
        elif fill_with == "mean":
            fill_with = self.data.mean()

        if columns is not None:
            self.data[columns] = self.data[columns].fillna(value=fill_with)
        else:
            self.data.fillna(value=fill_with, inplace=True)
        self._update_all_bounds()
        return fill_with
    
    @classmethod
    def check_duplicates(self, columns: List = None) -> int:
        """
        Return number of duplicated rows in the dataset, optionally considering only certain columns.

        Parameters
        ----------
        columns : list, optional
            List of column names as string or tuple in case of multi-level index to check for duplicates.
            By default use all the columns.

        Returns
        -------
        int
            Number of duplicated rows in the dataset.
        """
        return self.data.duplicated(columns).sum()
    
    @classmethod
    def drop_duplicates(self, columns: List = None) -> None:
        """
        Remove duplicate rows from the dataset, optionally considering only certain columns. This method
        transform the dataset in place.

        Parameters
        ----------
        columns : list, optional
            List of column names as string or tuples of str in case of multi-level index to check for duplicates.
            By default use all the columns.
        """
        self.data.drop_duplicates(subset=columns, inplace=True)
    
    @classmethod
    def variance(
        self,
        columns: List = None,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = None,
    ) -> pd.Series:
        """
        Return unbiased variance over requested axis of the dataset. Normalized by N by default.

        Parameters
        ----------
        columns : list, optional
            List of column names as string or tuples of str in case of multi-level index to check for variance.
            By default use all the columns.
        axis : {0, 1}, default 0
            0 for index, 1 for columns
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA
        numeric_only : bool, default False
            Include only float, int, boolean columns. If None, will attempt to use everything, then use only numeric
            data. Not implemented for Series.

        Returns
        -------
        pd.Series
            Variance over requested axis as descending sorted pandas series.
        """
        X = copy.deepcopy(self)
        X.fill_na_values(fill_with="median")
        if X.check_na_values():
            X.fill_na_values(fill_with="NaN")
        X.categorical_to_ordinal()
        return (
            X.subset(columns)
            .var(axis=axis, skipna=skipna, ddof=0, numeric_only=numeric_only)
            .sort_values(ascending=False)
            if columns is not None
            else X.data.var(
                axis=axis, skipna=skipna, ddof=0, numeric_only=numeric_only
            ).sort_values(ascending=False)
        )
    
    @classmethod
    def map_column(self, column: Union[str, Tuple[str]], dict_map: Dict) -> None:
        """
        Map values of a column according to the 'dict_map' correspondence. It substitute each value in the
        columns with another value. This method transform the dataset in place. Might be a better idea to use
        map_columns.

        Parameters
        ----------
        column : str or tuple of str
            Name of the column to map as a str or tuple of str in case of multi-level index.
        dict_map : dict
            Dictionary containing the correspondences value_to_map -> new_value.
        """
        col_temp = self.data[column].to_numpy()
        for old_val, new_val in dict_map.items():
            col_temp[col_temp == old_val] = new_val
        self.data[column] = col_temp
        self._update_column_bounds(column)
    
    @classmethod
    def map_columns(self, mapping_cols: Dict) -> None:
        """
        Map the values of some columns of the dataset to new values.

        Parameters
        ----------
        mapping_cols : dict
            A dictionary that contains the columns to map as keys and the values_map as values
        """
        for f, f_map in mapping_cols.items():
            self.map_column(f, f_map)
    
    @classmethod
    def discretize(
        self,
        column: Union[str, Tuple],
        bins: Union[int, List] = 4,
        strategy: str = "edges",
        quantiles: Union[int, List[float]] = 4,
        labels: List[str] = None,
        right: bool = True,
        precision: int = 3,
    ) -> None:
        """
        Bin columns values into discrete intervals. Supports binning into an equal number of bins, a pre-specified
        array of bins or quantile-based discretization that discretize variable into equal-sized buckets based on rank
        or on sample quantiles. It is useful to convert a continuous variable into a categorical variable. This
        method transform the dataset in place.

        Parameters
        ----------
        column : str or tuple of str
            Name of the column to bin as a str or tuple of str in case of multi-level index.
        bins : int or list of scalars, default 4
            - int: defines the number of equal-width bins in the range of column. The range of column is extended by
            .1% on each side to include the minimum and maximum values of column.
            - list of scalars: defines the bin edges allowing for non-uniform width. No extension of the range of column
            is done.
        strategy : {'edges', 'quantile'}, default 'edges'
            Strategy to perform the discretization. 'edges' for a simple discretization into an equal number of bins or
            a pre-specified array of bins. 'quantiles' for quantile-based discretization.
        quantiles : int or list of scalars, default 4
            Number of quantiles: 10 for deciles, 4 for quartile, etc. Alternately array of quantiles,
            e.g. [0, .25, .5, .75, 1.] for quartiles
        labels : list of string, optional
            Labels string names for the returned bins. Must be the same length as the resulting bins. If None, returns
            only integer indicators of the bins.
        right : bool, default True
            Whether the bins includes the rightmost edge or not. If right == True, then the bins [1, 2, 3, 4]
            correspond to (1,2], (2,3], (3,4].
        precision : int, default 3
            The precision at which to store and display the bins labels.
        """
        if strategy == "quantiles":
            self.data[column] = pd.qcut(
                self.data[column],
                quantiles,
                labels=labels,
                precision=precision,
                duplicates="drop",
            )
        elif strategy == "edges":
            self.data[column] = pd.cut(
                self.data[column],
                bins=bins,
                labels=labels,
                right=right,
                precision=precision,
            )
        else:
            raise ValueError(
                "There is not a '{}' strategy. No operation performed.".format(strategy)
            )
        self._update_column_bounds(column)
    
    @classmethod
    def scaler(self, column: Union[str, Tuple[str]], strategy: str = "min-max") -> None:
        """
        Scale values of a numeric column.

        Parameters
        ----------
        column : str or tuple of str
            Name of the column to scale as a str or tuple of str in case of multi-level index.
        strategy : {'min-max', 'standard'}, default 'min-max'
            The scaler strategy.

        Notes
        ------
        Generally, Machine Learning algorithms don't perform well when the input numerical attributes have very
        different scales. Note that scaling the target values is generally not required.
        There are two ways to scale the numeric values:
        * min-max (normalization): values are shifted and rescaled so that they end up ranging from 0 to 1. We do this
        subtracting the minimum and dividing by the maximum minus the minimum;
        * standard: first it subtracts the mean value (so standardized values always have a zero mean), and then
        it divides by the variance so that the resulting distribution has unit variance. Unlike min-max scaling,
        standardization does not bound values to a specific range, which may be a problem for some algorithms but is
        much less affected by outliers.
        """
        if strategy == "min-max":
            min_val = self.data[column].min()
            max_val = self.data[column].max()
            min_max = max_val - min_val
            self.data[column] = (self.data[column] - min_val) / min_max
        elif strategy == "standard":
            mean_val = self.data[column].mean()
            var_val = self.data[column].var()
            self.data[column] = (self.data[column] - mean_val) / var_val
        else:
            raise ValueError(
                "There is no '{}' scaler. Choose from [min-max, standard].".format(
                    strategy
                )
            )
        self._update_column_bounds(column)
    
    @classmethod
    def scale_numeric_columns(self, strategy: str = "min-max") -> None:
        """
        Scale every numeric column in the dataset. This method transform the dataset in place.

        Parameters
        ----------
        strategy : {'min-max', 'standardization'}, default 'min-max'
            The scaler strategy. Check scaler() docs for furthers information.
        """
        num_subset = self.data.select_dtypes(include="number")
        numeric_features = [f for f in list(num_subset.columns)]
        for nf in numeric_features:
            self.scaler(nf, strategy=strategy)
    
    @classmethod
    def numerical_encoder(self, column: Union[str, Tuple]) -> None:
        """
        Encode categorical values of a column to ordinal values. This method transform the dataset in place.

        Parameters
        ----------
        column : str or tuple of str
            Name of the column to encode as a str or tuple of str in case of multi-level index.
        """

        X = self.data[column]
        enc = OrdinalEncoder()
        self.data[column] = enc.fit_transform(X.to_numpy().reshape(-1, 1))
        self._update_column_bounds(column)
    
    @classmethod
    def categorical_to_ordinal(self) -> None:
        """
        Encode every categorical column in the dataset to ordinal type. This method transform the dataset in place.
        """
        categorical_features = [f for f in self.columns() if self.data[f].dtype == "O"]
        for cf in categorical_features:
            self.numerical_encoder(cf)
    
    @classmethod
    def shuffle(self, reset_index: bool = False) -> None:
        """
        Shuffle the dataset rows in place.

        Parameters
        ----------
        reset_index : bool, default False
            If True reset the rows index after shuffling.
        """
        self.data = (
            self.data.sample(frac=1).reset_index(drop=True)
            if reset_index
            else self.data.sample(frac=1)
        )
    
    @classmethod
    def train_test_split(
        self, frac: float = 0.8, random_state: int = None
    ) -> Tuple["Dataset", "Dataset"]:
        """
        Split the instance dataset into random train and test subsets as two new Dataset instances.

        Parameters
        ----------
        frac : float, default 0.8
            Ratio between training and test set size.
        random_state : int, optional
            Seed for the random number generator. Use it for reproducibility.

        Returns
        -------
        tuple
            The training and the test set as two new Dataset instances.
        """
        train_set_df = self.data.sample(frac=frac, random_state=random_state)
        test_set_df = self.data.drop(train_set_df.index)
        return (
            Dataset(
                data=train_set_df,
                name=self.name + " Training" if self.name else "Training",
                target_column=self.target_column,
                timestamp=self.timestamp,
                bounds=self.bounds,
                sequence_index=self.sequence_index,
                group_by=self.group_by,
                column_types=self.column_types,
                regression=self.regression,
            ),
            Dataset(
                data=test_set_df,
                name=self.name + " Test" if self.name else "Test",
                target_column=self.target_column,
                timestamp=self.timestamp,
                bounds=self.bounds,
                sequence_index=self.sequence_index,
                group_by=self.group_by,
                column_types=self.column_types,
                regression=self.regression,
            ),
        )
    
    @classmethod
    def save(self, path: str) -> None:
        """
        Exports the Dataset object as serialized pickle file, given a filepath of the pickle file to create.

        Parameters
        ----------
        path: str
            Filepath of the pickle file to create.
        """
        pickle_file = open(path, "wb")
        pickle.dump(self, pickle_file)
        pickle_file.close()
    
    @classmethod
    def _update_all_bounds(self) -> None:
        """
        Update bounds for every column of dataset. To use internally after a modification of the dataset values.
        """
        numerical_cols = self.subset_by_type(include=["number", "datetime", "timedelta"])
        categorical_cols = self.subset_by_type(include=["object", "category", "bool"])
        bounds: Dict = {}
        for num in list(numerical_cols.columns):
            bounds[num] = {"min": self.data[num].min(), "max": self.data[num].max()}
        for cat in list(categorical_cols.columns):
            bounds[cat] = {c for c in self.data[cat].dropna().unique()}
        self.bounds = bounds
    
    @classmethod
    def _update_column_bounds(self, column_name: Union[str, Tuple]) -> None:
        """
        Update bounds for one column of dataset. To use internally after a modification of the dataset values.
        """
        if isinstance(self.bounds[column_name], Dict):
            self.bounds[column_name] = {
                "min": self.data[column_name].min(),
                "max": self.data[column_name].max(),
            }
        else:
            self.bounds[column_name] = {
                c for c in self.data[column_name].dropna().unique()
            }
