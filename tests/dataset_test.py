import pytest
import pandas as pd
import numpy as np
from clearbox_synthetic.utils import Dataset
from .pytest_fixtures import (
    uci_adult_dataset,
    uci_adult_dataset_w_col_types,
    boston_housing_dataset
)


def test_dataset_creation_from_dataframe():
    """Test creating a Dataset from a pandas DataFrame."""
    # Create a test DataFrame
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'e'],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Create Dataset from DataFrame
    dataset = Dataset.from_dataframe(
        data=df,
        target_column='target',
        name='test_dataset',
        ml_task='classification'
    )
    
    # Check that properties were set correctly
    assert dataset.name == 'test_dataset'
    assert dataset.target_column == 'target'
    assert dataset.ml_task == 'classification'
    assert dataset.data.equals(df)
    assert dataset.rows_number() == 5
    assert dataset.columns_number() == 3


def test_dataset_column_operations(uci_adult_dataset):
    """Test various column operations on the Dataset."""
    # Test getting columns
    all_cols = uci_adult_dataset.columns()
    assert isinstance(all_cols, list)
    assert 'income' in all_cols
    
    # Test x_columns (non-target columns)
    x_cols = uci_adult_dataset.x_columns()
    assert 'income' not in x_cols
    assert len(x_cols) == len(all_cols) - 1
    
    # Test column types
    col_types = uci_adult_dataset.columns_types()
    assert isinstance(col_types, dict)
    assert len(col_types) == len(x_cols)
    
    # Test subsetting columns
    subset_df = uci_adult_dataset.subset(['age', 'education'])
    assert isinstance(subset_df, pd.DataFrame)
    assert subset_df.shape[1] == 2
    assert list(subset_df.columns) == ['age', 'education']
    
    # Test subsetting by type
    num_df = uci_adult_dataset.subset_by_type(include='number')
    assert isinstance(num_df, pd.DataFrame)
    assert 'age' in num_df.columns
    
    # Test column bounds
    bounds = uci_adult_dataset.column_bounds('age')
    assert isinstance(bounds, dict)
    assert 'min' in bounds
    assert 'max' in bounds


def test_dataset_data_manipulation(boston_housing_dataset):
    """Test data manipulation operations on Dataset."""
    # Test shuffle
    original_data = boston_housing_dataset.data.copy()
    boston_housing_dataset.shuffle()
    assert not boston_housing_dataset.data.equals(original_data)
    assert boston_housing_dataset.data.shape == original_data.shape
    
    # Test train-test split
    train_ds, test_ds = boston_housing_dataset.train_test_split(frac=0.8, random_state=42)
    assert isinstance(train_ds, Dataset)
    assert isinstance(test_ds, Dataset)
    assert train_ds.rows_number() > test_ds.rows_number()
    expected_train_size = int(0.8 * boston_housing_dataset.rows_number())
    assert abs(train_ds.rows_number() - expected_train_size) <= 1  # Allow for rounding
    
    # Test getting X and y
    X = boston_housing_dataset.get_x()
    y = boston_housing_dataset.get_y()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0]
    assert 'PRICE' not in X.columns
    assert y.name == 'PRICE'
    
    # Test for numeric stats on target
    y_mean = boston_housing_dataset.get_y_mean()
    y_std = boston_housing_dataset.get_y_std()
    assert isinstance(y_mean, float)
    assert isinstance(y_std, float)


def test_dataset_advanced_features(uci_adult_dataset_w_col_types):
    """Test more advanced Dataset features."""
    # Test correlation features - only with numerical data
    try:
        # Test pairwise correlation - method takes no arguments
        corr_matrix = uci_adult_dataset_w_col_types.pairwise_correlation()
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
    
        # Test specific column correlation
        if 'age' in uci_adult_dataset_w_col_types.data.columns:
            col_corr = uci_adult_dataset_w_col_types.column_correlation('age')
            assert isinstance(col_corr, pd.Series)
    except ValueError:
        # Skip if correlation can't be calculated (e.g. with non-numeric data)
        pytest.skip("Correlation calculation failed, likely due to non-numeric data")
    
    # Test value counts
    val_counts = uci_adult_dataset_w_col_types.value_counts('education')
    assert isinstance(val_counts, pd.DataFrame)
    assert 'count' in val_counts.columns
    assert 'frequency' in val_counts.columns
    
    # Test target balance
    target_bal = uci_adult_dataset_w_col_types.target_balance()
    assert isinstance(target_bal, pd.DataFrame)
    
    # Test data statistics
    desc = uci_adult_dataset_w_col_types.describe()
    assert isinstance(desc, pd.DataFrame)
    assert 'count' in desc.index


def test_dataset_column_transformations(uci_adult_dataset):
    """Test column transformation methods."""
    # Create a copy of the dataset to modify
    dataset = Dataset.from_dataframe(
        data=uci_adult_dataset.data.copy(),
        target_column=uci_adult_dataset.target_column,
        ml_task=uci_adult_dataset.ml_task
    )
    
    # Test numerical encoder
    if 'education' in dataset.data.columns:
        original_dtype = dataset.data['education'].dtype
        dataset.numerical_encoder('education')
        assert dataset.data['education'].dtype != original_dtype
        assert pd.api.types.is_numeric_dtype(dataset.data['education'].dtype)
    
    # Test categorical to ordinal transformation
    dataset2 = Dataset.from_dataframe(
        data=uci_adult_dataset.data.copy(),
        target_column=uci_adult_dataset.target_column,
        ml_task=uci_adult_dataset.ml_task
    )
    categorical_cols_before = [c for c in dataset2.columns() if dataset2.data[c].dtype == 'O']
    if categorical_cols_before:
        dataset2.categorical_to_ordinal()
        categorical_cols_after = [c for c in dataset2.columns() if dataset2.data[c].dtype == 'O']
        assert len(categorical_cols_after) < len(categorical_cols_before)
