import pytest
import pandas as pd
import numpy as np
from clearbox_synthetic.utils import Dataset
from .pytest_fixtures import (
    uci_adult_dataset,
    uci_adult_dataset_w_col_types,
    boston_housing_dataset
)

def test_dataset_creation(uci_adult_dataset):
    """Test basic dataset creation and properties"""
    assert isinstance(uci_adult_dataset, Dataset)
    assert isinstance(uci_adult_dataset.data, pd.DataFrame)
    assert uci_adult_dataset.target_column == "income"
    assert not uci_adult_dataset.regression

def test_dataset_methods(uci_adult_dataset):
    """Test various Dataset methods"""
    # Test get_x and get_y
    X = uci_adult_dataset.get_x()
    y = uci_adult_dataset.get_y()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "income" not in X.columns
    assert y.name == "income"

    # Test rows and columns methods
    assert uci_adult_dataset.rows_number() == len(uci_adult_dataset.data)
    assert uci_adult_dataset.columns_number() == len(uci_adult_dataset.data.columns)

    # Test value counts
    value_counts = uci_adult_dataset.value_counts("income")
    assert isinstance(value_counts, pd.DataFrame)
    assert "count" in value_counts.columns
    assert "freq" in value_counts.columns

def test_dataset_with_column_types(uci_adult_dataset_w_col_types):
    """Test dataset with explicitly specified column types"""
    assert isinstance(uci_adult_dataset_w_col_types.column_types, dict)
    assert "age" in uci_adult_dataset_w_col_types.column_types
    assert uci_adult_dataset_w_col_types.column_types["age"] == "number"
    assert uci_adult_dataset_w_col_types.column_types["work_class"] == "string"

def test_dataset_bounds(uci_adult_dataset):
    """Test dataset bounds functionality"""
    assert isinstance(uci_adult_dataset.bounds, dict)
    
    # Test numeric bounds
    assert "age" in uci_adult_dataset.bounds
    assert isinstance(uci_adult_dataset.bounds["age"], dict)
    assert "min" in uci_adult_dataset.bounds["age"]
    assert "max" in uci_adult_dataset.bounds["age"]
    
    # Test categorical bounds
    assert "work_class" in uci_adult_dataset.bounds
    assert isinstance(uci_adult_dataset.bounds["work_class"], set)

def test_dataset_regression(boston_housing_dataset):
    """Test dataset with regression target"""
    assert boston_housing_dataset.regression
    assert boston_housing_dataset.target_column == "PRICE"
    
    # Test normalized y values
    y_normalized = boston_housing_dataset.get_normalized_y()
    assert isinstance(y_normalized, np.ndarray)
