import pytest
import numpy as np
from clearbox_synthetic.utils import Preprocessor
from .pytest_fixtures import (
    uci_adult_dataset,
    boston_housing_dataset
)

def test_preprocessor_initialization(uci_adult_dataset):
    """Test Preprocessor initialization"""
    preprocessor = Preprocessor(uci_adult_dataset)
    assert preprocessor is not None
    assert hasattr(preprocessor, 'transformer')
    assert hasattr(preprocessor, 'numerical_features')
    assert hasattr(preprocessor, 'categorical_features')

def test_preprocessor_feature_inference(uci_adult_dataset):
    """Test feature type inference"""
    preprocessor = Preprocessor(uci_adult_dataset)
    
    # Check that numerical and categorical features were correctly identified
    assert len(preprocessor.numerical_features) > 0
    assert len(preprocessor.categorical_features) > 0
    assert "age" in preprocessor.numerical_features
    assert "work_class" in preprocessor.categorical_features

def test_preprocessor_transform(uci_adult_dataset):
    """Test data transformation"""
    preprocessor = Preprocessor(uci_adult_dataset)
    X_raw = uci_adult_dataset.get_x()
    X_transformed = preprocessor.transform(X_raw)
    
    # Check that output is numpy array
    assert isinstance(X_transformed, np.ndarray)
    
    # Check that all values are numeric
    assert np.issubdtype(X_transformed.dtype, np.number)
    
    # Check that there are no missing values
    assert not np.isnan(X_transformed).any()

def test_preprocessor_feature_sizes(uci_adult_dataset):
    """Test getting feature sizes"""
    preprocessor = Preprocessor(uci_adult_dataset)
    num_size, cat_sizes = preprocessor.get_features_sizes()
    
    # Check that sizes are lists of integers
    assert isinstance(num_size, list)
    assert isinstance(cat_sizes, list)
    assert all(isinstance(x, int) for x in num_size)
    assert all(isinstance(x, int) for x in cat_sizes)
    
    # Check that sizes match number of features
    assert num_size[0] == len(preprocessor.numerical_features)
    assert len(cat_sizes) == len(preprocessor.categorical_features)

def test_preprocessor_with_threshold(uci_adult_dataset):
    """Test preprocessor with different threshold values"""
    # Test with high threshold to force feature selection
    preprocessor_high_threshold = Preprocessor(uci_adult_dataset, threshold=0.9)
    
    # Test with low threshold to keep most features
    preprocessor_low_threshold = Preprocessor(uci_adult_dataset, threshold=0.01)
    
    # High threshold should select fewer features than low threshold
    assert len(preprocessor_high_threshold.numerical_features) <= len(preprocessor_low_threshold.numerical_features)
    assert len(preprocessor_high_threshold.categorical_features) <= len(preprocessor_low_threshold.categorical_features)

def test_preprocessor_with_numerical_bins(uci_adult_dataset):
    """Test preprocessor with numerical binning"""
    n_bins = 5
    preprocessor = Preprocessor(uci_adult_dataset, n_numerical_bins=n_bins)
    X_raw = uci_adult_dataset.get_x()
    X_transformed = preprocessor.transform(X_raw)
    
    # Check that numerical features were binned
    assert X_transformed is not None
    assert isinstance(X_transformed, np.ndarray)

#def test_preprocessor_with_rules(uci_adult_dataset):
#    """Test preprocessor with custom rules"""
#    rules = {
#        "age": ["embed_category", 10, 5],  # Example rule for age column
#    }
#    preprocessor = Preprocessor(uci_adult_dataset, rules=rules)
#    X_raw = uci_adult_dataset.get_x()
#    X_transformed = preprocessor.transform(X_raw)
#    assert X_transformed is not None
#    assert isinstance(X_transformed, np.ndarray)

def test_preprocessor_with_regression_dataset(boston_housing_dataset):
    """Test preprocessor with regression dataset"""
    preprocessor = Preprocessor(boston_housing_dataset)
    X_raw = boston_housing_dataset.get_x()
    X_transformed = preprocessor.transform(X_raw)
    
    assert X_transformed is not None
    assert isinstance(X_transformed, np.ndarray)
    assert np.issubdtype(X_transformed.dtype, np.number)