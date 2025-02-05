import pytest
import numpy as np
from clearbox_synthetic.utils import Dataset, Preprocessor
from clearbox_synthetic.generation import TabularEngine
from .pytest_fixtures import (
    uci_adult_dataset,
    boston_housing_dataset
)

@pytest.fixture
def preprocessed_data(uci_adult_dataset):
    """Fixture to provide preprocessed data for testing"""
    preprocessor = Preprocessor(uci_adult_dataset)
    X_raw = uci_adult_dataset.get_x()
    X = preprocessor.transform(X_raw)
    Y = uci_adult_dataset.get_one_hot_encoded_y()
    return X, Y, preprocessor

def test_tabular_engine_initialization(preprocessed_data):
    """Test TabularEngine initialization"""
    X, Y, preprocessor = preprocessed_data
    
    engine = TabularEngine(
        layers_size=[50],
        x_shape=X[0].shape,
        y_shape=Y[0].shape,
        numerical_feature_sizes=preprocessor.get_features_sizes()[0],
        categorical_feature_sizes=preprocessor.get_features_sizes()[1],
    )
    
    assert engine is not None
    assert hasattr(engine, 'model')
    assert hasattr(engine, 'params')
    assert hasattr(engine, 'architecture')

def test_tabular_engine_fit(preprocessed_data):
    """Test TabularEngine fitting"""
    X, Y, preprocessor = preprocessed_data
    
    engine = TabularEngine(
        layers_size=[50],
        x_shape=X[0].shape,
        y_shape=Y[0].shape,
        numerical_feature_sizes=preprocessor.get_features_sizes()[0],
        categorical_feature_sizes=preprocessor.get_features_sizes()[1],
    )
    
    # Fit with small number of epochs for testing
    engine.fit(X, y_train_ds=Y, epochs=2, batch_size=32, learning_rate=0.001)
    
    assert engine.train_loss is not None
    assert isinstance(engine.train_loss, dict)
    assert 'loss' in engine.train_loss

def test_tabular_engine_encode(preprocessed_data):
    """Test encoding functionality"""
    X, Y, preprocessor = preprocessed_data
    
    engine = TabularEngine(
        layers_size=[50],
        x_shape=X[0].shape,
        y_shape=Y[0].shape,
        numerical_feature_sizes=preprocessor.get_features_sizes()[0],
        categorical_feature_sizes=preprocessor.get_features_sizes()[1],
    )
    
    # Fit with minimal epochs
    engine.fit(X, y_train_ds=Y, epochs=1)
    
    # Test encoding
    encoded = engine.encode(X[:10], Y[:10])
    assert len(encoded[0].shape) == 2  # Should be 2D array

def test_tabular_engine_with_validation(preprocessed_data):
    """Test TabularEngine with validation data"""
    X, Y, preprocessor = preprocessed_data
    
    # Split data into train and validation
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]
    
    engine = TabularEngine(
        layers_size=[50],
        x_shape=X_train[0].shape,
        y_shape=Y_train[0].shape,
        numerical_feature_sizes=preprocessor.get_features_sizes()[0],
        categorical_feature_sizes=preprocessor.get_features_sizes()[1],
    )
    
    # Fit with validation data
    engine.fit(
        X_train, 
        y_train_ds=Y_train,
        epochs=2,
        val_ds=X_val,
        y_val_ds=Y_val
    )
    
    assert engine.val_loss is not None
    assert isinstance(engine.val_loss, dict)
    assert 'loss' in engine.val_loss

def test_tabular_engine_with_different_architectures(preprocessed_data):
    """Test TabularEngine with different architectures"""
    X, Y, preprocessor = preprocessed_data
    
    # Test with different layer sizes
    architectures = [
        [20],
        [50, 25],
        [100, 50, 25]
    ]
    
    for layers in architectures:
        engine = TabularEngine(
            layers_size=layers,
            x_shape=X[0].shape,
            y_shape=Y[0].shape,
            numerical_feature_sizes=preprocessor.get_features_sizes()[0],
            categorical_feature_sizes=preprocessor.get_features_sizes()[1],
        )
        
        # Quick fit to ensure it works
        engine.fit(X, y_train_ds=Y, epochs=1)
        assert engine.train_loss is not None

def test_tabular_engine_with_regression(boston_housing_dataset):
    """Test TabularEngine with regression dataset"""
    # Prepare regression data
    preprocessor = Preprocessor(boston_housing_dataset)
    X_raw = boston_housing_dataset.get_x()
    X = preprocessor.transform(X_raw)
    Y = boston_housing_dataset.get_normalized_y()
    
    engine = TabularEngine(
        layers_size=[50],
        x_shape=X[0].shape,
        y_shape=Y[0].shape if len(Y.shape) > 1 else (1,),
        numerical_feature_sizes=preprocessor.get_features_sizes()[0],
        categorical_feature_sizes=preprocessor.get_features_sizes()[1],
    )
    
    # Quick fit to ensure it works with regression data
    engine.fit(X, y_train_ds=Y, epochs=1)
    assert engine.train_loss is not None 