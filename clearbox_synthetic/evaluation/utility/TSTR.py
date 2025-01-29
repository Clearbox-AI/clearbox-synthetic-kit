"""
The TSTRScore class calculates the Train on Synthetic Test on Real (TSTR) score
for both regression and classification tasks using XGBoost models. The class compares the performance of
models trained on original and synthetic datasets.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    max_error,
)
from clearbox_synthetic.utils.dataset.dataset import Dataset
from clearbox_synthetic.utils.preprocessor.preprocessor import Preprocessor


class TSTRScore:
    """
    A class to calculate the Train on Synthetic Test on Real (TSTR) score using XGBoost models.

    Attributes
    ----------
    original_dataset : Dataset
        The original dataset containing real-world data.
    synthetic_dataset : Dataset
        The synthetic dataset generated for evaluation.
    validation_dataset : Dataset
        The validation dataset used for model testing.
    preprocessor : Preprocessor
        The preprocessor for handling data transformation. If not provided, a default 
        preprocessor based on the original dataset is used.
    """

    original_dataset: Dataset
    synthetic_dataset: Dataset
    preprocessor: Preprocessor

    def __init__(
        self,
        original_dataset: Dataset,
        synthetic_dataset: Dataset,
        validation_dataset: Dataset,
        preprocessor: Preprocessor = None,
    ) -> None:
        """
        Initializes the TSTRScore class with both original and synthetic datasets.

        Parameters
        ----------
        original_dataset : Dataset
            The original dataset containing real-world data.
        synthetic_dataset : Dataset
            The synthetic dataset generated for evaluation.
        validation_dataset : Dataset
            The validation dataset used for model testing.
        preprocessor : Preprocessor, optional
            The preprocessor for handling data transformation. If None, a default preprocessor
            based on the original dataset is used. Default is None.
        """
        self.original_dataset = original_dataset
        self.synthetic_dataset = synthetic_dataset
        self.validation_dataset = validation_dataset
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor(original_dataset)
        )

    def get(self, features_to_hide: list = []) -> dict:
        """
        Calculates the TSTR score by training XGBoost models on original and synthetic datasets
        and evaluating them on the validation dataset. The method returns metrics for either
        regression or classification tasks.

        Parameters
        ----------
        features_to_hide : list, optional
            List of features to exclude from the feature importance calculations. 
            Defaults to an empty list.

        Returns
        -------
        dict
            A dictionary containing TSTR metrics and feature importances.

        Notes
        -----
        The method operates through the following steps:

        1. Prepares the Data
            - Samples equal-sized subsets from original and synthetic datasets.
            - Applies feature transformations to ensure consistency.
            - Prepares training and testing datasets.

        2. Trains Models for Regression or Classification
            - Regression models (XGBRegressor) are trained on real and synthetic data.
            - Classification models (XGBClassifier) are trained using label-encoded targets.

        3. Computes Performance Metrics
            - Calculates MSE, RMSE, MAE, R² score for regression.
            - Computes accuracy, precision, recall, F1 score for classification.

        4. Compares Model Performance
            - The difference in performance between real and synthetic models is analyzed.
            - If performance on synthetic data is close to real data, the synthetic dataset is high quality.

        5. Computes TSTR Score
            - The score is calculated based on the difference between real and synthetic models' errors.
            - If TSTR score is high (~1.0), the synthetic dataset is very realistic.

        6. Determines Feature Importance
            - Extracts feature importance scores from both real and synthetic models.
            - Identifies which features most affect predictions.

        > [!NOTE]
        > A high TSTR score (>0.9) means that synthetic data is highly useful.
        > Lower TSTR scores indicate that synthetic data does not generalize well.

        Examples
        --------
        Example of dictionary returned for regression task:

        .. code-block:: python

            >>> dict
            {
                "task": "regression",
                "MSE": {"training": 0.25, "synthetic": 0.27},
                "RMSE": {"training": 0.5, "synthetic": 0.52},
                "MAE": {"training": 0.4, "synthetic": 0.42},
                "max_error": {"training": 1.2, "synthetic": 1.3},
                "r2_score": {"training": 0.85, "synthetic": 0.82},
                "score": 0.92,  # Final TSTR Score
                "feature_importances": {
                    "training": {"age": 0.12, "income": 0.08, "education": 0.15},
                    "synthetic": {"age": 0.11, "income": 0.09, "education": 0.14}
                }
            }
        
        """
        n_rows = min(
            self.original_dataset.get_x().shape[0],
            self.synthetic_dataset.get_x().shape[0],
        )

        # Preprocess datasets
        preprocessed_original_dataset = self.preprocessor.transform(
            self.original_dataset.get_x().head(n_rows)
        )
        preprocessed_synthetic_dataset = self.preprocessor.transform(
            self.synthetic_dataset.get_x().head(n_rows)
        )
        preprocessed_validation_dataset = self.preprocessor.transform(
            self.validation_dataset.get_x()
        )

        TSTR_score = {
            "feature_importances": {
                "training": {},
                "synthetic": {},
            }
        }

        if self.original_dataset.regression:
            from xgboost import XGBRegressor as xgb

            # Regression task
            TSTR_score["task"] = "regression"
            TSTR_score["MSE"] = {}
            TSTR_score["RMSE"] = {}
            TSTR_score["MAE"] = {}
            TSTR_score["max_error"] = {}
            TSTR_score["r2_score"] = {}

            training_Y = self.original_dataset.get_y().values[:n_rows]
            synthetic_Y = self.synthetic_dataset.get_y().values[:n_rows]

            # Train and evaluate model on original data
            model_original_data = xgb()
            model_original_data.fit(preprocessed_original_dataset, training_Y)

            predictions = model_original_data.predict(preprocessed_validation_dataset)
            TSTR_score["MSE"]["training"] = float(
                round(
                    mean_squared_error(
                        y_true=self.validation_dataset.get_y().values, y_pred=predictions
                    ),
                    4,
                )
            )
            TSTR_score["RMSE"]["training"] = float(
                round(
                    mean_squared_error(
                        y_true=self.validation_dataset.get_y().values,
                        y_pred=predictions,
                        squared=False,
                    ),
                    4,
                )
            )
            TSTR_score["MAE"]["training"] = float(
                round(
                    mean_absolute_error(
                        y_true=self.validation_dataset.get_y().values, y_pred=predictions
                    ),
                    4,
                )
            )
            TSTR_score["max_error"]["training"] = float(
                round(
                    max_error(
                        y_true=self.validation_dataset.get_y().values, y_pred=predictions
                    ),
                    4,
                )
            )
            TSTR_score["r2_score"]["training"] = float(
                round(
                    r2_score(
                        y_true=self.validation_dataset.get_y().values, y_pred=predictions
                    ),
                    4,
                )
            )

            # Train and evaluate model on synthetic data
            model_synthetic_data = xgb()
            model_synthetic_data.fit(preprocessed_synthetic_dataset, synthetic_Y)

            predictions = model_synthetic_data.predict(preprocessed_validation_dataset)
            TSTR_score["MSE"]["synthetic"] = float(
                round(
                    mean_squared_error(
                        y_true=self.validation_dataset.get_y().values, y_pred=predictions
                    ),
                    4,
                )
            )
            TSTR_score["RMSE"]["synthetic"] = float(
                round(
                    mean_squared_error(
                        y_true=self.validation_dataset.get_y().values,
                        y_pred=predictions,
                        squared=False,
                    ),
                    4,
                )
            )
            TSTR_score["MAE"]["synthetic"] = float(
                round(
                    mean_absolute_error(
                        y_true=self.validation_dataset.get_y().values, y_pred=predictions
                    ),
                    4,
                )
            )
            TSTR_score["max_error"]["synthetic"] = float(
                round(
                    max_error(
                        y_true=self.validation_dataset.get_y().values, y_pred=predictions
                    ),
                    4,
                )
            )
            TSTR_score["r2_score"]["synthetic"] = float(
                round(
                    r2_score(
                        y_true=self.validation_dataset.get_y().values, y_pred=predictions
                    ),
                    4,
                )
            )

            TSTR_score["score"] = round(
                float(
                    1
                    - (
                        abs(
                            TSTR_score["MAE"]["training"]
                            - TSTR_score["MAE"]["synthetic"]
                        )
                        / max(
                            TSTR_score["max_error"]["training"],
                            TSTR_score["max_error"]["synthetic"],
                        )
                    )
                ),
                4,
            )
        else:
            from xgboost import XGBClassifier as xgb

            # Classification task
            Y, Y_labels = self.validation_dataset.get_label_encoded_y()
            TSTR_score["task"] = "classification"
            TSTR_score["accuracy"] = {}
            TSTR_score["metrics"] = {"training": [], "synthetic": []}

            training_Y = self.original_dataset.get_y().astype("category").cat.codes
            training_Y = training_Y.values[:n_rows]

            synthetic_Y = self.synthetic_dataset.get_y().astype("category").cat.codes
            synthetic_Y = synthetic_Y.values[:n_rows]

            y_true = self.validation_dataset.get_y().astype("category").cat.codes
            y_true = y_true.values[:n_rows]

            # Train and evaluate model on original data
            model_original_data = xgb(eval_metric="logloss")
            model_original_data.fit(preprocessed_original_dataset, training_Y)

            predictions = model_original_data.predict(preprocessed_validation_dataset)
            TSTR_score["accuracy"]["training"] = float(
                round(accuracy_score(y_true=y_true, y_pred=predictions), 4)
            )
            precisions, recalls, fscores, supports = precision_recall_fscore_support(
                y_true, predictions
            )
            for label in np.unique(Y):
                TSTR_score["metrics"]["training"].append(
                    {
                        "label": str(Y_labels[label]),
                        "precision": float(round(precisions[label], 4)),
                        "recall": float(round(recalls[label], 4)),
                        "fscore": float(round(fscores[label], 4)),
                        "support": float(round(supports[label], 4)),
                    }
                )

            # Train and evaluate model on synthetic data
            model_synthetic_data = xgb()
            model_synthetic_data.fit(preprocessed_synthetic_dataset, synthetic_Y)

            predictions = model_synthetic_data.predict(preprocessed_validation_dataset)
            TSTR_score["accuracy"]["synthetic"] = round(
                accuracy_score(y_true=y_true, y_pred=predictions), 4
            )
            precisions, recalls, fscores, supports = precision_recall_fscore_support(
                y_true, predictions
            )
            for label in np.unique(Y):
                TSTR_score["metrics"]["synthetic"].append(
                    {
                        "label": str(Y_labels[label]),
                        "precision": float(round(precisions[label], 4)),
                        "recall": float(round(recalls[label], 4)),
                        "fscore": float(round(fscores[label], 4)),
                        "support": float(round(supports[label], 4)),
                    }
                )

            TSTR_score["score"] = round(
                float(
                    1
                    - abs(
                        TSTR_score["accuracy"]["training"]
                        - TSTR_score["accuracy"]["synthetic"]
                    )
                ),
                4,
            )

        # Feature importances
        numerical_features_sizes, categorical_features_sizes = (
            self.preprocessor.get_features_sizes()
        )
        preprocessed_numerical_features = []
        preprocessed_categorical_features = []
        preprocessed_datetime_features = []

        if numerical_features_sizes:
            preprocessed_numerical_features = self.preprocessor.transformer.transformers[
                0
            ][2]
            if self.preprocessor.get_datetime_features():
                preprocessed_datetime_features = (
                    self.preprocessor.transformer.transformers[1][2]
                )
                if categorical_features_sizes:
                    preprocessed_categorical_features = (
                        self.preprocessor.transformer.transformers[2][2]
                    )
            else:
                if categorical_features_sizes:
                    preprocessed_categorical_features = (
                        self.preprocessor.transformer.transformers[1][2]
                    )
        else:
            if self.preprocessor.get_datetime_features():
                preprocessed_datetime_features = (
                    self.preprocessor.transformer.transformers[0][2]
                )
                if categorical_features_sizes:
                    preprocessed_categorical_features = (
                        self.preprocessor.transformer.transformers[1][2]
                    )
            else:
                if categorical_features_sizes:
                    preprocessed_categorical_features = (
                        self.preprocessor.transformer.transformers[0][2]
                    )

        # Calculate feature importances for both training and synthetic models
        index = 0
        for feature, importance in zip(
            preprocessed_numerical_features, model_original_data.feature_importances_
        ):
            if feature not in features_to_hide:
                TSTR_score["feature_importances"]["training"][feature] = round(
                    float(importance), 4
                )
            index += 1

        if preprocessed_datetime_features:
            for feature, importance in zip(
                preprocessed_datetime_features,
                model_original_data.feature_importances_[index:],
            ):
                if feature not in features_to_hide:
                    TSTR_score["feature_importances"]["training"][feature] = round(
                        float(importance), 4
                    )
                index += 1

        for feature, feature_size in zip(
            preprocessed_categorical_features, categorical_features_sizes
        ):
            importance = np.sum(
                model_original_data.feature_importances_[index : index + feature_size]
            )
            index += feature_size
            if feature not in features_to_hide:
                TSTR_score["feature_importances"]["training"][feature] = round(
                    float(importance), 4
                )

        index = 0
        for feature, importance in zip(
            preprocessed_numerical_features, model_synthetic_data.feature_importances_
        ):
            if feature not in features_to_hide:
                TSTR_score["feature_importances"]["synthetic"][feature] = round(
                    float(importance), 4
                )
            index += 1

        if preprocessed_datetime_features:
            for feature, importance in zip(
                preprocessed_datetime_features,
                model_synthetic_data.feature_importances_[index:],
            ):
                if feature not in features_to_hide:
                    TSTR_score["feature_importances"]["synthetic"][feature] = round(
                        float(importance), 4
                    )
                index += 1

        for feature, feature_size in zip(
            preprocessed_categorical_features, categorical_features_sizes
        ):
            importance = np.sum(
                model_synthetic_data.feature_importances_[index : index + feature_size]
            )
            index += feature_size
            if feature not in features_to_hide:
                TSTR_score["feature_importances"]["synthetic"][feature] = round(
                    float(importance), 4
                )

        return TSTR_score
