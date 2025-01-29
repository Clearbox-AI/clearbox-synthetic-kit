"""
The DetectionScore class computes a detection score
for evaluating the quality of a synthetic dataset by training a classifier to 
distinguish between original and synthetic data.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from clearbox_synthetic.utils.dataset.dataset import Dataset
from clearbox_synthetic.utils.preprocessor.preprocessor import Preprocessor


class DetectionScore:
    """
    The DetectionScore class computes a detection score
    for evaluating the quality of a synthetic dataset by training a classifier to 
    distinguish between original and synthetic data.

    Attributes
    ----------
    original_dataset : Dataset
        The original dataset containing real-world data.
    synthetic_dataset : Dataset
        The synthetic dataset generated for evaluation.
    preprocessor : Preprocessor
        The preprocessor responsible for handling feature extraction and transformation.

    Methods
    -------
    get(features_to_hide: list = []) -> dict
        Trains a classifier to differentiate between real and synthetic data and computes 
        the detection score, accuracy, ROC-AUC, and feature importance.
    """

    original_dataset: Dataset
    synthetic_dataset: Dataset
    preprocessor: Preprocessor

    def __init__(
        self,
        original_dataset: Dataset,
        synthetic_dataset: Dataset,
        preprocessor: Preprocessor = None,
    ) -> None:
        """
        Initializes the DetectionScore class with both original and synthetic datasets.

        Parameters
        ----------
        original_dataset : Dataset
            The original dataset containing real-world data.
        synthetic_dataset : Dataset
            The synthetic dataset generated for evaluation.
        preprocessor : Preprocessor, optional
            The preprocessor responsible for handling feature extraction and transformation.
            If None, a default preprocessor is used. Default is None.
        """
        self.original_dataset = original_dataset
        self.synthetic_dataset = synthetic_dataset
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor(original_dataset)
        )

    def get(self, features_to_hide: list = []) -> dict:
        """
        Computes the detection score by training an XGBoost model to differentiate between
        original and synthetic data. The lower the model's accuracy, the higher the quality
        of the synthetic data.

        Parameters
        ----------
        features_to_hide : list, optional
            List of features to exclude from importance analysis. Defaults to an empty list.

        Returns
        -------
        dict
            A dictionary containing accuracy, ROC AUC score, detection score, and 
                    feature importances.

        Notes
        -----
        These are the operational steps of the method:

        1. Prepares the datasets:
            - Samples the original dataset to match the size of the synthetic dataset.
            - Preprocesses both datasets to ensure consistent feature representation.
            - Labels original data as `0` and synthetic data as `1`.
        
        2. Builds a classification model:
            - Uses XGBoost to train a model that classifies data points as either real or synthetic.
            - Splits the data into training and test sets.
            - Trains the model using a 33% test split.

        3. Computes Evaluation Metrics:
            - Accuracy: Measures classification correctness.
            - ROC-AUC Score: Measures the model’s discriminatory power.
            - Detection Score
            
        4. Extracts Feature Importances:
            - Identifies which features contribute most to distinguishing real vs. synthetic data.
            - Helps detect which synthetic features deviate from real-world patterns.

        
        The detection score is calculated as:

        .. code-block:: python

            detection_score["score"] = (1 - detection_score["ROC_AUC"]) * 2

        - If ``ROC_AUC <= 0.5``, the synthetic data is considered indistinguishable (``score = 1``).
        - A lower score means better synthetic data quality.
        - Feature importance analysis helps detect which synthetic features deviate most from real data.

        Example of dictionary returned:

        .. code-block:: python

            >>> detection_results = detection_score.get()
            >>> print(detection_results)
            {
                "accuracy": 0.85,   # How often the model classifies correctly
                "ROC_AUC": 0.90,   # The ability to distinguish real vs synthetic
                "score": 0.2,     # The final detection score (lower = better)
                "feature_importances": {"feature_1": 0.34, "feature_2": 0.21, ...} 
            }

        """
        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        # Sample from the original dataset to match the size of the synthetic dataset
        original_df = self.original_dataset.data.sample(
            n=len(self.synthetic_dataset.data)
        ).copy()

        # Replace minority labels in the original data
        for i in self.preprocessor.discarded[1].keys():
            list_minority_labels = self.preprocessor.discarded[1][i]
            for j in list_minority_labels:
                original_df[i] = original_df[i].replace(j, '*')

        # Preprocess and label the original data
        preprocessed_original_df = self.preprocessor.transform(original_df)
        original_df = self.preprocessor.reverse_transform(preprocessed_original_df)
        original_df["label"] = np.zeros(len(original_df)).astype(int)

        # Preprocess and label the synthetic data
        synthetic_df = self.synthetic_dataset.data.copy()
        preprocessed_synthetic_df = self.preprocessor.transform(synthetic_df)
        synthetic_df = self.preprocessor.reverse_transform(preprocessed_synthetic_df)
        synthetic_df["label"] = np.ones(len(synthetic_df)).astype(int)

        # Create a combined dataset with labels
        column_types = (
            self.original_dataset.column_types.copy()
            if self.original_dataset.column_types
            else None
        )
        if column_types:
            cols = [col for col in column_types]
            for col in cols:
                if col not in original_df.columns.tolist():
                    column_types.pop(col)
            column_types["label"] = "number"

        dataset = Dataset(
            data=pd.concat([original_df, synthetic_df])
            .sample(frac=1)
            .reset_index(drop=True),
            target_column="label",
            regression=False,
            column_types=column_types,
        )

        preprocessor = Preprocessor(dataset)

        # Prepare training and testing data
        Y = dataset.get_y().values
        seed = 42
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.get_x(), Y, test_size=test_size, random_state=seed
        )

        train_ds = preprocessor.transform(X_train)
        test_ds = preprocessor.transform(X_test)

        # Train an XGBoost model
        model = xgb.XGBClassifier(
            max_depth=3, n_estimators=50, use_label_encoder=False, eval_metric="logloss"
        )
        model.fit(train_ds, y_train)

        # Make predictions and compute metrics
        y_pred = model.predict(test_ds)
        detection_score = {}
        detection_score["accuracy"] = round(
            accuracy_score(y_true=y_test, y_pred=y_pred), 4
        )
        detection_score["ROC_AUC"] = round(
            roc_auc_score(
                y_true=y_test,
                y_score=model.predict_proba(test_ds)[:, 1],
                average=None,
            ),
            4,
        )
        detection_score["score"] = (
            1
            if detection_score["ROC_AUC"] <= 0.5
            else (1 - detection_score["ROC_AUC"]) * 2
        )

        detection_score["feature_importances"] = {}

        # Determine feature importances
        numerical_features_sizes, categorical_features_sizes = preprocessor.get_features_sizes()
        preprocessed_numerical_features = []
        preprocessed_categorical_features = []
        preprocessed_datetime_features = []

        if numerical_features_sizes:
            preprocessed_numerical_features = preprocessor.transformer.transformers[0][2]
            if preprocessor.get_datetime_features():
                preprocessed_datetime_features = preprocessor.transformer.transformers[1][2]
                if categorical_features_sizes:
                    preprocessed_categorical_features = preprocessor.transformer.transformers[2][2]
            else:
                if categorical_features_sizes:
                    preprocessed_categorical_features = preprocessor.transformer.transformers[1][2]
        else:
            if preprocessor.get_datetime_features():
                preprocessed_datetime_features = preprocessor.transformer.transformers[0][2]
                if categorical_features_sizes:
                    preprocessed_categorical_features = preprocessor.transformer.transformers[1][2]
            else:
                if categorical_features_sizes:
                    preprocessed_categorical_features = preprocessor.transformer.transformers[0][2]

        index = 0
        for feature, importance in zip(preprocessed_numerical_features, model.feature_importances_):
            if feature not in features_to_hide:
                detection_score["feature_importances"][feature] = round(float(importance), 4)
            index += 1

        if preprocessed_datetime_features:
            for feature, importance in zip(preprocessed_datetime_features, model.feature_importances_[index:]):
                if feature not in features_to_hide:
                    detection_score["feature_importances"][feature] = round(float(importance), 4)
                index += 1

        for feature, feature_size in zip(preprocessed_categorical_features, categorical_features_sizes):
            importance = np.sum(model.feature_importances_[index : index + feature_size])
            index += feature_size
            if feature not in features_to_hide:
                detection_score["feature_importances"][feature] = round(float(importance), 4)

        return detection_score
