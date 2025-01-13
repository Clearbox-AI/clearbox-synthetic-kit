"""
This module defines the QueryPower class, which assesses the quality of a synthetic dataset
by running queries that compare it to the original dataset. The closer the query results
are between both datasets, the higher the quality of the synthetic data.
"""

import random
from clearbox_engine import Dataset, Preprocessor


class QueryPower:
    """
    A class to evaluate the quality of a synthetic dataset by running comparative
    queries against the original dataset.

    Attributes:
        original_dataset (Dataset): The original dataset.
        synthetic_dataset (Dataset): The synthetic dataset.
        preprocessor (Preprocessor): The preprocessor for handling data transformation.
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
        Initializes the QueryPower class.

        Args:
            original_dataset (Dataset): The original dataset object.
            synthetic_dataset (Dataset): The synthetic dataset object.
            preprocessor (Preprocessor, optional): The preprocessor for data transformation.
                                                   Defaults to None, using a default 
                                                   preprocessor for the original dataset.
        """
        self.original_dataset = original_dataset
        self.synthetic_dataset = synthetic_dataset
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor(original_dataset)
        )

    def get(self) -> dict:
        """
        Generates and runs queries to compare the original and synthetic datasets.

        This method creates random queries that filter data from both datasets.
        The similarity between the sizes of the filtered results is used to score
        the quality of the synthetic data.

        Returns:
            dict: A dictionary containing query texts, the number of matches for each
                  query in both datasets, and an overall score indicating the quality
                  of the synthetic data.
        """
        query_power = {"queries": []}

        # Sample data from the original and synthetic datasets
        original_df = self.original_dataset.data.sample(
            n=len(self.synthetic_dataset.data)
        ).copy()
        preprocessed_original_df = self.preprocessor.transform(original_df)
        original_df = self.preprocessor.reverse_transform(preprocessed_original_df)

        synthetic_df = self.synthetic_dataset.data.copy()
        preprocessed_synthetic_df = self.preprocessor.transform(synthetic_df)
        synthetic_df = self.preprocessor.reverse_transform(preprocessed_synthetic_df)

        # Extract feature types
        numerical_features = self.preprocessor.get_numerical_features()
        categorical_features = self.preprocessor.get_categorical_features()
        datetime_features = self.preprocessor.get_datetime_features()

        # Prepare the feature list, excluding datetime features
        features = list(set(original_df.columns.tolist()) - set(datetime_features))

        # Define query parameters
        quantiles = [0.25, 0.5, 0.75]
        numerical_ops = ["<=", ">="]
        categorical_ops = ["==", "!="]
        logical_ops = ["and"]

        queries_score = []

        # Generate and run up to 5 queries
        while len(features) >= 2 and len(query_power["queries"]) < 5:
            # Randomly select two features for the query
            feats = [random.choice(features)]
            features.remove(feats[0])
            feats.append(random.choice(features))
            features.remove(feats[1])

            queries = []

            # Construct query conditions for each selected feature
            for feature in feats:
                if feature in numerical_features:
                    op = random.choice(numerical_ops)
                    value = original_df.quantile(
                        q=random.choice(quantiles), numeric_only=True
                    )[feature]
                elif feature in categorical_features:
                    op = random.choice(categorical_ops)
                    value = random.choice(original_df[feature].unique())
                    value = f"'{value}'"

                queries.append(f"`{feature}` {op} {value}")

            # Combine query conditions with a logical operator
            text = f" {random.choice(logical_ops)} ".join(queries)
            try:
                query = {
                    "text": text,
                    "original_df": len(original_df.query(text)),
                    "synthetic_df": len(synthetic_df.query(text)),
                }
            except Exception:
                query = {"text": "Invalid query", "original_df": 0, "synthetic_df": 0}

            # Append the query and calculate the score
            query_power["queries"].append(query)
            queries_score.append(
                1 - abs(query["original_df"] - query["synthetic_df"]) / len(original_df)
            )

        # Calculate the overall query power score
        query_power["score"] = round(float(sum(queries_score) / len(queries_score)), 4)

        return query_power
