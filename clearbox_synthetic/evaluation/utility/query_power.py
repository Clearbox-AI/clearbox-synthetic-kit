"""
The QueryPower class assesses the quality of a synthetic dataset
by running randomized queries that compare it to the original dataset. The closer the query results
are between both datasets, the higher the quality of the synthetic data.
"""

import random
from clearbox_synthetic.utils.dataset.dataset import Dataset
from clearbox_synthetic.utils.preprocessor.preprocessor import Preprocessor


class QueryPower:
    """
    A class to evaluate the quality of a synthetic dataset by running comparative
    queries against the original dataset.

    Attributes
    ----------
    original_dataset : Dataset
        The original dataset containing real-world data.
    synthetic_dataset : Dataset
        The synthetic dataset generated for evaluation.
    preprocessor : Preprocessor
        The preprocessor responsible for handling data transformation.
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
        Initializes the QueryPower class with both original and synthetic datasets.

        Parameters
        ----------
        original_dataset : Dataset
            The original dataset containing real-world data.
        synthetic_dataset : Dataset
            The synthetic dataset generated for evaluation.
        preprocessor : Preprocessor, optional
            The preprocessor responsible for handling data transformation.
            If None, a default preprocessor based on the original dataset is used.
            Default is None.
        """
        self.original_dataset = original_dataset
        self.synthetic_dataset = synthetic_dataset
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor(original_dataset)
        )

    def get(self) -> dict:
        """
        Generates and runs randomized queries to compare the original and synthetic datasets, and calculates a query power score.

        This method creates random queries that filter data from both datasets.
        The similarity between the sizes of the filtered results is used to score
        the quality of the synthetic data.

        Returns
        -------
        dict 
            A dictionary containing query texts, the number of matches for each
            query in both datasets, and an overall score indicating the quality
            of the synthetic data.

        Notes
        -----

        The method operates through the following steps:

        1. Prepares the Data
            - Samples the original dataset to match the synthetic dataset size.
            - Applies preprocessing transformations and reverse transformations to ensure consistency.
        
        2. Extracts Feature Types
        - Identifies numerical, categorical, and datetime features.
        - Removes datetime features, which are not used for queries.
        
        3. Defines Query Components
        - Numerical feature queries use quantiles (<=, >=).
        - Categorical feature queries use equality (==) or inequality (!=).
        - Logical operators (AND) combine multiple conditions.
        
        4. Generates Up to 5 Random Queries
        - Randomly selects two features for each query.
        - Constructs query conditions based on the feature type.
        - Runs the same query on both datasets and counts matching records.
        
        5. Computes Query Score
        - Calculates differences in query results between the original and synthetic datasets.
        - Aggregates the query scores into a final query power score.

        > [!NOTE]
        > - The score represents the overall similarity between the datasets. A high score (close to 1.0) means that the synthetic dataset closely mimics real-world patterns.
        > - Queries are selected randomly and may involve numerical, categorical, 
        or logical conditions.

        Examples
        --------

        Example of dictionary returned:

        .. code-block:: python

            >>> dict
            {
                "queries": [
                    {"text": "`age` >= 35 and `gender` == 'Male'", "original_df": 320, "synthetic_df": 310},
                    {"text": "`income` <= 50000 and `education` != 'Bachelor’s'", "original_df": 280, "synthetic_df": 275}
                ],
                "score": 0.95
            }

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
