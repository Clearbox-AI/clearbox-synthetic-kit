# Utility evaluation
A synthetic dataset must accurately capture the statistical properties and patterns of the original data to remain useful for downstream tasks such as machine learning, analytics, and decision-making.

Assessing the utility of a synthetic dataset is crucial to ensure that it remains a viable alternative to real data. Poorly generated synthetic data can lead to biased models, incorrect predictions, and misleading insights, ultimately defeating its purpose. By evaluating utility, organizations can determine whether a synthetic dataset retains the necessary statistical integrity, predictive power, and representativeness required for their specific use cases.

In the following sections, we will explore methodologies for assessing synthetic data utility integrated in Clearbox Synthetic Kit, from statistical fidelity metrics to machine learning performance benchmarks, ensuring that synthetic data is both privacy-preserving and functionally robust.

## 1. Train on Synthetic, Test on Real
The Train on Synthetic, Test on Real (TSTR) Score is a widely used metric for evaluating the utility of synthetic data. It assesses how well a machine learning model trained on synthetic data generalizes to real-world data.

The TSTRScore class implements this metric using ``XGBoost`` models for both regression and classification tasks. It provides a performance comparison between models trained on original data and those trained on synthetic data, evaluating their effectiveness on a real validation dataset.

This approach is essential for understanding the quality and reliability of synthetic datasets in real-world applications.

#### 1.1 Key functionalities
1. Evaluates Model Performance on Synthetic vs. Real Data
    The module compares:
    - A model trained on real data vs. tested on real data.
    - A model trained on synthetic data vs. tested on real data.
    - The closer the results, the higher the quality of the synthetic dataset.

2. Computes Performance Metrics for Both Regression and Classification
    Regression Tasks:
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - Mean Absolute Error (MAE)
        - Max Error
        - R² Score (Goodness of fit)
    Classification Tasks:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - Support (Class distribution)

3. Provides a Unified TSTR Score
    The final score indicates how well synthetic data generalizes to real-world data.\
    If TSTR score is high (~1.0), it means synthetic data closely resembles real data.\
    A low TSTR score (~0.0) suggests poor synthetic data quality.

4. Computes Feature Importances
    Identifies which features contribute most to model predictions.\
    Helps analyze feature distribution differences between synthetic and real datasets.\
    Supports feature exclusion, allowing users to hide specific features from analysis.

5. Preprocesses Data for Consistency
    Applies feature transformations for structured comparison.\
    Ensures numerical and categorical features are properly formatted.\
    Supports label encoding for classification tasks.

## 2. Anomalies

The `Anomalies` module provides a systematic approach for detecting anomalies in tabular data. It leverages a combination of:

- Reconstruction error calculations (using an anomaly detection engine),
- Preprocessing transformations (to prepare data for detection),
- Statistical analysis of feature deviations.

This module is particularly useful in fraud detection, quality control, security monitoring, and AI model debugging, where identifying unusual or outlier data points is crucial.

The `Anomalies` class provides a structured framework for detecting anomalies in datasets. 

#### 2.1 Key functionalities
1. Detecting Anomalies Using Reconstruction Errors
    Uses the `TabularEngine` model to compute reconstruction errors with the assumption that higher reconstruction errors indicate higher anomaly likelihood. It thene extracts top-N most anomalous data points based on this error.
2. Feature Transformation for Anomaly Detection
    Preprocessing pipeline (via the `Preprocessor` class) ensures that data is normalized, encoded, and structured properly before anomaly detection, handling categorical, numerical, and missing values efficiently.
3. Probabilistic Anomaly Score Calculation
    A Numerical Anomaly Score is computed, using a Gaussian probability function to determine how far a numerical feature deviates from its reconstructed version.\
    The Categorical Anomaly Scorem instead, uses a comparison between input and reconstructed categorical values to determine discrepancies.
4. Configurable Detection Mechanism
    Users can specify the number of anomalies (n) to retrieve. The `Anomalies` class returns a dictionary containing feature values and computed anomaly probabilities.

## 3. Autocorrelation

The `Autocorrelation` module provides tools for computing and comparing the autocorrelation between an original dataset and a synthetic dataset. This functionality is essential in evaluating the temporal consistency of synthetic data and ensuring that the synthetic time series maintain the same time-dependent patterns as the real dataset.

#### 3.1 Key functionalities
1. Computes Autocorrelation of Time-Series Features
    It determines the autocorrelation function (ACF) for a given feature and normalizes results to ensure comparability.
2. Compares Autocorrelation Between Original and Synthetic Data
    It Computes autocorrelation curves for both datasets and measures how well the synthetic dataset preserves temporal dependencies.
3. Quantifies Differences Using Area Under the Curve (AUC)
    The module uses numerical integration to calculate the area under the autocorrelation curve. Then it computes the absolute difference (``diff_area``) between the original and synthetic AUC values to assess similarity.\
    The AUC metric provides a single numeric value to summarize the similarity between datasets:
    - If ``diff_area`` is close to 0, the synthetic dataset has high fidelity.
    - If ``diff_area`` is large, the synthetic dataset has lost important time-dependent structures.
4. Handles Sequence-Based Grouping
    It supports grouped time-series analysis for multivariate time series, where data is divided based on an identifier (``id``), ensuring correct indexing and alignment before computing autocorrelation.

## 4. Detection score
The `DetectionScore` module provides a powerful method to evaluate the quality of synthetic datasets by training a machine learning model to distinguish between real and synthetic data. The main idea is:
- If a classifier can easily separate synthetic data from real data, the synthetic dataset likely contains artifacts, biases, or unrealistic patterns.
- If the classifier struggles to differentiate between real and synthetic data (i.e., detection score is low), it suggests that the synthetic data is high quality and closely resembles the original dataset.

#### 4.1 Key functionalities
1. Evaluates the Realism of Synthetic Data
    The DetectionScore module plays a crucial role in evaluating the realism of synthetic datasets. It employs a binary classification approach to assess how easily a model can differentiate between real and synthetic data. If the classifier achieves high accuracy, it indicates that the synthetic dataset is easily distinguishable from real data, suggesting low quality. Conversely, if the accuracy is low, it implies that the synthetic dataset is indistinguishable from real data, which is a desirable outcome indicating high fidelity to the original data.
2. Computes Performance Metrics
    To quantify the quality of the synthetic dataset, the module computes several performance metrics. The Accuracy Score determines how often the classification model correctly labels instances as real or synthetic. The ROC-AUC Score evaluates the model’s ability to separate the two datasets, with values closer to 0.5 indicating that the synthetic dataset is highly realistic. The Detection Score is an inverse measure of dataset quality, where lower values indicate better similarity to real data.
3. Uses ``XGBoost`` as the Classifier
    At the core of the module is an ``XGBoost`` classifier, a powerful machine learning algorithm known for its efficiency in handling tabular data. If the classifier performs poorly (i.e., if its AUC is close to 0.5), it suggests that the synthetic dataset closely mimics real-world patterns, making it difficult to distinguish between real and synthetic records.
4. Feature Importance Analysis
    Additionally, the module provides feature importance analysis, which helps identify which features contribute the most to distinguishing real and synthetic data. This analysis is useful in pinpointing specific synthetic features that deviate from real-world distributions, enabling improvements in synthetic data generation.
5. Configurable Feature Exclusion
    To offer further flexibility, the module allows users to exclude specific features from the detection test. This capability is particularly useful when focusing on key attributes while avoiding bias from known sensitive fields that may otherwise disproportionately influence the classification process.

## 5. Features comparison
The `FeaturesComparison` module provides a structured approach to analyzing and comparing the statistical properties of features in an original dataset and a synthetic dataset. By evaluating differences in numerical, categorical, and datetime features, this module helps assess the accuracy and reliability of synthetic data generation methods.

This functionality is particularly useful in data quality assessment, machine learning model validation, and privacy-preserving synthetic data evaluation.

#### 5.1 Key functionalities
1. Statistical Comparison of Synthetic and Original Features
    The module enables a detailed statistical comparison of features between the real dataset and the synthetic dataset. By examining descriptive statistics, it provides insights into how well the synthetic dataset replicates the real data’s distribution.

    For numerical features, it computes statistics such as:
    - Mean and standard deviation to measure central tendency and spread.
    - Minimum and maximum values to detect potential outliers.
    - Quartiles to understand the overall distribution shape.
    
    For categorical features, it analyzes:
    - Unique values and their distribution.
    - Frequency counts to identify whether synthetic data retains similar categorical patterns.
    
    For datetime features, it evaluates:
    - Time range (min/max values) to ensure synthetic timestamps align with real-world patterns.
    - Most frequent timestamps, useful for detecting irregular synthetic time distributions.
    
2. Handling Numerical, Categorical, and Datetime Features
    The module ensures a comprehensive comparison by separately analyzing:
    - Numerical features (e.g., age, salary, temperature).
    - Categorical features (e.g., country, product type).
    - Datetime features (e.g., timestamps, event dates).

    Each type of feature is processed using specific statistical methods suited to its data type.

3. Identifies Missing Values and Data Completeness
    To assess data completeness, the module tracks missing values (NaN occurrences) in both the original and synthetic datasets. This ensures that the synthetic dataset does not introduce or remove missing values in a way that distorts the original distribution.

    For each feature type, it returns:
    - The number of missing values in both datasets.
    - A comparison of missingness patterns, which is critical for validating data imputation strategies in synthetic data generation.

4. Computes Feature-Level Differences
    By analyzing distributions and variability, the module helps identify:
    - Features where the synthetic dataset closely mimics the real data.
    - Features that diverge significantly, which may require further tuning in the synthetic data generation process.
    
    For categorical features, it computes:
    - The most frequent values in both datasets.
    - Frequency distributions of categorical values.
    
    For numerical features, it compares:
    - Spread (variance, standard deviation).
    - Central tendency (mean, median).
    - Outliers (min/max values, quartiles).

5. Supports Feature Exclusion for Focused Analysis
    Users can exclude specific features from the comparison to focus on key variables. This is useful in cases where:

    - Certain features contain sensitive information.
    - Some variables are not relevant to the comparison.

## 6. Mutual information

The `MutualInformation` module provides a method for evaluating the statistical similarity between an original dataset and a synthetic dataset by computing mutual information scores for feature pairs. This comparison helps determine how well the relationships between features are preserved in the synthetic dataset.

Mutual information measures how much knowing the value of one feature reduces the uncertainty of another, making it particularly useful for identifying dependencies between variables. By comparing mutual information matrices of original vs. synthetic data, this module provides insights into how well feature relationships are maintained in synthetic datasets.

#### 6.1 Key functionalities
1. Evaluates Mutual Information Between Features
    The module computes mutual information scores for feature pairs in both the original dataset and the synthetic dataset. Mutual information quantifies the dependency between two variables, helping assess whether synthetic data preserves important statistical relationships from the real dataset.

2. Computes a Difference Matrix for Feature Relationships
    The module constructs a difference correlation matrix, where each element represents the absolute difference in mutual information between the original and synthetic datasets.\
    The difference between these matrices helps identify how well feature dependencies are preserved. A higher difference indicates that the synthetic dataset fails to maintain the original feature relationships.

3. Calculates an Overall Similarity Score
    To summarize dataset similarity, the module computes a single score that represents how well the feature dependencies are retained in the synthetic dataset:
    - A high score (~1.0) indicates that the synthetic dataset maintains strong statistical relationships.
    - A low score (~0.0) suggests that feature dependencies are not well preserved.

4. Handles Missing and Categorical Data Effectively
    The module ensures robust handling of missing values (``NaN``) and categorical features:

    - Numerical values are binned into discrete ranges for comparison.
    - Categorical features are filled with default values before computing mutual information.

## 7. Query power
The `QueryPower` module evaluates the quality of a synthetic dataset by running randomized queries that compare it to the original dataset. The goal is to assess how closely the synthetic dataset mimics the statistical properties of the real data by checking whether queries return similar results from both datasets. If query results from the synthetic dataset match those from the original dataset, it indicates that statistical patterns are well-preserved.

#### 7.1 Key functionalities
1. Randomized Query-Based Evaluation of Synthetic Data
    The module runs randomly generated queries on both datasets and compares the number of matching records. If both datasets return similar results, it indicates that the synthetic dataset maintains the same statistical properties as the real data.

2. Evaluates Numerical and Categorical Feature Distributions
    Queries are designed to test both:
    - Numerical features using conditional operators (``<=``, ``>=``).
    - Categorical features using exact match or negation (``==``, ``!=``).

3. Computes a Query Power Score
    The module calculates an overall query power score, which represents the closeness of synthetic data to real data.
    - A high score (~1.0) suggests that the synthetic dataset closely mimics real-world distributions.
    - A low score (~0.0) indicates that the synthetic data fails to replicate the original dataset's patterns.

4. Supports Randomized Query Generation
    Queries are dynamically generated based on:

    - Random feature selection from both numerical and categorical variables.
    - Logical operations (``AND``) to combine multiple conditions.
    - Quantile-based thresholds for numerical features.

5. Handles Data Preprocessing and Feature Selection
    - Extracts numerical, categorical, and datetime features.
    - Removes datetime features, as they are typically not suitable for direct query-based evaluation.
    - Applies data transformations before querying.

## 8. Reconstruction error
The ReconstructionError module provides a method for evaluating the quality of a synthetic dataset by calculating the reconstruction error using a `TabularEngine`. Reconstruction error measures how well the synthetic data captures the patterns in the original dataset. If the error distribution of the synthetic dataset closely matches the original dataset, it suggests that the synthetic data faithfully represents the original data's structure.

#### 8..1 Key functionalities
1. Computes Reconstruction Error for Original and Synthetic Data
    The module calculates the reconstruction error for both the original dataset and the synthetic dataset using a TabularEngine. A low reconstruction error indicates that the dataset maintains the original feature relationships and distributions, while a high error suggests deviations or inconsistencies.

2. Generates Reconstruction Error Histograms
    The histograms generated help visualize the reconstruction error distributions of both datasets. If the synthetic dataset follows a similar reconstruction error distribution as the original dataset, it indicates high data fidelity, while significant differences in histograms indicate poor synthetic data quality.
3. Handles Regression and Classification Datasets
    For regression datasets, the module normalizes the target variable before computing reconstruction error.\
    For classification datasets, it one-hot encodes categorical target variables to ensure a proper comparison between original and synthetic datasets.
4. Uses a Preprocessing Pipeline for Feature Transformation
    The module applies feature transformations before computing reconstruction error. It ensures consistency between original and synthetic datasets by using a standardized preprocessing pipeline.