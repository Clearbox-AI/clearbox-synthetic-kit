# Preprocessor

The preprocessor module is embedded in the Engine classes.\ 
The Preprocessor module is a fast and flexible data manipulation component designed for preprocessing tabular and time-series data. By tuning the relevant parameters of the Engine classes it enables various transformations such as encoding categorical variables, handling missing values, feature selection, and numerical transformation, preparing the real dataset for the generation process.

## Key Functionalities
The Preprocessor class provides several preprocessing capabilities, including:

1. Handling Different Data Types
    - **Numerical Features**: Standardizes, normalizes, or discretizes numerical columns.
    - **Categorical Features**: Encodes categorical variables using one-hot encoding or embedding-based methods.
    - **Datetime Features**: Converts datetime columns into numerical representations.
2. Feature Engineering & Selection
    - **Feature Selection**: Identifies and removes non-informative features based on data statistics.
    - **Binning**: Discretizes numerical features into categorical bins for better processing.
    - **Categorical Embedding Rules**: Generates embedding rules for categorical features to enhance model performance.
3. Missing Value Handling
    - **Filling Missing Values**: Handles missing data by replacing with a default value (e.g., `-0.001` for numerical data or `NaN` for categorical data).
    - **Inference-Based Imputation**: Uses similarity-based methods to infer missing values.
4. Data Transformation
        - `NumericalTransformer`: Processes numerical values with various scaling and binning techniques.
        - `CategoricalTransformer`: Encodes categorical features.
        - `DatetimeTransformer`: Converts datetime data into useful features.
5. Time-Series Support
    - **Supports Sequential Data**: Handles time-indexed datasets and processes time-based features separately.
    - **Normalization & Scaling**: Normalizes time-series features based on computed means and standard deviations.
    - **Fixed Sequence Lengths**: Reshapes time-series data into sequences of fixed length for deep learning models.
6. Data Encoding & Decoding
    - **Transformation** (`transform`): Applies preprocessing steps to raw data and converts it into a machine-learning-friendly format.
    - **Reverse Transformation** (`inverse_transform`): Converts the processed data back to its original format, allowing interpretability.
