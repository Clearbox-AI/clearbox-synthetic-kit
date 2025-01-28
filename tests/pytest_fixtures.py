import pytest
from clearbox_synthetic.utils import Dataset
# TODO: Add fixtures for dataset with boolean columns, datetime columns, only categorical columns

# --------------------------AMES--------------------------------------------------------#


@pytest.fixture
def ames_dataset() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/ames_dataset/dataset.csv",
        regression=True,
        target_column="SalePrice",
    )

# --------------------------SYNTHETIC AMES--------------------------------------------------------#


@pytest.fixture
def ames_synthetic_dataset() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/ames_dataset/synthetic_dataset.csv",
        regression=True,
        target_column="SalePrice",
    )

# -------------------------AUTO_IMPORTS-------------------------------------------------#


@pytest.fixture
def auto_imports_dataset() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/auto_imports_dataset/dataset.csv",
        regression=True,
        target_column="price",
    )

# -------------------------SYNTHETIC AUTO_IMPORTS-------------------------------------------------#


@pytest.fixture
def auto_imports_synthetic_dataset() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/auto_imports_dataset/synthetic_dataset.csv",
        regression=True,
        target_column="price",
    )

# ------------------------AUTO_IMPORTS_VALIDATION--------------------------------------#


@pytest.fixture
def auto_imports_validation() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/auto_imports_dataset/validation.csv",
        regression=True,
        target_column="price",
    )


# ------------------BOSTON_HOUSING-------------------------------------------#


@pytest.fixture
def boston_housing_dataset() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/boston_housing_dataset/dataset.csv",
        regression=True,
        target_column="PRICE",
    )

# ------------------SYNTHETIC BOSTON_HOUSING-------------------------------------------#


@pytest.fixture
def boston_housing_synthetic_dataset() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/boston_housing_dataset/synthetic_dataset.csv",
        regression=True,
        target_column="PRICE",
    )

# ------------------HOSPITAL_READMISSION_DATASET---------------------------#


@pytest.fixture
def hospital_readmission_dataset() -> Dataset:
    return Dataset.from_csv("tests/resources/hospital_readmission_dataset/dataset.csv")


# ------------------HOSPITAL_READMISSION_VALIDATION---------------------------#


@pytest.fixture
def hospital_readmission_validation() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/hospital_readmission_dataset/validation.csv"
    )


# --------------------------GLASS_DATASET-------------------------------------#


@pytest.fixture
def glass_dataset() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/glass_dataset/dataset.csv", target_column="Type"
    )


# --------------------------GLASS_DATASET_VALIDATION-------------------------------------#


@pytest.fixture
def glass_validation() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/glass_dataset/validation_dataset.csv", target_column="Type"
    )


# ------------------------------UCI_ADULT------------------------------------#


@pytest.fixture
def uci_adult_dataset() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/uci_adult_dataset/dataset.csv", target_column="income"
    )


# ------------------------------SYNTHETIC UCI_ADULT------------------------------------#


@pytest.fixture
def uci_adult_synthetic_dataset() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/uci_adult_dataset/synthetic_dataset.csv", target_column="income"
    )



# ------------------------------UCI_ADULT_W_COL_TYPES------------------------------------#


@pytest.fixture
def uci_adult_dataset_w_col_types() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/uci_adult_dataset/dataset.csv",
        target_column="income",
        column_types={
            "age": "number",
            "work_class": "string",
            "education": "string",
            "marital_status": "string",
            "occupation": "string",
            "relationship": "string",
            "race": "string",
            "sex": "string",
            "capital_gain": "number",
            "capital_loss": "number",
            "hours_per_week": "number",
            "native_country": "string",
            "income": "string",
        },
    )


# ------------------------------UCI_ADULT_VALIDATION------------------------------------#


@pytest.fixture
def uci_adult_validation() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/uci_adult_dataset/validation_dataset.csv",
        target_column="income",
    )


# -----------------------------HEPATITIS---------------------------------------------#


@pytest.fixture
def hepatitis_dataset() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/hepatitis_dataset/dataset.csv", target_column="class"
    )

# -----------------------------SYNTHETIC HEPATITIS---------------------------------------------#


@pytest.fixture
def hepatitis_synthetic_dataset() -> Dataset:
    return Dataset.from_csv(
        "tests/resources/hepatitis_dataset/synthetic_dataset.csv", target_column="class"
    )
