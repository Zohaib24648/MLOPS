import logging
import os
import sys
from typing import Tuple, cast

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from model.data_cleaning import DataCleaning, DataPreprocessStrategy, DataDivideStrategy
from steps.ingest_data import IngestData
from zenml.steps import step


@step
def data_test_prep_step():
    """Test the data cleaning process with the strategy pattern."""
    try:
        # Get raw data
        ingest_data = IngestData()
        df = ingest_data.get_data()
        
        # Apply preprocessing strategy
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        
        # Use the cast function to tell the type checker we know this returns a DataFrame
        preprocessed_data = cast(pd.DataFrame, data_cleaning.handle_data())
        
        # Check if derived features were created
        required_derived_features = ['season', 'repayment_days_ratio', 
                                     'previous_transaction_count', 'repayment_ethics', 
                                     'defaulted']
                                     
        for feature in required_derived_features:
            assert feature in preprocessed_data.columns, f"Derived feature {feature} is missing"
        
        # Test target variable in preprocessed data
        test_target_variable(preprocessed_data)
        
        # Test missing values imputation in preprocessed data
        test_missing_values_imputation(preprocessed_data)
        
        # Apply dividing strategy
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        
        # For the divide strategy, we know it returns a tuple of (X_train, X_test, y_train, y_test)
        result = data_cleaning.handle_data()
        
        # Explicitly unpack the tuple
        X_train, X_test, y_train, y_test = result
        
        # Check types
        assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
        assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame" 
        assert isinstance(y_train, pd.Series), "y_train should be a Series"
        assert isinstance(y_test, pd.Series), "y_test should be a Series"
        
        # Check train-test split ratio (should be roughly 80/20)
        total_rows = len(X_train) + len(X_test)
        assert 0.79 <= len(X_train) / total_rows <= 0.81, "Train-test split ratio should be approximately 80/20"
        
        # Check for data leakage between train and test sets
        check_data_leakage(X_train, X_test)
        
        logging.info("Data preparation test passed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        pytest.fail(str(e))


@step
def check_data_leakage(X_train, X_test):
    """Test if there is any data leakage between train and test sets."""
    try:
        assert (
            len(X_train.index.intersection(X_test.index)) == 0
        ), "There is data leakage between train and test sets."
        logging.info("Data Leakage test passed.")
    except Exception as e:
        logging.error(f"Data leakage test failed: {str(e)}")
        pytest.fail(str(e))


def test_target_variable(df):
    """Test the target variable 'defaulted' to ensure it contains boolean values."""
    try:
        assert 'defaulted' in df.columns, "Target variable 'defaulted' is missing from the dataset"
        
        # Check that defaulted is boolean (True/False) or binary (0/1)
        is_boolean = df['defaulted'].dtype == bool or df['defaulted'].isin([0, 1, True, False]).all()
        assert is_boolean, "Target variable 'defaulted' should contain only boolean or binary values"
        
        logging.info("Target variable test passed.")
    except Exception as e:
        logging.error(f"Target variable test failed: {str(e)}")
        pytest.fail(str(e))


def test_missing_values_imputation(df):
    """Test that critical columns don't have missing values after imputation."""
    try:
        # Critical columns that should not have missing values
        critical_columns = ['distributor_name_final', 'material_code_final', 'tenure', 'financing_amount']
        
        for col in critical_columns:
            if col in df.columns:
                assert df[col].isnull().sum() == 0, f"Column {col} still has missing values after imputation"
        
        logging.info("Missing values imputation test passed.")
    except Exception as e:
        logging.error(f"Missing values test failed: {str(e)}")
        pytest.fail(str(e))
