import logging
from typing import Tuple

import pandas as pd
from model.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)
from typing_extensions import Annotated
from zenml import step


@step
def clean_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Data cleaning step which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame - Raw input data

    Returns:
        x_train: pd.DataFrame - Training features
        x_test: pd.DataFrame - Testing features  
        y_train: pd.Series - Training target
        y_test: pd.Series - Testing target
    """
    try:
        logging.info("Starting data cleaning process...")
        
        # Step 1: Preprocess the data
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()
        
        logging.info(f"Data preprocessing completed. Shape: {preprocessed_data.shape}")
        
        # Step 2: Split the data into train and test sets
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        
        logging.info("Data cleaning and splitting completed successfully")
        return x_train, x_test, y_train, y_test
        
    except Exception as e:
        logging.error(f"Error in clean_data step: {e}")
        raise e