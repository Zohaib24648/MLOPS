import logging

import pandas as pd
from zenml import step


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass

    def get_data(self) -> pd.DataFrame:
        # Load the dataset
        df = pd.read_excel("./data/FYP.xlsx")
        
        # Clean column names by stripping whitespace (important for "district " column)
        df.columns = df.columns.str.strip()
        
        logging.info(f"Original dataset columns: {df.columns.tolist()}")
        logging.info(f"Dataset shape: {df.shape}")
        
        # Handle data type issues to ensure compatibility with parquet format
        df = self._fix_data_types(df)
        
        # Basic validation - ensure we have key columns for target derivation
        required_columns = ['tenure', 'actual_tenure']
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            raise ValueError(f"Missing required columns for target derivation: {missing_required}")
        
        return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix data type issues that cause problems with parquet serialization.
        """
        try:
            # Handle date columns that might have mixed types
            date_columns = ['kibor_date', 'disbursement_date', 'maturity_date', 
                           'invoice_due_date', 'cos_date', 'repayment_date', 'created_at']
            
            for col in date_columns:
                if col in df.columns:
                    # Convert to datetime, then to string to avoid parquet issues
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # Convert back to string to ensure consistent type for parquet
                    df[col] = df[col].astype(str)
            
            # Handle numeric columns that might have mixed types
            numeric_columns = ['s_no', 'quantity', 'musharaka', 'profit_rate', 'kibor_rate',
                              'financing_amount', 'tenure', 'actual_tenure', 'repayment_amount',
                              'remaining_principal_amount', 'charity_amount', 'cos_qty',
                              'business_years', 'collection_period_days']
            
            for col in numeric_columns:
                if col in df.columns:
                    # Convert to numeric, replacing invalid values with NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle string columns - ensure they are proper strings
            string_columns = ['cif_final', 'distributor_name_final', 'transaction_number',
                             'material_code_final', 'material_description', 'kibor_tenure_name',
                             'program', 'stock_type', 'repayment_status', 'district',
                             'entity_type']
            
            for col in string_columns:
                if col in df.columns:
                    # Convert to string, replacing NaN with empty string
                    df[col] = df[col].astype(str)
                    df[col] = df[col].replace('nan', '')
            
            # Handle the transaction_number column which might be too large for int
            if 'transaction_number' in df.columns:
                df['transaction_number'] = df['transaction_number'].astype(str)
            
            logging.info("Data type fixes applied successfully")
            return df
            
        except Exception as e:
            logging.error(f"Error fixing data types: {e}")
            # If fixing fails, convert all object columns to strings as fallback
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str)
            return df
        
        logging.info(f"Dataset shape after selecting available columns: {df.shape}")
        logging.info(f"Available columns: {existing_columns}")
        
        # Convert date columns to datetime
        date_columns = ['Kibor Date', 'Disbursement Date', 'Maturity Date']
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    logging.warning(f"Could not convert {col} to datetime: {e}")
        
        # Ensure numeric columns are numeric
        numeric_columns = ['Quantity', 'Musharaka', 'Profit Rate', 'Kibor Rate',
                          'Financing Amount', 'Tenure', 'Business years', 'Collection Period Days']
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logging.warning(f"Could not convert {col} to numeric: {e}")
        
        return df


@step
def ingest_data() -> pd.DataFrame:
    """
    Ingest data step for ZenML pipeline.
    
    Returns:
        df: pd.DataFrame - The ingested dataset
    """
    try:
        ingest_data_obj = IngestData()
        df = ingest_data_obj.get_data()
        logging.info(f"Data ingested successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error in ingest_data: {e}")
        raise e