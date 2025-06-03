import logging
import re
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        """
        # Actual columns in dataset:
        # s_no, cif_final, distributor_name_final, transaction_number, material_code_final, 
        # material_description, quantity, musharaka, profit_rate, kibor_rate, kibor_tenure_name, 
        # kibor_date, financing_amount, disbursement_date, maturity_date, tenure, actual_tenure, 
        # invoice_due_date, program, stock_type, cos_date, cos_qty, repayment_status, 
        # repayment_amount, repayment_date, remaining_principal_amount, created_at, charity_amount, 
        # district, business_years, entity_type, collection_period_days  
        # season, repayment_days_ratio,previous_transaction_count,repayment_ethics

        #derived_columns = [season, repayment_days_ratio,previous_transaction_count,repayment_ethics]
        
        #columns_to_use = [s_no,cif_final,distributor_name_final,material_code_final,
        # material_description,quantity,musharaka,profit_rate,kibor_rate,
        # kibor_tenure_name,kibor_date,financing_amount,tenure,
        # program,stock_type,district, business_years,entity_type, collection_period_days,
        # season, repayment_days_ratio,previous_transaction_count,repayment_ethics]

        try:
            logging.info(f"Starting data preprocessing. Initial shape: {data.shape}")
            
            # First, handle column name issues (e.g., trailing spaces)
            data = self._clean_column_names(data)
            
            # Handle missing values through orchestrator function
            data = self._impute_missing_values(data)
            
            # Apply feature derivation
            data = self._derive_features(data)
            
            # Encode categorical variables BEFORE dropping columns
            data = self._encode_categorical_variables(data)
            
            # Drop Columns that are not needed
            cols_to_drop = ['transaction_number', 'disbursement_date', 'maturity_date',
                           'actual_tenure', 'invoice_due_date', 'cos_date', 'cos_qty',
                           'repayment_status', 'repayment_amount',
                           'repayment_date', 'remaining_principal_amount',
                           'created_at', 'charity_amount','kibor_date']
            
            data = data.drop(columns=cols_to_drop, errors='ignore')
            
            logging.info(f"Data preprocessing completed. Final shape: {data.shape}")
            return data
        except Exception as e:
            logging.error(f"Error in handle_data: {e}")
            raise e

    def _clean_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names by removing trailing/leading spaces.
        """
        data.columns = data.columns.str.strip()
        return data

    def _impute_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrator function that applies all imputation methods sequentially.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with imputed missing values
        """
        try:
            # Apply each imputation function sequentially
            data = self._impute_charity_amount(data)
            data = self._impute_tenure(data)
            data = self._impute_repayment_data(data)
            data = self._impute_kibor_tenure_name(data)
            data = self._impute_cos_qty(data)
            
            return data
        except Exception as e:
            logging.error(f"Error in imputation orchestrator: {e}")
            raise e
    
    def _impute_charity_amount(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing charity_amount values with 0.
        """
        try:
            if 'charity_amount' in data.columns:
                data['charity_amount'] = data['charity_amount'].fillna(0)
            return data
        except Exception as e:
            logging.error(f"Error imputing charity_amount: {e}")
            return data
    
    def _impute_tenure(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing tenure values with mode of tenure for each distributor.
        """
        try:
            if 'tenure' in data.columns and 'distributor_name_final' in data.columns:
                # Calculate mode for each distributor
                def safe_mode(x):
                    mode_values = x.mode()
                    if len(mode_values) > 0:
                        return mode_values[0]
                    else:
                        # Return overall mode if distributor has no mode
                        overall_mode = data['tenure'].mode()
                        return overall_mode[0] if len(overall_mode) > 0 else data['tenure'].median()
                
                tenure_mode = data.groupby('distributor_name_final')['tenure'].agg(safe_mode)
                
                # Apply the imputation
                mask = data['tenure'].isnull()
                data.loc[mask, 'tenure'] = data.loc[mask, 'distributor_name_final'].map(tenure_mode)
                
                # Fill any remaining NaN values with overall median
                data['tenure'] = data['tenure'].fillna(data['tenure'].median())
            return data
        except Exception as e:
            logging.error(f"Error imputing tenure: {e}")
            return data
    
    def _impute_repayment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute repayment_status, repayment_amount and actual_tenure 
        based on distributor behavior.
        """
        try:
            if 'repayment_status' in data.columns and 'distributor_name_final' in data.columns:
                # Identify distributors that have at least one "Pending Settlement" record
                distributors_with_pending = data[
                    data['repayment_status'] == 'Pending Settlement'
                ]['distributor_name_final'].unique()
                
                # For distributors WITH at least 1 "Pending Settlement" record
                if 'repayment_amount' in data.columns and 'financing_amount' in data.columns:
                    mask_with_pending = (
                        data['repayment_status'].isnull() & 
                        data['distributor_name_final'].isin(distributors_with_pending)
                    )
                    data.loc[mask_with_pending, 'repayment_amount'] = data.loc[mask_with_pending, 'financing_amount']
                    data.loc[mask_with_pending, 'repayment_status'] = 'Paid'
                    
                    # Fill actual_tenure with tenure where missing
                    if 'actual_tenure' in data.columns and 'tenure' in data.columns:
                        mask_actual_tenure = data['actual_tenure'].isnull()
                        data.loc[mask_actual_tenure, 'actual_tenure'] = data.loc[mask_actual_tenure, 'tenure']
                
                # For distributors WITHOUT any "Pending Settlement" record
                mask_without_pending = (
                    data['repayment_status'].isnull() & 
                    ~data['distributor_name_final'].isin(distributors_with_pending)
                )
                data.loc[mask_without_pending, 'repayment_status'] = 'Pending Settlement'
                if 'repayment_amount' in data.columns:
                    data.loc[mask_without_pending, 'repayment_amount'] = 0
            
            return data
        except Exception as e:
            logging.error(f"Error imputing repayment data: {e}")
            return data
    
    def _impute_kibor_tenure_name(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing kibor_tenure_name values using KNN imputation based on kibor_rate and profit_rate.
        Creates the column if it doesn't exist.
        """
        try:
            # Create kibor_tenure_name column if it doesn't exist
            if 'kibor_tenure_name' not in data.columns:
                data['kibor_tenure_name'] = np.nan
                logging.info("kibor_tenure_name column created as it was missing")
            
            # Check if we have the required columns and missing values to impute
            if 'kibor_rate' in data.columns and 'profit_rate' in data.columns and data['kibor_tenure_name'].isnull().any():
                before_nulls = data['kibor_tenure_name'].isnull().sum()
                
                # Function to extract numeric values from tenure strings
                def extract_numeric_tenure(tenure_str):
                    if pd.isna(tenure_str):
                        return np.nan
                    try:
                        # Use regex to find the first number in the string
                        match = re.search(r'\d+', str(tenure_str))
                        if match:
                            return int(match.group())
                        else:
                            return np.nan
                    except:
                        return np.nan
                
                # Create numeric tenure column
                data['numeric_tenure'] = data['kibor_tenure_name'].apply(extract_numeric_tenure)
                
                # Check if we have any non-null values to work with
                non_null_mask = data['numeric_tenure'].notna()
                
                if non_null_mask.sum() > 0:
                    # Use KNN imputation with kibor_rate and profit_rate as features
                    imputer = KNNImputer(n_neighbors=5)
                    
                    # Create feature matrix for imputation
                    features_for_imputation = data[['numeric_tenure', 'kibor_rate', 'profit_rate']].copy()
                    
                    # Perform KNN imputation
                    imputed_result = imputer.fit_transform(features_for_imputation)
                    
                    # Get the imputed numeric tenure values and round to integers
                    imputed_numeric_tenure = np.round(imputed_result[:, 0]).astype(int)
                    
                    # Convert back to original format "X - Month"
                    data['kibor_tenure_name'] = [f"{tenure} - Month" for tenure in imputed_numeric_tenure]
                    
                    logging.info(f"kibor_tenure_name: filled {before_nulls} missing values using KNN based on kibor_rate and profit_rate")
                    
                else:
                    # If no existing values, create default mapping based on rates
                    def assign_default_tenure(kibor_rate, profit_rate):
                        if pd.isna(kibor_rate) and pd.isna(profit_rate):
                            return "6 - Month"  # default
                        
                        # Use average of available rates
                        avg_rate = np.nanmean([kibor_rate, profit_rate])
                        
                        if avg_rate <= 10:
                            return "12 - Month"  # Lower rates = longer tenure
                        elif avg_rate <= 15:
                            return "6 - Month"   # Medium rates = medium tenure
                        elif avg_rate <= 20:
                            return "3 - Month"   # Higher rates = shorter tenure
                        else:
                            return "1 - Month"   # Very high rates = very short tenure
                    
                    data['kibor_tenure_name'] = data.apply(
                        lambda row: assign_default_tenure(row['kibor_rate'], row['profit_rate']), 
                        axis=1
                    )
                    logging.info(f"kibor_tenure_name: created default values based on kibor_rate and profit_rate")
                
                # Clean up temporary column
                data.drop('numeric_tenure', axis=1, inplace=True)
            
            elif 'kibor_rate' not in data.columns or 'profit_rate' not in data.columns:
                logging.warning("kibor_rate or profit_rate columns not found for imputation")
            
            return data
            
        except Exception as e:
            logging.error(f"Error imputing kibor_tenure_name: {e}")
            return data
        
    def _impute_cos_qty(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing cos_qty values with quantity.
        """
        try:
            if 'cos_qty' in data.columns and 'quantity' in data.columns:
                data['cos_qty'] = data['cos_qty'].fillna(data['quantity'])
            return data
        except Exception as e:
            logging.error(f"Error imputing cos_qty: {e}")
            return data

    def _encode_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables to make them compatible with XGBoost.
        Uses Label Encoding for all categorical variables.
        """
        try:
            from sklearn.preprocessing import LabelEncoder
            
            # Get all object columns (categorical)
            categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
            
            # Exclude the target variable if it exists
            if 'defaulted' in categorical_columns:
                categorical_columns.remove('defaulted')
            
            logging.info(f"Encoding categorical columns: {categorical_columns}")
            
            for col in categorical_columns:
                if col in data.columns:
                    # Create label encoder
                    le = LabelEncoder()
                    
                    # Handle missing values by filling with 'Unknown'
                    data[col] = data[col].fillna('Unknown')
                    
                    # Convert to string to ensure consistency
                    data[col] = data[col].astype(str)
                    
                    # Apply label encoding
                    data[col] = le.fit_transform(data[col])
                    
                    logging.info(f"Encoded column '{col}' - unique values: {data[col].nunique()}")
            
            # Verify no object columns remain (except target)
            remaining_objects = data.select_dtypes(include=['object']).columns.tolist()
            if 'defaulted' in remaining_objects:
                remaining_objects.remove('defaulted')
            
            if remaining_objects:
                logging.warning(f"Warning: Object columns still remain: {remaining_objects}")
            else:
                logging.info("All categorical variables successfully encoded")
            
            return data
            
        except Exception as e:
            logging.error(f"Error encoding categorical variables: {e}")
            # Fallback: convert all object columns to numeric if possible
            object_cols = data.select_dtypes(include=['object']).columns
            for col in object_cols:
                if col != 'defaulted':
                    try:
                        data[col] = pd.factorize(data[col])[0]
                    except:
                        data[col] = 0
            return data

    def _derive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Derive new features from existing columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with derived features
        """
        try:
            # Apply each feature generation function
            data = self._derive_season_feature(data)
            data = self._derive_repayment_days_ratio_feature(data)
            data = self._derive_previous_transaction_count_feature(data)
            data = self._derive_repayment_ethics_feature(data)
            data = self._derive_defaulted_feature(data)  # This is our target variable
            
            return data
        except Exception as e:
            logging.error(f"Error deriving features: {e}")
            raise e
    
    def _derive_defaulted_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Derive defaulted feature based on the ratio of actual_tenure to tenure.
        If actual_tenure > 1.25 * tenure, then defaulted = True, else False.
        This is our TARGET VARIABLE for classification.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with defaulted feature added
        """
        try:
            if 'actual_tenure' in data.columns and 'tenure' in data.columns:
                # Calculate the defaulted status based on the 1.25x threshold
                data['defaulted'] = (data['actual_tenure'] > (1.25 * data['tenure']))
                
                # Fill NaN values in defaulted column with False
                data['defaulted'] = data['defaulted'].fillna(False)
                
                # Convert boolean to integer (0/1) for ML models
                data['defaulted'] = data['defaulted'].astype(int)
                
                # Log target distribution
                target_counts = data['defaulted'].value_counts()
                logging.info(f"Target variable (defaulted) distribution: {target_counts.to_dict()}")
                
            else:
                logging.warning("Cannot derive 'defaulted' feature: required columns missing")
                # Create a default column with zeros if columns are missing
                data['defaulted'] = 0
            return data
        except Exception as e:
            logging.error(f"Error deriving defaulted feature: {e}")
            return data
    
    def _derive_season_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Derive season feature from disbursement_date.
        """
        try:
            if 'disbursement_date' in data.columns:
                # Convert to datetime, handling errors
                data['disbursement_date'] = pd.to_datetime(data['disbursement_date'], errors='coerce')
                # Extract month and map to season
                data['season'] = data['disbursement_date'].dt.month.map(self._get_season)
                # Fill NaN values with a default season
                data['season'] = data['season'].fillna('Unknown')
            else:
                logging.warning("disbursement_date column not found. Cannot derive season feature.")
                data['season'] = 'Unknown'
            return data
        except Exception as e:
            logging.error(f"Error deriving season feature: {e}")
            return data
            
    def _derive_repayment_days_ratio_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate repayment_days_ratio for each distributor.
        """
        try:
            if all(col in data.columns for col in ['actual_tenure', 'tenure', 'distributor_name_final']):
                # Calculate average actual_tenure and tenure per distributor
                distributor_stats = data.groupby('distributor_name_final').agg({
                    'actual_tenure': 'mean',
                    'tenure': 'mean'
                })
                
                # Calculate ratio, handling division by zero
                distributor_stats['repayment_ratio'] = np.where(
                    distributor_stats['tenure'] > 0,
                    distributor_stats['actual_tenure'] / distributor_stats['tenure'],
                    1.0  # Default ratio if tenure is 0
                )
                
                # Map back to original dataframe
                data['repayment_days_ratio'] = data['distributor_name_final'].map(
                    distributor_stats['repayment_ratio']
                )
                
                # Fill any NaN values with 1.0
                data['repayment_days_ratio'] = data['repayment_days_ratio'].fillna(1.0)
            else:
                logging.warning("Required columns for repayment_days_ratio not found")
                data['repayment_days_ratio'] = 1.0
            return data
        except Exception as e:
            logging.error(f"Error deriving repayment_days_ratio feature: {e}")
            return data
    
    def _derive_previous_transaction_count_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate previous transaction count for each distributor.
        """
        try:
            if 'distributor_name_final' in data.columns:
                # Use cumcount to get the count of previous transactions
                data = data.sort_values('s_no') if 's_no' in data.columns else data
                data['previous_transaction_count'] = data.groupby('distributor_name_final').cumcount()
            else:
                logging.warning("distributor_name_final column not found")
                data['previous_transaction_count'] = 0
            return data
        except Exception as e:
            logging.error(f"Error deriving previous_transaction_count feature: {e}")
            return data
    
    def _derive_repayment_ethics_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate repayment ethics for each distributor.
        Note: This is the same as repayment_days_ratio but kept for compatibility.
        """
        try:
            if 'repayment_days_ratio' in data.columns:
                # Use the same value as repayment_days_ratio
                data['repayment_ethics'] = data['repayment_days_ratio']
            elif all(col in data.columns for col in ['actual_tenure', 'tenure', 'distributor_name_final']):
                # Calculate if repayment_days_ratio wasn't calculated
                distributor_stats = data.groupby('distributor_name_final').agg({
                    'actual_tenure': 'mean',
                    'tenure': 'mean'
                })
                
                distributor_stats['ethics_ratio'] = np.where(
                    distributor_stats['tenure'] > 0,
                    distributor_stats['actual_tenure'] / distributor_stats['tenure'],
                    1.0
                )
                
                data['repayment_ethics'] = data['distributor_name_final'].map(
                    distributor_stats['ethics_ratio']
                )
                data['repayment_ethics'] = data['repayment_ethics'].fillna(1.0)
            else:
                logging.warning("Required columns for repayment_ethics not found")
                data['repayment_ethics'] = 1.0
            return data
        except Exception as e:
            logging.error(f"Error deriving repayment_ethics feature: {e}")
            return data
    
    @staticmethod
    def _get_season(month: Optional[float]) -> str:
        """
        Convert month number to season name.
        
        Args:
            month: Month as integer (1-12) or float
            
        Returns:
            Season name as string
        """
        if pd.isna(month):
            return 'Unknown'
            
        month = int(month)
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Unknown'


class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            if "defaulted" not in data.columns:
                raise ValueError("Target column 'defaulted' not found in the data. "
                               "Make sure DataPreprocessStrategy was applied first.")
            
            X = data.drop("defaulted", axis=1)
            y = data["defaulted"]
            
            # Ensure we have data to split
            if len(X) == 0:
                raise ValueError("No data available for splitting")
            
            # Check class distribution
            class_distribution = y.value_counts(normalize=True)
            logging.info(f"Class distribution before split: {class_distribution.to_dict()}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logging.info(f"Data split completed. Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in data splitting: {e}")
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)