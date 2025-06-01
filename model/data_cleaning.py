import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
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
            # Handle missing values through orchestrator function
            data = self._impute_missing_values(data)
            
            # Apply feature derivation
            data = self._derive_features(data)
            
            # Drop Columns that are not needed
            cols_to_drop = [    'transaction_number','disbursement_date','maturity_date',
                            'actual_tenure','invoice_due_date','cos_date','cos_qty',
                            'repayment_status','repayment_amount',
                            'repayment_date','remaining_principal_amount',
                            'created_at','charity_amount']
            
            data = data.drop(columns=cols_to_drop, errors='ignore')
            
            return data
        except Exception as e:
            logging.error(f"Error in handle_data: {e}")
            raise e

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
                tenure_mode = data.groupby('distributor_name_final')['tenure'].agg(
                    lambda x: x.mode()[0] if not x.mode().empty else np.nan
                )
                data['tenure'] = data.apply(
                    lambda row: tenure_mode[row['distributor_name_final']] 
                    if pd.isnull(row['tenure']) else row['tenure'], 
                    axis=1
                )
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
            if 'repayment_status' in data.columns:
                # Identify distributors that have at least one "Pending Settlement" record
                distributors_with_pending = data[data['repayment_status'] == 'Pending Settlement']['distributor_name_final'].unique()
                
                # For distributors WITH at least 1 "Pending Settlement" record
                if 'repayment_amount' in data.columns and 'financing_amount' in data.columns:
                    mask_with_pending = (data['repayment_status'].isnull()) & (data['distributor_name_final'].isin(distributors_with_pending))
                    data.loc[mask_with_pending, 'repayment_amount'] = data.loc[mask_with_pending, 'financing_amount']
                    data.loc[mask_with_pending, 'repayment_status'] = 'Paid'
                    
                    # Fill actual_tenure with tenure where missing
                    if 'actual_tenure' in data.columns and 'tenure' in data.columns:
                        data.loc[data['actual_tenure'].isnull(), 'actual_tenure'] = data.loc[data['actual_tenure'].isnull(), 'tenure']
                
                # For distributors WITHOUT any "Pending Settlement" record
                mask_without_pending = (data['repayment_status'].isnull()) & (~data['distributor_name_final'].isin(distributors_with_pending))
                data.loc[mask_without_pending, 'repayment_status'] = 'Pending Settlement'
                if 'repayment_amount' in data.columns:
                    data.loc[mask_without_pending, 'repayment_amount'] = 0
            
            return data
        except Exception as e:
            logging.error(f"Error imputing repayment data: {e}")
            return data
    
    def _impute_kibor_tenure_name(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing kibor_tenure_name values using KNN imputation based on kibor_rate.
        """
        try:
            if 'kibor_tenure_name' in data.columns and 'kibor_rate' in data.columns:
                # Only apply imputation if there are missing values
                if data['kibor_tenure_name'].isnull().any():
                    imputer = KNNImputer(n_neighbors=5)
                    data['kibor_tenure_name'] = imputer.fit_transform(data[['kibor_tenure_name', 'kibor_rate']])[:, 0]
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
            
            return data
        except Exception as e:
            logging.error(f"Error deriving features: {e}")
            raise e
    
    def _derive_season_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Derive season feature from disbursement_date.
        """
        try:
            if 'disbursement_date' in data.columns:
                data['disbursement_date'] = pd.to_datetime(data['disbursement_date'])
                data['season'] = data['disbursement_date'].dt.month.apply(self._get_season)
            return data
        except Exception as e:
            logging.error(f"Error deriving season feature: {e}")
            return data
            
    def _derive_repayment_days_ratio_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate repayment_days_ratio for each distributor.
        """
        try:
            if 'actual_tenure' in data.columns and 'tenure' in data.columns and 'distributor_name_final' in data.columns:
                avg_actual_tenure = data.groupby('distributor_name_final')['actual_tenure'].mean()
                avg_tenure = data.groupby('distributor_name_final')['tenure'].mean()
                repayment_ratio = avg_actual_tenure / avg_tenure
                data['repayment_days_ratio'] = data['distributor_name_final'].map(repayment_ratio)
            return data
        except Exception as e:
            logging.error(f"Error deriving repayment_days_ratio feature: {e}")
            return data
    
    def _derive_previous_transaction_count_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate previous transaction count for each distributor.
        """
        try:
            if 'distributor_name_final' in data.columns and 's_no' in data.columns:
                data['previous_transaction_count'] = data.groupby('distributor_name_final')['s_no'].transform('count') - 1
            return data
        except Exception as e:
            logging.error(f"Error deriving previous_transaction_count feature: {e}")
            return data
    
    def _derive_repayment_ethics_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate repayment ethics for each distributor.
        """
        try:
            if 'distributor_name_final' in data.columns and 'actual_tenure' in data.columns and 'tenure' in data.columns:
                avg_actual_tenure = data.groupby('distributor_name_final')['actual_tenure'].mean()
                avg_tenure = data.groupby('distributor_name_final')['tenure'].mean()
                ethics_ratio = avg_actual_tenure / avg_tenure
                data['repayment_ethics'] = data['distributor_name_final'].map(ethics_ratio)
            return data
        except Exception as e:
            logging.error(f"Error deriving repayment_ethics feature: {e}")
            return data
    
    @staticmethod
    def _get_season(month):
        """
        Convert month number to season name.
        
        Args:
            month: Month as integer (1-12)
            
        Returns:
            Season name as string
        """
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # month in [9, 10, 11]
            return 'Fall'

class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)
