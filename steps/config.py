from zenml.config.base_settings import BaseSettings as BaseParameters


class ModelNameConfig(BaseParameters):
    """Model Configurations for Loan Default Prediction"""
    model_name: str = "xgboost"  # Options: xgboost, lightgbm, randomforest, catboost, logistic_regression
    fine_tuning: bool = True  # Enable hyperparameter tuning