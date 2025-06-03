import logging

import mlflow
import pandas as pd
from typing import Union
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from zenml import step
from zenml.client import Client

from steps.config import ModelNameConfig

try:
    experiment_tracker = Client().active_stack.experiment_tracker
    experiment_tracker_name = experiment_tracker.name if experiment_tracker else None
except:
    experiment_tracker_name = None


@step(experiment_tracker=experiment_tracker_name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> Union[LGBMClassifier, RandomForestClassifier, XGBClassifier, CatBoostClassifier, LogisticRegression]:
    """
    Train a classification model for loan default prediction.
    
    Args:
        x_train: pd.DataFrame - Training features
        x_test: pd.DataFrame - Test features
        y_train: pd.Series - Training labels (0=Non_Defaulter, 1=Defaulter)
        y_test: pd.Series - Test labels
        config: ModelNameConfig - Model configuration
        
    Returns:
        model: ClassifierMixin - Trained classification model
    """
    try:
        # Ensure target is integer type for classification
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        
        # Calculate class weight for imbalanced data
        scale_pos_weight_value = len(y_train[y_train==0]) / len(y_train[y_train==1])
        logging.info(f"scale_pos_weight: {scale_pos_weight_value:.4f} (to handle class imbalance)")
        
        # Select and train model based on configuration
        if config.model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=10,
                random_state=42,
                verbose=-1,
                class_weight='balanced'
            )
            
        elif config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight_value
            )
            
        elif config.model_name == "catboost":
            mlflow.catboost.autolog()
            model = CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=10,
                random_state=42,
                verbose=False,
                class_weights=[1, scale_pos_weight_value]
            )
            
        elif config.model_name == "logistic_regression":
            mlflow.sklearn.autolog()
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
            
        else:
            raise ValueError(f"Model '{config.model_name}' not supported. "
                           f"Choose from: lightgbm, randomforest, xgboost, catboost, logistic_regression")
        
        # Train the model
        logging.info(f"Training {config.model_name} model...")
        model.fit(x_train, y_train)
        
        # Log model performance on training set
        train_score = model.score(x_train, y_train)
        mlflow.log_metric("train_accuracy", float(train_score))
        logging.info(f"Training accuracy: {train_score:.4f}")
        
        # Log model performance on test set
        test_score = model.score(x_test, y_test)
        mlflow.log_metric("test_accuracy", float(test_score))
        logging.info(f"Test accuracy: {test_score:.4f}")
        
        # Log additional metrics
        mlflow.log_metric("scale_pos_weight", float(scale_pos_weight_value))
        mlflow.log_param("model_type", config.model_name)
        mlflow.log_param("fine_tuning", config.fine_tuning)
        
        return model
        
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise e