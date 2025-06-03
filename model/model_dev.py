import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, cast

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Any:
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
            **kwargs: Additional parameters for the model

        Returns:
            Trained model
        """
        pass

    @abstractmethod
    def optimize(self, trial: optuna.Trial, x_train: pd.DataFrame, y_train: pd.Series, 
                x_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            x_train: Training data
            y_train: Target data
            x_test: Testing data
            y_test: Testing target

        Returns:
            Model score on test data
        """
        pass


class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """

    def train(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> RandomForestRegressor:
        """
        Train Random Forest model with given parameters.
        
        Args:
            x_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for the model
            
        Returns:
            Trained Random Forest model
        """
        try:
            # Set default parameters if not provided
            default_params: Dict[str, Any] = {
                'random_state': 42
            }
            
            # Only add valid RandomForest parameters from kwargs
            valid_params = [
                'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                'max_features', 'max_leaf_nodes', 'min_impurity_decrease',
                'bootstrap', 'oob_score', 'n_jobs', 'verbose', 'warm_start'
            ]
            
            for param in valid_params:
                if param in kwargs:
                    default_params[param] = kwargs[param]
            
            reg = RandomForestRegressor(**default_params)
            reg.fit(x_train, y_train)
            logging.info("Random Forest model trained successfully")
            return reg
        except Exception as e:
            logging.error(f"Error training Random Forest model: {e}")
            raise e

    def optimize(self, trial: optuna.Trial, x_train: pd.DataFrame, y_train: pd.Series,
                x_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Optimize hyperparameters for Random Forest using Optuna.
        
        Args:
            trial: Optuna trial object
            x_train: Training features
            y_train: Training target
            x_test: Testing features
            y_test: Testing target
            
        Returns:
            Model score on test data
        """
        try:
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            
            reg = self.train(
                x_train, y_train,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )
            score = reg.score(x_test, y_test)
            return float(score)
        except Exception as e:
            logging.error(f"Error optimizing Random Forest model: {e}")
            raise e


class LightGBMModel(Model):
    """
    LightGBMModel that implements the Model interface.
    """

    def train(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> LGBMRegressor:
        """
        Train LightGBM model with given parameters.
        
        Args:
            x_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for the model
            
        Returns:
            Trained LightGBM model
        """
        try:
            # Set default parameters if not provided
            default_params: Dict[str, Any] = {
                'random_state': 42,
                'verbose': -1  # Suppress LightGBM warnings
            }
            
            # Only add valid LightGBM parameters from kwargs
            valid_params = [
                'n_estimators', 'max_depth', 'learning_rate', 'num_leaves', 
                'min_child_samples', 'subsample', 'colsample_bytree', 
                'reg_alpha', 'reg_lambda', 'min_split_gain', 'min_child_weight'
            ]
            
            for param in valid_params:
                if param in kwargs:
                    default_params[param] = kwargs[param]
            
            reg = LGBMRegressor(**default_params)
            reg.fit(x_train, y_train)
            logging.info("LightGBM model trained successfully")
            return reg
        except Exception as e:
            logging.error(f"Error training LightGBM model: {e}")
            raise e

    def optimize(self, trial: optuna.Trial, x_train: pd.DataFrame, y_train: pd.Series,
                x_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Optimize hyperparameters for LightGBM using Optuna.
        
        Args:
            trial: Optuna trial object
            x_train: Training features
            y_train: Training target
            x_test: Testing features
            y_test: Testing target
            
        Returns:
            Model score on test data
        """
        try:
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
            num_leaves = trial.suggest_int("num_leaves", 10, 100)
            min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
            
            reg = self.train(
                x_train, y_train,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples
            )
            # Use R2 score instead of .score() method for compatibility
            y_pred = reg.predict(x_test)
            # Ensure y_pred is a proper numpy array for r2_score
            y_pred = np.asarray(y_pred).ravel()
            y_test_array = np.asarray(y_test).ravel()
            score = r2_score(y_test_array, y_pred)
            return float(score)
        except Exception as e:
            logging.error(f"Error optimizing LightGBM model: {e}")
            raise e


class XGBoostModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """

    def train(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> xgb.XGBRegressor:
        """
        Train XGBoost model with given parameters.
        
        Args:
            x_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for the model
            
        Returns:
            Trained XGBoost model
        """
        try:
            # Set default parameters if not provided
            default_params: Dict[str, Any] = {
                'random_state': 42,
                'eval_metric': 'rmse'
            }
            
            # Only add valid XGBoost parameters from kwargs
            valid_params = [
                'n_estimators', 'max_depth', 'learning_rate', 'subsample', 
                'colsample_bytree', 'colsample_bylevel', 'colsample_bynode',
                'reg_alpha', 'reg_lambda', 'gamma', 'min_child_weight'
            ]
            
            for param in valid_params:
                if param in kwargs:
                    default_params[param] = kwargs[param]
            
            reg = xgb.XGBRegressor(**default_params)
            reg.fit(x_train, y_train)
            logging.info("XGBoost model trained successfully")
            return reg
        except Exception as e:
            logging.error(f"Error training XGBoost model: {e}")
            raise e

    def optimize(self, trial: optuna.Trial, x_train: pd.DataFrame, y_train: pd.Series,
                x_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Optimize hyperparameters for XGBoost using Optuna.
        
        Args:
            trial: Optuna trial object
            x_train: Training features
            y_train: Training target
            x_test: Testing features
            y_test: Testing target
            
        Returns:
            Model score on test data
        """
        try:
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            
            reg = self.train(
                x_train, y_train,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree
            )
            score = reg.score(x_test, y_test)
            return float(score)
        except Exception as e:
            logging.error(f"Error optimizing XGBoost model: {e}")
            raise e


class CatBoostModel(Model):
    """
    CatBoostModel that implements the Model interface.
    """

    def train(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> CatBoostRegressor:
        """
        Train CatBoost model with given parameters.
        
        Args:
            x_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for the model
            
        Returns:
            Trained CatBoost model
        """
        try:
            # Set default parameters if not provided
            default_params: Dict[str, Any] = {
                'random_state': 42,
                'verbose': False  # Suppress CatBoost output
            }
            
            # Only add valid CatBoost parameters from kwargs
            valid_params = [
                'iterations', 'depth', 'learning_rate', 'l2_leaf_reg', 
                'model_size_reg', 'rsm', 'loss_function', 'border_count',
                'feature_border_type', 'per_float_feature_quantization'
            ]
            
            for param in valid_params:
                if param in kwargs:
                    default_params[param] = kwargs[param]
            
            reg = CatBoostRegressor(**default_params)
            reg.fit(x_train, y_train)
            logging.info("CatBoost model trained successfully")
            return reg
        except Exception as e:
            logging.error(f"Error training CatBoost model: {e}")
            raise e

    def optimize(self, trial: optuna.Trial, x_train: pd.DataFrame, y_train: pd.Series,
                x_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Optimize hyperparameters for CatBoost using Optuna.
        
        Args:
            trial: Optuna trial object
            x_train: Training features
            y_train: Training target
            x_test: Testing features
            y_test: Testing target
            
        Returns:
            Model score on test data
        """
        try:
            iterations = trial.suggest_int("iterations", 50, 200)
            depth = trial.suggest_int("depth", 3, 10)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10)
            
            reg = self.train(
                x_train, y_train,
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg
            )
            # Use R2 score for consistency
            y_pred = reg.predict(x_test)
            # Ensure y_pred is a proper numpy array for r2_score
            y_pred = np.asarray(y_pred).ravel()
            y_test_array = np.asarray(y_test).ravel()
            score = r2_score(y_test_array, y_pred)
            return float(score)
        except Exception as e:
            logging.error(f"Error optimizing CatBoost model: {e}")
            raise e


class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> LinearRegression:
        """
        Train Linear Regression model with given parameters.
        
        Args:
            x_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for the model
            
        Returns:
            Trained Linear Regression model
        """
        try:
            # Set default parameters if not provided
            default_params: Dict[str, Any] = {}
            
            # Only add valid LinearRegression parameters from kwargs
            valid_params = [
                'fit_intercept', 'copy_X', 'n_jobs', 'positive'
            ]
            
            for param in valid_params:
                if param in kwargs:
                    default_params[param] = kwargs[param]
            
            reg = LinearRegression(**default_params)
            reg.fit(x_train, y_train)
            logging.info("Linear Regression model trained successfully")
            return reg
        except Exception as e:
            logging.error(f"Error training Linear Regression model: {e}")
            raise e

    def optimize(self, trial: optuna.Trial, x_train: pd.DataFrame, y_train: pd.Series,
                x_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        For linear regression, there are no hyperparameters to tune in the basic version,
        so we simply return the score of the trained model.
        
        Args:
            trial: Optuna trial object (unused for linear regression)
            x_train: Training features
            y_train: Training target
            x_test: Testing features
            y_test: Testing target
            
        Returns:
            Model score on test data
        """
        try:
            reg = self.train(x_train, y_train)
            score = reg.score(x_test, y_test)
            return float(score)
        except Exception as e:
            logging.error(f"Error optimizing Linear Regression model: {e}")
            raise e


class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__(self, model: Model, x_train: pd.DataFrame, y_train: pd.Series, 
                 x_test: pd.DataFrame, y_test: pd.Series):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            model: Model instance that implements the Model interface
            x_train: Training features
            y_train: Training target
            x_test: Testing features
            y_test: Testing target
        """
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            n_trials: Number of optimization trials to run
            
        Returns:
            Dictionary of best hyperparameters found
        """
        try:
            logging.info(f"Starting hyperparameter optimization with {n_trials} trials")
            
            def objective(trial: optuna.Trial) -> float:
                return self.model.optimize(
                    trial, self.x_train, self.y_train, self.x_test, self.y_test
                )
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            
            logging.info(f"Best trial score: {study.best_trial.value}")
            logging.info(f"Best parameters: {study.best_trial.params}")
            
            return study.best_trial.params
        except Exception as e:
            logging.error(f"Error during hyperparameter optimization: {e}")
            raise e