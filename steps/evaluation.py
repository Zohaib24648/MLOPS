import logging
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

try:
    experiment_tracker = Client().active_stack.experiment_tracker
    experiment_tracker_name = experiment_tracker.name if experiment_tracker else None
except:
    experiment_tracker_name = None


@step(experiment_tracker=experiment_tracker_name)
def evaluation(
    model: ClassifierMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "accuracy"], Annotated[float, "f1_score"]]:
    """
    Evaluate a classification model for loan default prediction.
    
    Args:
        model: ClassifierMixin - The trained classification model
        x_test: pd.DataFrame - Test features
        y_test: pd.Series - Test labels (0=Non_Defaulter, 1=Defaulter)
        
    Returns:
        accuracy: float - Model accuracy
        f1_score: float - Model F1 score
    """
    try:
        # Ensure target is integer type
        y_test = y_test.astype(int)
        
        # Make predictions
        prediction = model.predict(x_test)
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_test, prediction)
        mlflow.log_metric("accuracy", float(accuracy))
        
        precision = precision_score(y_test, prediction, average='binary')
        mlflow.log_metric("precision", float(precision))
        
        recall = recall_score(y_test, prediction, average='binary')
        mlflow.log_metric("recall", float(recall))
        
        f1 = f1_score(y_test, prediction, average='binary')
        mlflow.log_metric("f1_score", float(f1))
        
        # Try to get probability predictions for ROC AUC
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(x_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
                mlflow.log_metric("roc_auc", float(roc_auc))
                logging.info(f"ROC AUC Score: {roc_auc:.4f}")
        except Exception as e:
            logging.warning(f"Could not calculate ROC AUC: {e}")
        
        # Confusion Matrix metrics
        try:
            conf_matrix = confusion_matrix(y_test, prediction)
            tn, fp, fn, tp = conf_matrix.ravel()
            
            # Log confusion matrix components
            mlflow.log_metric("true_positives", int(tp))
            mlflow.log_metric("true_negatives", int(tn))
            mlflow.log_metric("false_positives", int(fp))
            mlflow.log_metric("false_negatives", int(fn))
            
            # Calculate additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            mlflow.log_metric("specificity", float(specificity))
            mlflow.log_metric("false_positive_rate", float(fpr))
            mlflow.log_metric("false_negative_rate", float(fnr))
            mlflow.log_metric("negative_predictive_value", float(npv))
            
            logging.info("Confusion Matrix:")
            logging.info(f"True Positives (TP): {tp}")
            logging.info(f"True Negatives (TN): {tn}")
            logging.info(f"False Positives (FP): {fp}")
            logging.info(f"False Negatives (FN): {fn}")
            
        except Exception as e:
            logging.warning(f"Could not calculate confusion matrix metrics: {e}")
        
        # Log classification report
        try:
            class_report = classification_report(y_test, prediction, output_dict=True)
            # Log metrics for each class
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict):  # Skip string summaries
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):  # Only log numeric values
                            mlflow.log_metric(f"{class_name}_{metric_name}", float(value))
        except Exception as e:
            logging.warning(f"Could not log classification report: {e}")
        
        logging.info(f"Model evaluation completed - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        return float(accuracy), float(f1)
        
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise e