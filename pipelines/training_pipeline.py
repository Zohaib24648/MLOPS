from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from steps.config import ModelNameConfig

docker_settings = DockerSettings(required_integrations=[MLFLOW])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline():
    """
    Training pipeline for loan default prediction that connects all steps with proper data flow.
    
    Returns:
        None
    """
    # Create model configuration
    model_config = ModelNameConfig(
        model_name="xgboost",  # Options: xgboost, lightgbm, randomforest, catboost, logistic_regression
        fine_tuning=True
    )
    
    # Step 1: Ingest the data
    df = ingest_data()
    
    # Step 2: Clean the data and split into train/test
    x_train, x_test, y_train, y_test = clean_data(df)
    
    # Step 3: Train the classification model with configuration
    model = train_model(
        x_train=x_train, 
        x_test=x_test, 
        y_train=y_train, 
        y_test=y_test,
        config=model_config
    )
    
    # Step 4: Evaluate the classification model
    accuracy, f1_score = evaluation(model, x_test, y_test)