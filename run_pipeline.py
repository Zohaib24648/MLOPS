from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    # Run the loan default prediction training pipeline
    print("=== STARTING LOAN DEFAULT PREDICTION PIPELINE ===")
    train_pipeline()
    
    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment.\n"
        "Here you'll also be able to compare different model runs."
    )