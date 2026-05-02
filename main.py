from src.data_preprocessing import preprocess_pipeline
from src.train_model import train_model 
from src.evaluate_model import evaluate_model
raw_data_path = 'data/raw/train.csv'
preprocced_data_path = 'data/processed/encoded_data.csv'
model_path = 'models/random_forest_model.joblib'



if __name__ == "__main__":
    # Preprocess the data
    encoded_data = preprocess_pipeline(raw_data_path, preprocced_data_path)

    # Train the model
    model = train_model(preprocced_data_path, model_path)

    # Evaluate the model
    rmse,r2 = evaluate_model(preprocced_data_path, model_path)
