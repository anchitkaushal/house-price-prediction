import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error 
from sklearn.model_selection import train_test_split

def evaluate_model(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop('SalePrice', axis=1)
    y = np.log1p(data['SalePrice'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load(model_path)
    y_predict = model.predict(X_test)
    r2 = r2_score(y_test,y_predict)
    rmse = np.sqrt(mean_squared_error(y_test,y_predict))
    print("Model Evaluation results:")
    print("R2: ",r2)
    print("RMSE: ",rmse)
    return rmse , r2



