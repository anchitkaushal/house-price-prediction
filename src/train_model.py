from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import joblib

def train_model(data_path, model_path): 
    data = pd.read_csv(data_path)
    X = data.drop('SalePrice', axis=1)
    y = np.log1p(data['SalePrice'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(random_state=42)
    param_dic ={
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split':[2, 5, 10],
        'min_samples_leaf':[1, 2, 4]
    }
    random_search = RandomizedSearchCV(
        estimator = rf,
        param_distributions = param_dic,
        n_iter = 20,
        cv = 5,
        scoring = 'r2',
        n_jobs = -1,
        random_state = 42
          )
    random_search.fit(X_train, y_train)
    best_rf = random_search.best_estimator_
    joblib.dump(best_rf, model_path)
    return best_rf

