from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import joblib

def train_model(data_path, model_path): 
    data = pd.read_csv(data_path)