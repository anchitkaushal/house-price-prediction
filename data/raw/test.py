import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('house_prices_practice.csv')

# Create price categories (Low, Medium, High) for classification
df['PriceCategory'] = pd.cut(df['SalePrice'], 
                              bins=3, 
                              labels=['Low', 'Medium', 'High'])

# Select features for the model
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
            'YearBuilt', 'FullBath', 'BedroomAbvGr', 'LotArea']

X = df[features]
y = df['PriceCategory']

# Handle missing values
X = X.fillna(X.mean())

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for Naive Bayes)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = nb_model.predict(X_test_scaled)

# Evaluate the model
print("=== Naive Bayes Classifier Results ===\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nPredicted Classes:", np.unique(y_pred))
print("Actual Classes:", np.unique(y_test))