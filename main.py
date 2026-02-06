import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/raw/house_prices_practice.csv")
print("Rows,columns :",data.shape)
print(data.head())

x = data.drop("SalePrice",axis=1)
y= data["SalePrice"]

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

# Baseline Linear Regression model
# Used to understand feature impact and model behavior
model = LinearRegression()
model.fit(X_train,y_train)
Train_score = model.score(X_train,y_train)
Test_score  = model.score(X_test,y_test) 
print("Train R^2: ",Train_score)
print("Test R^2: ",Test_score)
