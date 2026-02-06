# House Price Prediction using Linear Regression

The goal of this project is to predict house prices based on numerical features such as overall quality, living area, number of bathrooms, and year built. This project helps understand how different factors influence house prices.

## Dataset
The dataset was taken from Kaggle and contains numerical features related to house properties.
The target variable is SalePrice.
The dataset has no missing values and is suitable for learning core machine learning concepts.

## Approach
1. Loaded the dataset using Python's pandas library.
2. Separated features (X) and target variable (y).
3. Split the data into training and testing sets (80% training, 20% testing).
4. Trained a Linear Regression model.
5. Evaluated the model using R² score.

## Results
- Train R^2:  0.9790
- Test R^2:  0.9724

The close train and test scores indicate good generalization and no significant overfitting.

## Conclusion
This project helped me understand the complete machine learning workflow, including data preparation, model training, and evaluation. In the future, this project can be extended using feature scaling, categorical variables, or more advanced models.
