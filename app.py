import numpy as np
import pandas as pd
import joblib
import streamlit as st 

model = joblib.load('models/random_forest_model.joblib')
preprocessor = joblib.load('models/preprocessor.joblib')
raw_columns = joblib.load("models/raw_columns.joblib")
defaults = joblib.load("models/default_values.joblib")


st.title("🏠 House Price Prediction")
overall_qual = st.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Living Area (sq ft)", 300, 6000, 1500)
garage_cars = st.slider("Garage Cars", 0, 5, 2)
total_bsmt_sf = st.number_input("Basement Area", 0, 6500, 800)
year_built = st.number_input("Year Built", 1870, 2010, 2000)

if st.button("Predict Price"):
    row = {col: defaults[col] for col in raw_columns}
    input_df = pd.DataFrame([row])
    input_df["OverallQual"] = overall_qual
    input_df["GrLivArea"] = gr_liv_area
    input_df["GarageCars"] = garage_cars
    input_df["TotalBsmtSF"] = total_bsmt_sf
    input_df["YearBuilt"] = year_built
    input_df = input_df[raw_columns]
    preprocessed_data = preprocessor.transform(input_df)
    prediction = model.predict(preprocessed_data)
    st.success(f"Predicted Price: ₹ {np.expm1(prediction[0]):,.0f}")