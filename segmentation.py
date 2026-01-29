import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans=joblib.load("kmeans_model.pkl")
scaler=joblib.load("scaler.pkl")


st.title("Customer Segmentation Prediction")
st.write("Enter customer details to predict their segment.")
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=2000000, value=50000)
total_spending = st.number_input("Total_Spending (sum of purchases)", min_value=0, max_value=50000, value=2000)
num_web_purchases= st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases= st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)


input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spend": [total_spending],     
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    