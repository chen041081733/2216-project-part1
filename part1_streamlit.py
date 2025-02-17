
import os
import sys

#os.path.join(os.path.dirname(__file__), '..')：返回 src 的上一级目录，sys.path.append(...)：这样就可以让 Python 在 2216 project-part1 目录下查找 src 目录里的模块。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging


import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from src.model_select import data_preparation, Linear_regression_prediction, decision_tree_prediction, Random_Forest_prediction, pickle_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:

    #file_path = r"C:\algonquin\2025W\2216_ML\2216_project\2216_project-part1\data\cleaned_df.csv"
    file_path = "https://raw.githubusercontent.com/chen041081733/2216-project-part1/main/data/cleaned_df.csv"
    x_train, x_test, y_train, y_test = data_preparation(file_path)
    logging.info("data load successfully")
except Exception as e:
    logging.error(f"data load fail: {e}")
    st.error("data load fail, please check again。")
    st.stop()




# input form
st.subheader("please input info of house")

# create input 
year_sold = st.number_input("Year_Sold", min_value=1900, max_value=2025, step=1)
property_tax = st.number_input("Property Tax", min_value=0)
insurance = st.number_input("Insurance", min_value=0)
beds = st.number_input("beds", min_value=1, step=1)
baths = st.number_input("baths", min_value=1, step=1)
sqft = st.number_input("sqft)", min_value=1)
year_built = st.number_input("year_built", min_value=1900, max_value=2025, step=1)
lot_size = st.number_input("lot_size", min_value=1)
basement = st.selectbox("basement", ["Y", "N"])

# property type selection
property_type = st.selectbox("property_type", ["Bunglow", "Condo"])

# transfer 'basement' to binary
basement_binary = 1 if basement == "Y" else 0

# input for price prediction 
if st.button("prediction of house price"):
    input_data = pd.DataFrame({
        'year_sold': [year_sold],
        'property_tax': [property_tax],
        'insurance': [insurance],
        'beds': [beds],
        'baths': [baths],
        'sqft': [sqft],
        'year_built': [year_built],
        'lot_size': [lot_size],
        'basement': [basement_binary],
        'property_type_Bunglow': [1 if property_type == "Bunglow" else 0],
        'property_type_Condo': [1 if property_type == "Condo" else 0],
    })
    
    # use Random Forest model 
    rfmodel=Random_Forest_prediction(x_train, y_train, x_test, y_test)
    prediction = pickle_model(rfmodel, input_data)
    
    st.write(f"predict the housr price is：${prediction[0]:,.2f}")