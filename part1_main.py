# part1_main.py
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# import pickle to save model


#os.path.join(os.path.dirname(__file__), '..')：返回 src 的上一级目录，sys.path.append(...)：这样就可以让 Python 在 2216 project-part1 目录下查找 src 目录里的模块。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#如果 sys.path.append(...) 写在 import 之后，Python 还是找不到 src，所以一定要在 import 之前添加路径。
from src.model_select import data_preparation, Linear_regression_prediction, decision_tree_prediction, Random_Forest_prediction, pickle_model

file_path = r"C:\algonquin\2025W\2216_ML\2216_project\2216_project-part1\data\cleaned_df.csv"

x_train, x_test, y_train, y_test = data_preparation(file_path)
#print(x_train.head())
#print(x_test.head())
#print(y_train.head())
#print(y_test.head())

'''
print('result using linear regression model is below:***')
Linear_regression_prediction(x_train, y_train, x_test, y_test)


print('result using decision tree model is below:&&&')
dtmodel=decision_tree_prediction(x_train, y_train, x_test, y_test)
print(pickle_model(dtmodel))
'''

print('result using random forest model is below:$$$')
#Random_Forest_prediction(x_train, y_train, x_test, y_test)
rfmodel=Random_Forest_prediction(x_train, y_train, x_test, y_test)
# 假设 x_train 是训练时使用的数据，确保预测数据有相同的列名
predict_data = [[2012, 216, 74, 1, 1, 618, 2000, 600, 1, 1, 0]]
# 转换为 DataFrame，并确保列名与训练数据一致
predict_data_df = pd.DataFrame(predict_data, columns=x_train.columns)

print(f'the price prediction using rfmodel is:', pickle_model(rfmodel, predict_data_df))

