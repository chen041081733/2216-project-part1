
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

# Prepare data for model training
def data_preparation(filepath):
    df=pd.read_csv(filepath)
    
# seperate input features in x
    x = df.drop('price', axis=1)  #keep all column except price as input for training model
    x = pd.get_dummies(x, columns=['property_type'], prefix='property_type')
# store the target variable in y
    y = df['price']  #select price as target variable
   
# change type from bull to int
    x['property_type_Bunglow'] = x['property_type_Bunglow'].astype(int)  
    x['property_type_Condo'] = x['property_type_Condo'].astype(int)
# Split the dataset
    # x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=x.property_type_Bunglow)
    return(train_test_split(x,y, test_size=0.2, stratify=x.property_type_Bunglow))


# train model

def Linear_regression_prediction(x_train, y_train, x_test, y_test):
    model = LinearRegression()
    lrmodel = model.fit(x_train, y_train)
    print(lrmodel.coef_)
    print(lrmodel.intercept_)

    # make preditions on train set
    train_pred = lrmodel.predict(x_train)
    print(train_pred)

# evaluate your model
# we need mean absolute error

    train_mae = mean_absolute_error(train_pred, y_train)
    print('Train error is', train_mae)

# make predictions om test set
    ypred = lrmodel.predict(x_test)

#evaluate the model
    test_mae = mean_absolute_error(ypred, y_test)
    print('Test error is', test_mae)
#Our model is still not good beacuse we need a model with Mean Absolute Error < $70,000

def decision_tree_prediction(x_train, y_train, x_test, y_test):
    dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567) # create an instance of the class

    # train the model
    dtmodel = dt.fit(x_train,y_train)

    # make predictions using the test set
    ytrain_pred = dtmodel.predict(x_train)

    # evaluate the model
    train_mae = mean_absolute_error(ytrain_pred, y_train)
    
    # make predictions using the test set
    ytest_pred = dtmodel.predict(x_test)

    test_mae = mean_absolute_error(ytest_pred, y_test)
    print(test_mae)

    return dtmodel


    

def Random_Forest_prediction(x_train, y_train, x_test, y_test):
    # create an instance of the model
    rf = RandomForestRegressor(n_estimators=200, criterion='absolute_error')

    # train the model
    rfmodel = rf.fit(x_train,y_train)

    # make prediction on train set
    ytrain_pred = rfmodel.predict(x_train)

    # make predictions on the x_test values
    ytest_pred = rfmodel.predict(x_test)

    # evaluate the model
    test_mae = mean_absolute_error(ytest_pred, y_test)
    print(f'rfmodel test mae is:',test_mae)

    return rfmodel


import pickle

def pickle_model(m,data):
    # Save the trained model on the drive
    pickle.dump(m, open('RE_Model','wb'))

    # Load the pickled model
    RE_Model = pickle.load(open('RE_Model','rb'))

    # Use the loaded pickled model to make predictions
    return(RE_Model.predict(data))