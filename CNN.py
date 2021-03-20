import numpy as np # linear algebra
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from utils import *
from data_analysis import *
import pandas as pd 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


#Prediction

#first seperate categorical and numerical data to apply different transformations
'''
Function: cat_numer
    Function to separate categorical and numerical data to apply different transformations.
Parameter: 
    Data: DataSet
Return:
    DataSet transformed df_transformed
''' 
def cat_numer(data):
    print(data.head())
    y = data.condition
    df_transformed = data.copy(deep=True)
    numeric_features = ['age','trestbps','chol','thalach','ca']
    categorical_features = ['sex','cp','fbs','restecg','exang','slope','thal']

    enc = OneHotEncoder(sparse=False,drop='first')
    enc.fit(df_transformed[categorical_features])
    
    col_names = enc.get_feature_names(categorical_features)
    print(col_names)
    df_transformed = pd.concat([df_transformed.drop(categorical_features, 1),
          pd.DataFrame(enc.transform(df_transformed[categorical_features]),columns = col_names)], axis=1).reindex()
    print(df_transformed.head())

    #Categorical are on onehotencoder
    scaler = StandardScaler()
    df_transformed[numeric_features]  = scaler.fit_transform(df_transformed[numeric_features])
    df_transformed.head()
    return df_transformed
'''
Function: split_data
    Function to split the data between test and train.
Parameter: 
    Data: DataSet
Return:
    Node
''' 
def split_data(df_transformed):
    
    #Split the data 
    X_train, X_test, y_train, y_test = train_test_split(df_transformed.drop('condition',axis=1), df_transformed['condition'], test_size = .2, random_state=10)
    rf_model = RandomForestClassifier(max_depth=5, random_state=137)
    rf_model.fit(X_train, y_train)

if __name__ == '__main__':
    df = run_analysis()
    df_transformed = cat_numer(df)
    split_data(df_transformed)