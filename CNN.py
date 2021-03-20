import numpy as np # linear algebra
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from utils import *
from data_analysis import *
import pandas as pd 
from tensorflow import keras
from tensorflow.python.keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import RMSprop
import tensorflow as tf
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
    model = keras.Sequential()
    model.add(Dense(64, input_shape=[len(X_train.keys())] ))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=10))
    model.add(Dense(256,kernel_regularizer=l2(0.01),kernel_initializer = 'random_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    rms = RMSprop()
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    
    # The patience parameter is the amount of epochs to check for improvement

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    #model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    model.fit(x = X_train, y = y_train, epochs = 100, batch_size = 16, validation_split=0.2, callbacks=[early_stop])

    
    loss, mae, mse = model.evaluate(X_train, y_train, verbose=2)
    print('Test mae:', mae)
    print('Test loss:', loss)
    print('Test mse:', mse)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}
    
if __name__ == '__main__':
    df = run_analysis()
    #df_transformed = cat_numer(df)
    split_data(df)