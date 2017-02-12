#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:59:07 2017

@author: guanchu
"""


import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import math

from sklearn import cross_validation
from sklearn.cross_validation import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sknn.mlp import Regressor, Layer
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error



data = pd.read_csv('network_backup_dataset.csv')
data.columns = ['week', 'day_of_week', 'start_time','work_flow','file_name','size','duration']

#============================Pre-process day of week===========================
day_of_week_set = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
for day_of_week in day_of_week_set:
    day_of_week_column_to_be_added = []
    for day_of_week_item in data['day_of_week']:
        if day_of_week_item == day_of_week:
            day_of_week_column_to_be_added.append(1)
        else:
            day_of_week_column_to_be_added.append(0)
    data.insert(data.shape[1], day_of_week, day_of_week_column_to_be_added)
data.drop('day_of_week', 1, inplace = True)
#==============================================================================

#============================Pre-process start time============================
start_time_set = [1,5,9,13,17,21]
for start_time in start_time_set:
    start_time_column_to_be_added = []
    for start_time_item in data['start_time']:
        if start_time_item == start_time:
            start_time_column_to_be_added.append(1)
        else:
            start_time_column_to_be_added.append(0)
    data.insert(data.shape[1], 'start_time_'+str(start_time), start_time_column_to_be_added)
data.drop('start_time', 1, inplace = True)
#==============================================================================

#=============================Pre-process work flow============================
work_flow_set = ['work_flow_0','work_flow_1','work_flow_2','work_flow_3','work_flow_4']
for work_flow in work_flow_set:
    work_flow_column_to_be_added = []
    for work_flow_item in data['work_flow']:
        if work_flow_item == work_flow:
            work_flow_column_to_be_added.append(1)
        else:
            work_flow_column_to_be_added.append(0)
    data.insert(data.shape[1], work_flow, work_flow_column_to_be_added)
data.drop('work_flow', 1, inplace = True)
#==============================================================================

#==============================Pre-process file name===========================
file_name_set = set([])
for file_name_item in data['file_name']:
    file_name_set.add(file_name_item)
for file_name in file_name_set:
    file_name_column_to_be_added = []
    for file_name_item in data['file_name']:
        if file_name_item == file_name:
            file_name_column_to_be_added.append(1)
        else:
            file_name_column_to_be_added.append(0)
    data.insert(data.shape[1], file_name, file_name_column_to_be_added)
data.drop('file_name', 1, inplace = True)
#==============================================================================

#============================Save pre-processed data===========================
data.to_csv('revised_data.csv', index = False)
data = pd.read_csv('revised_data.csv')
#==============================================================================


def calculate_RMSE(predicted, actual):  
    return math.sqrt(mean_squared_error(actual, predicted)) 

#===========================Neural Network Fitting=============================
training_data = data.copy()
training_data.drop('duration', 1, inplace = True)
target_data = training_data.pop('size')

#cross validation
X_train,X_test,y_train,y_test = cross_validation.train_test_split(training_data.values, target_data.values, test_size=0.1, random_state = 42)


i = 0.1
neu_net_reg = Regressor(layers=[Layer("Sigmoid", units=30),Layer("Linear")],learning_rate=i, n_iter=19)
neu_net_reg.fit(X_train, y_train)
predicted_target_data = neu_net_reg.predict(X_test)
print 'Learning rate: '+ str(i)+'   RMSE is: ' + str(calculate_RMSE(y_test, predicted_target_data))


#==============================================================================






















