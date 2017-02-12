#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:59:07 2017

@author: guanchu
"""


import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sknn.mlp import Regressor, Layer
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


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





#============================Catagorize the dataset============================
data.drop('duration', 1, inplace = True)

training_wf0 = data[data.work_flow_0 == 1]
training_wf1 = data[data.work_flow_1 == 1]
training_wf2 = data[data.work_flow_2 == 1]
training_wf3 = data[data.work_flow_3 == 1]
training_wf4 = data[data.work_flow_4 == 1]

target_wf0 = training_wf0.pop('size')
target_wf1 = training_wf1.pop('size')
target_wf2 = training_wf2.pop('size')
target_wf3 = training_wf3.pop('size')
target_wf4 = training_wf4.pop('size')

training_list = [training_wf0, training_wf1, training_wf2, training_wf3, training_wf4]
target_list = [target_wf0, target_wf1, target_wf2, target_wf3, target_wf4]

for i in range(0, 5):
    for work_flow in work_flow_set:
        training_list[i].drop(work_flow, 1, inplace = True)
#==============================================================================


def calculate_RMSE(predicted, actual):  
    return sp.sqrt(sp.mean((predicted - actual) ** 2))  


#=========================Piece-wise Linear Regression=========================

RMSE_list = []
for i in range(0, 5):
    print '======================================================'
    print '======================================================'
    print 'Calculating work flow '+str(i)
    training_data = training_list[i]
    target_data = target_list[i]
    lin_reg = LinearRegression()
    lin_reg_result = lin_reg.fit(training_data, target_data)
    predicted_target_data = cross_val_predict(lin_reg, training_data, target_data, cv = 10)
    RMSE_list.append(calculate_RMSE(predicted_target_data, target_data))
    
    '''
    fig, ax = plt.subplots()
    ax.scatter(target_data, predicted_target_data)
    ax.plot([target_data.min(), target_data.max()], [target_data.min(), target_data.max()], 'k--', lw = 4)
    ax.set_xlabel('Measured size', fontsize = 20)
    ax.set_ylabel('Predicted size', fontsize = 20)
    plt.title('Fitted values vs. Actual values', fontsize = 20)
    
    print "Fitted values vs. Actual values:"
    plt.show()
    
    
    
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(predicted_target_data, predicted_target_data-target_data)
    
    ax.set_xlabel('Predicted size', fontsize = 20)
    ax.set_ylabel('Residual', fontsize = 20)
    plt.title('Residuals vs. Fitted value', fontsize = 20)
    print "Residuals vs. Fitted value"
    plt.show()
    
    print "RMSE for this model is : " + str(RMSE_list[i])
    '''

print 'Summary:'
for i in range(0, 5):
    print '   Linear Regression RMSE for work flow '+str(i)+' is: '+str(RMSE_list[i])
#==============================================================================






#=============================Polynomial Regression============================
training_data = data.copy()
target_data = training_data.pop('size')


#Try to ignore irrelevant attributes - file name
for file_name in file_name_set:
    training_data.drop(file_name, 1, inplace = True)

training_data.drop('week', 1, inplace = True)
    
cross_validated_RMSE = []
not_cross_validated_RMSE = []

for i in range(1, 5):
    poly_feat = PolynomialFeatures(degree=i,include_bias=False, interaction_only = True)
    lin_reg = LinearRegression()
    pl = Pipeline([("Poly Feat", poly_feat), ("Lin Reg", lin_reg)])
    pl.fit(training_data, target_data)

    cross_validated_predicted_target_data = cross_val_predict(pl, training_data, target_data, cv = 10)
    predicted_target_data = pl.predict(training_data)
    
    cross_validated_RMSE.append(calculate_RMSE(cross_validated_predicted_target_data, target_data))
    not_cross_validated_RMSE.append(calculate_RMSE(predicted_target_data, target_data))
    print 'degree: '+str(i)+'  cross validated RMSE:'+str(cross_validated_RMSE[i-1])
    print 'degree: '+str(i)+'           normal RMSE:'+str(not_cross_validated_RMSE[i-1])

    
    
plt.figure()

plt.plot(range(1, len(cross_validated_RMSE)+1), cross_validated_RMSE)
plt.ylabel('Cross Validated RMSE', fontsize = 20)
plt.xlabel('Degree of Polynomial', fontsize = 20)
plt.title('Cross Validated RMSE   vs.   Degree of Polynomial', fontsize = 20)


plt.show()
    


plt.clf()

    
plt.figure()

plt.plot(range(1, len(not_cross_validated_RMSE)+1), not_cross_validated_RMSE)
plt.ylabel('Not Cross Validated RMSE', fontsize = 20)
plt.xlabel('Degree of Polynomial', fontsize = 20)
plt.title('Not Cross Validated RMSE   vs.   Degree of Polynomial', fontsize = 20)


plt.show()

#==============================================================================



























