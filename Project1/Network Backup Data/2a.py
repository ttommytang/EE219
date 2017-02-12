#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:53:08 2017

@author: guanchu
"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_predict





data = pd.read_csv('network_backup_dataset.csv')
data.columns = ['week', 'day_of_week', 'start_time','work_flow','file_name','size','duration']

'''
#================================Pre-process week==============================
week_set = set([])
for week_item in data['week']:
    week_set.add(week_item)
for week in week_set:
    week_column_to_be_added = []
    for week_item in data['week']:
        if week_item == week:
            week_column_to_be_added.append(1)
        else:
            week_column_to_be_added.append(0)
    data.insert(data.shape[1], 'week_'+str(week), week_column_to_be_added) 
data.drop('week', 1, inplace = True)
#==============================================================================
'''
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
    return sp.sqrt(sp.mean((predicted - actual) ** 2))  

#================================Linear Regression=============================
training_data = data.copy()
training_data.drop('duration', 1, inplace = True)
target_data = training_data.pop('size')

lin_reg = linear_model.LinearRegression(fit_intercept = False)
lin_reg_result = lin_reg.fit(training_data, target_data)

print lin_reg.coef_
print lin_reg.intercept_

predicted_target_data = cross_val_predict(lin_reg, training_data, target_data, cv = 10)


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

print "RMSE for this model is : " + str(calculate_RMSE(predicted_target_data, target_data))
#==============================================================================





#=========================Evaluate importance of week==========================
print "\nIgnoring week number:"
training_data = data.copy()
training_data.drop('duration', 1, inplace = True)
training_data.drop('week', 1, inplace = True)
target_data = training_data.pop('size')


lin_reg = linear_model.LinearRegression(fit_intercept = False)
lin_reg_result = lin_reg.fit(training_data, target_data)
'''
print lin_reg.coef_
print lin_reg.intercept_
'''
predicted_target_data = cross_val_predict(lin_reg, training_data, target_data, cv = 10)

'''#plot
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
'''
print "      RMSE for this model is : " + str(calculate_RMSE(predicted_target_data, target_data))
#==============================================================================




#=====================Evaluate importance of day of week=======================
print "\nIgnoring day of week:"
training_data = data.copy()
training_data.drop('duration', 1, inplace = True)
for day in day_of_week_set:
    training_data.drop(day, 1, inplace = True)
target_data = training_data.pop('size')


lin_reg = linear_model.LinearRegression(fit_intercept = False)
lin_reg_result = lin_reg.fit(training_data, target_data)
'''
print lin_reg.coef_
print lin_reg.intercept_
'''
predicted_target_data = cross_val_predict(lin_reg, training_data, target_data, cv = 10)

'''#plot
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
'''
print "      RMSE for this model is : " + str(calculate_RMSE(predicted_target_data, target_data))
#==============================================================================



#=====================Evaluate importance of start time========================
print "\nIgnoring start time:"
training_data = data.copy()
training_data.drop('duration', 1, inplace = True)
for start_time in start_time_set:
    training_data.drop('start_time_'+str(start_time), 1, inplace = True)
target_data = training_data.pop('size')


lin_reg = linear_model.LinearRegression(fit_intercept = False)
lin_reg_result = lin_reg.fit(training_data, target_data)
'''
print lin_reg.coef_
print lin_reg.intercept_
'''
predicted_target_data = cross_val_predict(lin_reg, training_data, target_data, cv = 10)

'''#plot
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
'''
print "      RMSE for this model is : " + str(calculate_RMSE(predicted_target_data, target_data))
#==============================================================================



#=====================Evaluate importance of work flow=========================
print "\nIgnoring work flow:"
training_data = data.copy()
training_data.drop('duration', 1, inplace = True)
for work_flow in work_flow_set:
    training_data.drop(work_flow, 1, inplace = True)
target_data = training_data.pop('size')


lin_reg = linear_model.LinearRegression(fit_intercept = False)
lin_reg_result = lin_reg.fit(training_data, target_data)
'''
print lin_reg.coef_
print lin_reg.intercept_
'''
predicted_target_data = cross_val_predict(lin_reg, training_data, target_data, cv = 10)

'''#plot
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
'''
print "      RMSE for this model is : " + str(calculate_RMSE(predicted_target_data, target_data))
#==============================================================================


#=====================Evaluate importance of file name=========================
print "\nIgnoring file name:"
training_data = data.copy()
training_data.drop('duration', 1, inplace = True)
for file_name in file_name_set:
    training_data.drop(file_name, 1, inplace = True)

target_data = training_data.pop('size')


lin_reg = linear_model.LinearRegression(fit_intercept = False)
lin_reg_result = lin_reg.fit(training_data, target_data)
'''
print lin_reg.coef_
print lin_reg.intercept_
'''
predicted_target_data = cross_val_predict(lin_reg, training_data, target_data, cv = 10)

'''#plot
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
'''
print "      RMSE for this model is : " + str(calculate_RMSE(predicted_target_data, target_data))
#==============================================================================


#=============Evaluate importance of file name & work flow=====================
print "\nIgnoring file name & work flow:"
training_data = data.copy()
training_data.drop('duration', 1, inplace = True)
for file_name in file_name_set:
    training_data.drop(file_name, 1, inplace = True)
for work_flow in work_flow_set:
    training_data.drop(work_flow, 1, inplace = True)
target_data = training_data.pop('size')


lin_reg = linear_model.LinearRegression(fit_intercept = False)
lin_reg_result = lin_reg.fit(training_data, target_data)
'''
print lin_reg.coef_
print lin_reg.intercept_
'''
predicted_target_data = cross_val_predict(lin_reg, training_data, target_data, cv = 10)

'''#plot
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
'''
print "      RMSE for this model is : " + str(calculate_RMSE(predicted_target_data, target_data))
#==============================================================================


















