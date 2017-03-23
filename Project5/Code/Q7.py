#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:17:17 2017

@author: guanchu
"""

import os
import json
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
import time
from sklearn.ensemble import RandomForestRegressor


month_map = {'Mar' : '03',
            'Feb' : '02',
            'Aug' : '08',
            'Sep' : '09',
            'May' : '05',
            'Jun' : '06',
            'Jul' : '07',
            'Jan' : '01',
            'Apr' : '04',
            'Nov' : '11',
            'Dec' : '12',
            'Oct' : '10',}


def convert_timestamp(t):
    time_str = t[-4:]+'-'+month_map[t[4:7]]+'-'+t[8:19]
    return time.mktime(time.strptime(time_str,'%Y-%m-%d %H:%M:%S'))






#===================== Function extracting data from file =====================
def extract_information(hashtag):
    hashtag_dict = {'#GoHawks' : ['tweets_#gohawks.txt', 188136],
                    '#GoPatriots' : ['tweets_#gopatriots.txt', 26232],
                    '#NFL' : ['tweets_#nfl.txt', 259024],
                    '#Patriots' : ['tweets_#patriots.txt', 489713],
                    '#SB49' : ['tweets_#sb49.txt', 826951],
                    '#SuperBowl' : ['tweets_#superbowl.txt', 1348767]}
    
                    
                    
    #----------------------- Extract data from file ---------------------------    
    time_since_register = [0]*hashtag_dict[hashtag][1]
    default_profile = [0]*hashtag_dict[hashtag][1]
    favourites_count = [0]*hashtag_dict[hashtag][1]
    followers_count = [0]*hashtag_dict[hashtag][1]
    friends_count = [0]*hashtag_dict[hashtag][1]
    geo_enabled  =[0]*hashtag_dict[hashtag][1]
    is_translator = [0]*hashtag_dict[hashtag][1]
    listed_count = [0]*hashtag_dict[hashtag][1]
    protected = [0]*hashtag_dict[hashtag][1]
    verified = [0]*hashtag_dict[hashtag][1]

    total_num_tweets = [0]*hashtag_dict[hashtag][1]
    
    
    
    input_file = open('./Original_data/'+hashtag_dict[hashtag][0])
    for (line, index) in zip(input_file, range(0, hashtag_dict[hashtag][1])):
        data = json.loads(line)
        time_since_register[index] = data['firstpost_date']-convert_timestamp(data['tweet']['user']['created_at'])
        default_profile[index] = 1 if data['tweet']['user']['default_profile'] else 0
        favourites_count[index] = data['tweet']['user']['favourites_count']
        followers_count[index] = data['tweet']['user']['followers_count']
        friends_count[index] = data['tweet']['user']['friends_count']
        geo_enabled [index] = 1 if data['tweet']['user']['geo_enabled'] else 0
        is_translator[index] = 1 if data['tweet']['user']['is_translator'] else 0
        listed_count[index] = data['tweet']['user']['listed_count']
        protected[index] = 1 if data['tweet']['user']['protected'] else 0
        verified[index] = 1 if data['tweet']['user']['verified'] else 0
        total_num_tweets[index] = data['tweet']['user']['statuses_count']
        
    input_file.close()
    

    
    #------------------ Build DataFrame and save to file ----------------------
    extracted_data = np.array([time_since_register,
                                default_profile,
                                favourites_count,
                                followers_count,
                                friends_count,
                                geo_enabled,
                                is_translator,
                                listed_count,
                                protected,
                                verified,
                                total_num_tweets])
    extracted_data = np.transpose(extracted_data)
    df = DataFrame(extracted_data)
    df.columns = ['time_since_register',
                    'default_profile',
                    'favourites_count',
                    'followers_count',
                    'friends_count',
                    'geo_enabled',
                    'is_translator',
                    'listed_count',
                    'protected',
                    'verified',
                    'total_num_tweets']
    if os.path.isdir('./Extracted_data'):
        pass
    else:
        os.mkdir('./Extracted_data')
    df.to_csv('./Extracted_data/Q7_'+hashtag+'.csv', index = False)   
#==============================================================================

#================== Function that performs linear regression===================
def calculate_linear_regression(hashtag):
    training_data = pd.read_csv('./Extracted_data/Q7_'+hashtag+'.csv')
    training_data.drop(np.where(np.isnan(training_data))[0], inplace=True)
    
    '''
    #---------------------------- One-hot encoding ----------------------------
    time_of_day_set = range(0,24)
    for time_of_day in time_of_day_set:
        time_of_day_column_to_be_added = []
        for time_of_day_item in training_data['time_of_day']:
            if time_of_day_item == time_of_day:
                time_of_day_column_to_be_added.append(1)
            else:
                time_of_day_column_to_be_added.append(0)
        training_data.insert(training_data.shape[1]-1,
                             str(time_of_day)+'th_hour',
                             time_of_day_column_to_be_added)
    '''
        
        
        
    #--------------------------- Linear Regression ----------------------------
    target_data = training_data.pop('total_num_tweets')
    
    lin_reg = RandomForestRegressor(n_estimators = 20, max_depth = 80)
    lin_reg_result = lin_reg.fit(training_data, target_data)
    

    
    predicted_values = lin_reg_result.predict(training_data)
    
  
    
    
    #------------------------- Plot scattered figure --------------------------
    fig, ax = plt.subplots()
    ax.scatter(target_data, predicted_values)
    ax.plot([target_data.min(), target_data.max()], [target_data.min(), target_data.max()], 'k--', lw = 4)
    ax.set_xlabel('Actual values', fontsize = 20)
    ax.set_ylabel('Predicted value', fontsize = 20)
    plt.title('Fitted values vs. Actual values', fontsize = 20)
    plt.show()
    
    plt.clf
    fig, ax = plt.subplots()
    ax.scatter(target_data, (predicted_values - target_data)/target_data)
    ax.plot([0,max(target_data)], [0,0], 'k--', lw = 4)
    ax.set_xlabel('Actual values', fontsize = 20)
    ax.set_ylabel('Relative Error', fontsize = 20)
    plt.title('Relative Error vs. Actual values', fontsize = 20)
    #plt.axis([0,10000,-1200,400])
    plt.show()
    
    
    
    
    
    total_error = 0.0
    for (actual, predicted) in zip(target_data, predicted_values):
        total_error += abs(actual - predicted)
    print 'Average prediction error:',total_error/len(target_data)
    
#==============================================================================
    
    
def obtain_data_and_perform_linear_regression(hashtag):
    print '#######################################################\n'
    print 'Processing hashtag "' + hashtag + '"......\n'
    extract_information(hashtag)
    calculate_linear_regression(hashtag)
    
    
    
obtain_data_and_perform_linear_regression('#GoHawks')
obtain_data_and_perform_linear_regression('#GoPatriots')
obtain_data_and_perform_linear_regression('#NFL')
obtain_data_and_perform_linear_regression('#Patriots')
obtain_data_and_perform_linear_regression('#SB49')
obtain_data_and_perform_linear_regression('#SuperBowl')

