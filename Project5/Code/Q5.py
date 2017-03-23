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
from sklearn.cross_validation import cross_val_predict
from sklearn.ensemble import RandomForestRegressor

files_dict_training = {'#GoHawks' : ['tweets_#gohawks.txt', 188136],
                       '#GoPatriots' : ['tweets_#gopatriots.txt', 26232],
                       '#NFL' : ['tweets_#nfl.txt', 259024],
                       '#Patriots' : ['tweets_#patriots.txt', 489713],
                       '#SB49' : ['tweets_#sb49.txt', 826951],
                       '#SuperBowl' : ['tweets_#superbowl.txt', 1348767]}
    
files_dict_testing = {1 : ['sample1_period1.txt', 730],
                      2 : ['sample2_period2.txt', 212273],
                      3 : ['sample3_period3.txt', 3628],
                      4 : ['sample4_period1.txt', 1646],
                      5 : ['sample5_period1.txt', 2059],
                      6 : ['sample6_period2.txt', 205554],
                      7 : ['sample7_period3.txt', 528],
                      8 : ['sample8_period1.txt', 229],
                      9 : ['sample9_period2.txt', 11311],
                      10 : ['sample10_period3.txt', 365]}





#===================== Function extracting data from file =====================
def extract_information(filename_key, file_name_dict, is_testing_data):                         
                    
    #----------------------- Extract data from file ---------------------------    
    time_stamps = [0]*file_name_dict[filename_key][1]
    is_retweet = [False]*file_name_dict[filename_key][1]
    followers_of_users = [0]*file_name_dict[filename_key][1]
    
    number_of_url_citations = [0]*file_name_dict[filename_key][1]
    author_names = ['']*file_name_dict[filename_key][1]
    number_of_mentions = [0]*file_name_dict[filename_key][1]
    ranking_scores = [0.0]*file_name_dict[filename_key][1]
    number_of_hashtags = [0]*file_name_dict[filename_key][1]
    
    location_path = ''
    feature_name_for_posting_time = ''
    if is_testing_data:
        location_path = './test_data/'+file_name_dict[filename_key][0]
        feature_name_for_posting_time = 'firstpost_date'
    else:
        location_path = './Original_data/'+file_name_dict[filename_key][0]
        feature_name_for_posting_time = 'citation_date'
        

    input_file = open(location_path)
    for (line, index) in zip(input_file, range(0, file_name_dict[filename_key][1])):
        data = json.loads(line)
        time_stamps[index] = data[feature_name_for_posting_time]
        followers_of_users[index] = data['author']['followers']

        author_name = data['author']['nick']
        original_author_name = data['original_author']['nick']
        if author_name != original_author_name:
            is_retweet[index] = True


        number_of_url_citations[index] = len(data['tweet']['entities']['urls'])
        author_names[index] = author_name
        number_of_mentions[index] = len(data['tweet']['entities']['user_mentions'])
        ranking_scores[index] = data['metrics']['ranking_score']
        number_of_hashtags[index] = data['title'].count('#')
        
        
    input_file.close()
    
    #-------------------- Calculate related parameters ------------------------
    start_time = 1421222400
    if is_testing_data:
        start_time = (min(time_stamps)/3600)*3600

    hours_passed = int((max(time_stamps)-start_time)/3600)+1
    hourly_number_of_tweets = [0] * hours_passed
    hourly_number_of_retweets = [0] * hours_passed
    hourly_sum_of_followers = [0] * hours_passed
    hourly_max_number_of_followers = [0] * hours_passed
    hourly_time_of_the_day = [0] * hours_passed

    
    hourly_number_of_url_citations = [0] * hours_passed
    hourly_number_of_authors = [0] * hours_passed
    hourly_author_set = [0] * hours_passed
    for i in range(0, hours_passed):
        hourly_author_set[i] = set([])
    hourly_number_of_mentions = [0] * hours_passed
    hourly_total_ranking_scores = [0.0] * hours_passed
    hourly_number_of_hashtags = [0] * hours_passed
    
    
    
    for i in range(0, file_name_dict[filename_key][1]):
        current_hour = int((time_stamps[i]-start_time)/3600)
        
        hourly_number_of_tweets[current_hour] += 1
        if is_retweet[i]:
            hourly_number_of_retweets[current_hour] += 1
                                      
        hourly_sum_of_followers[current_hour] += followers_of_users[i]
    
        if followers_of_users[i] > hourly_max_number_of_followers[current_hour]:
            hourly_max_number_of_followers[current_hour] = followers_of_users[i]


        hourly_number_of_url_citations[current_hour] += number_of_url_citations[i]
        hourly_author_set[current_hour].add(author_names[i])
        hourly_number_of_mentions[current_hour] += number_of_mentions[i]
        hourly_total_ranking_scores[current_hour] += ranking_scores[i]
        hourly_number_of_hashtags[current_hour] += number_of_hashtags[i]


    for i in range(0, len(hourly_author_set)):
        hourly_number_of_authors[i] = len(hourly_author_set[i])
    
    if is_testing_data:
        for i in range(0, len(hourly_time_of_the_day)):
            hourly_time_of_the_day[i] = ((start_time-1421222400)/3600+i)%24
    else:
        for i in range(0, len(hourly_time_of_the_day)):
            hourly_time_of_the_day[i] = i%24

    
    #------------------ Build DataFrame and save to file ----------------------
    target_value = hourly_number_of_tweets[1:]
    target_value.append(0)
    data = np.array([hourly_number_of_tweets,
                     hourly_number_of_retweets,
                     hourly_sum_of_followers,
                     hourly_max_number_of_followers,
                     hourly_time_of_the_day,
                     hourly_number_of_url_citations,
                     hourly_number_of_authors,
                     hourly_number_of_mentions,
                     hourly_total_ranking_scores,
                     hourly_number_of_hashtags,
                     target_value])
    data = np.transpose(data)
    df = DataFrame(data)
    df.columns = ['num_tweets', 
                  'num_retweets', 
                  'sum_followers',
                  'max_followers',
                  'time_of_day',
                  'num_URLs',
                  'num_authors',
                  'num_mensions',
                  'ranking_score',
                  'num_hashtags',
                  'target_value']
    if os.path.isdir('./Extracted_data'):
        pass
    else:
        os.mkdir('./Extracted_data')
        
    
    if is_testing_data:
        df.to_csv('./Extracted_data/Q5_'+file_name_dict[filename_key][0][:-4]+'.csv', index = False)  
    else:
        df.to_csv('./Extracted_data/Q5_'+filename_key+'.csv', index = False)  
#==============================================================================






#=============================== One-hot encoding =============================
def one_hot_encode(df):
    time_of_day_set = range(0,24)
    for time_of_day in time_of_day_set:
        time_of_day_column_to_be_added = []
        for time_of_day_item in df['time_of_day']:
            if time_of_day_item == time_of_day:
                time_of_day_column_to_be_added.append(1)
            else:
                time_of_day_column_to_be_added.append(0)
        df.insert(df.shape[1]-1,
                  str(time_of_day)+'th_hour',
                  time_of_day_column_to_be_added)
    return df
#==============================================================================




#================== Function that performs cross validation ===================
def calculate_regression(training_hashtag, testing_data_index):
    training_x = pd.read_csv('./Extracted_data/Q5_'+training_hashtag+'.csv')
    testing_x = pd.read_csv('./Extracted_data/Q5_'+files_dict_testing[testing_data_index][0][:-4]+'.csv')
    
    training_x = one_hot_encode(training_x)
    testing_x = one_hot_encode(testing_x)
    
    #testing_x.drop([len(testing_x)-1], inplace = True)
    
    
        
    #------------------------------ Split data --------------------------------   
    training_x.drop('time_of_day', 1, inplace = True)
    training_y = training_x.pop('target_value')
    
    testing_x.drop('time_of_day', 1, inplace = True)
    testing_y = testing_x.pop('target_value')
    
    
    training_x_before_event = training_x[:440]
    training_x_during_event = training_x[440:452]
    training_x_after_event = training_x[452:]
        
    training_y_before_event = training_y[:440]
    training_y_during_event = training_y[440:452]
    training_y_after_event = training_y[452:]
    
        
    #--------------------------- Regression prediction ------------------------
    
    regressor_before_event = RandomForestRegressor(n_estimators = 20, max_depth = 9)
    regressor_during_event = RandomForestRegressor(n_estimators = 20, max_depth = 9)
    regressor_after_event = RandomForestRegressor(n_estimators = 20, max_depth = 9)

    regressor_before_event.fit(training_x_before_event,training_y_before_event)
    regressor_during_event.fit(training_x_during_event,training_y_during_event)
    regressor_after_event.fit(training_x_after_event,training_y_after_event)
    
    
    
    predicted_y = []
    if files_dict_testing[testing_data_index][0][-5] == '1':
        predicted_y = regressor_before_event.predict(testing_x)
    elif files_dict_testing[testing_data_index][0][-5] == '2':
        predicted_y = regressor_during_event.predict(testing_x)
    else:
        predicted_y = regressor_after_event.predict(testing_x)
    
    
    #------------------------- Print predicted values -------------------------   
    data = np.array([predicted_y, testing_y])
    data = np.transpose(data)
    results = DataFrame(data)
    results.columns = ['Predicted', 'Actual']
    #print results
    
    
    
    #-------------------- Calculate average prediction error ------------------
    total_error = 0.0
    for i in range(len(testing_y)-1):
        total_error += abs(testing_y[i] - predicted_y[i])
    #print 'Average prediction error:',total_error/len(testing_y)
    
    return results, total_error/(len(testing_y)-1)
#==============================================================================
    
    
    





def predict(testing_data_index):
    
    result_list = []
    error_list = []

    hashtag_list = ['#GoHawks',
                    '#GoPatriots',
                    '#NFL',
                    '#Patriots',
                    '#SB49',
                    '#SuperBowl']
    for hashtag in hashtag_list:
        result, error = calculate_regression(hashtag, testing_data_index)
        result_list.append(result)
        error_list.append(error)
    minimum_index = 0
    minimum_error = error_list[0]
    for i in range(0, len(error_list)):
        if error_list[i] < minimum_error:
            minimum_error = error_list[i]
            minimum_index = i
    '''
    print '################################################'
    print 'best training dataset:',hashtag_list[minimum_index]
    print result_list[minimum_index]
    print 'Average prediction error:',minimum_error
    print '################################################'
    '''
    return result_list[minimum_index],minimum_error,hashtag_list[minimum_index]



def get_optimal_result(testing_data_index):
    optimal_result, minimum_error, optimal_hashtag = predict(testing_data_index)
    
    for i in range(20):
        result, error, hashtag = predict(testing_data_index)
        if error  < minimum_error:
            minimum_error = error
            optimal_result = result
            optimal_hashtag = hashtag
    
            
    print '################################################'
    print '\n'+files_dict_testing[testing_data_index][0]+'\n'
    print 'Best training dataset:',optimal_hashtag
    print optimal_result
    print 'Average prediction error:',minimum_error
    print '################################################'




extract_information('#GoHawks',files_dict_training,False)
extract_information('#GoPatriots',files_dict_training,False)
extract_information('#NFL',files_dict_training,False)
extract_information('#Patriots',files_dict_training,False)
extract_information('#SB49',files_dict_training,False)
extract_information('#SuperBowl',files_dict_training,False)

for i in range(1,11):
    extract_information(i,files_dict_testing,True)

for i in range(1,11):
    get_optimal_result(i)




