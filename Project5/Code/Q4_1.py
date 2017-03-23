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

#===================== Function extracting data from file =====================
def extract_information(hashtag):
    hashtag_dict = {'#GoHawks' : ['tweets_#gohawks.txt', 188136],
                    '#GoPatriots' : ['tweets_#gopatriots.txt', 26232],
                    '#NFL' : ['tweets_#nfl.txt', 259024],
                    '#Patriots' : ['tweets_#patriots.txt', 489713],
                    '#SB49' : ['tweets_#sb49.txt', 826951],
                    '#SuperBowl' : ['tweets_#superbowl.txt', 1348767]}
    
                    
                    
    #----------------------- Extract data from file ---------------------------    
    time_stamps = [0]*hashtag_dict[hashtag][1]
    is_retweet = [False]*hashtag_dict[hashtag][1]
    followers_of_users = [0]*hashtag_dict[hashtag][1]
    
    number_of_url_citations = [0]*hashtag_dict[hashtag][1]
    author_names = ['']*hashtag_dict[hashtag][1]
    number_of_mentions = [0]*hashtag_dict[hashtag][1]
    ranking_scores = [0.0]*hashtag_dict[hashtag][1]
    number_of_hashtags = [0]*hashtag_dict[hashtag][1]
    
    
    input_file = open('./Original_data/'+hashtag_dict[hashtag][0])
    for (line, index) in zip(input_file, range(0, hashtag_dict[hashtag][1])):
        data = json.loads(line)
        time_stamps[index] = data['citation_date']
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
    hours_passed = int((max(time_stamps)-min(time_stamps))/3600)+1
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
    
    
    start_time = min(time_stamps)
    for i in range(0, hashtag_dict[hashtag][1]):
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
    df.to_csv('./Extracted_data/Q4_1_'+hashtag+'.csv', index = False)   
#==============================================================================



#================== Function that performs cross validation ===================
def calculate_cross_validation(hashtag):
    training_data = pd.read_csv('./Extracted_data/Q4_1_'+hashtag+'.csv')
    
    
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
    
        
        
        
    #--------------------------- Cross Validation -----------------------------
    training_data.drop('time_of_day', 1, inplace = True)
    target_data = training_data.pop('target_value')
    
    lin_reg = LinearRegression(fit_intercept = False)
    
    cross_predicted_values = cross_val_predict(lin_reg,
                                               training_data,
                                               target_data,
                                               cv = 10)
    
    
    '''
    #------------------------- Plot scattered figure --------------------------
    fig, ax = plt.subplots()
    ax.scatter(target_data, cross_predicted_values)
    
    ax.plot([target_data.min(), target_data.max()], 
            [target_data.min(), target_data.max()], 
            'k--', 
            lw = 4)
    
    ax.set_xlabel('Actual values', fontsize = 20)
    ax.set_ylabel('Predicted value', fontsize = 20)
    plt.title('Fitted values vs. Actual values', fontsize = 20)
    plt.show()
    
    plt.clf
    fig, ax = plt.subplots()
    ax.scatter(target_data, (cross_predicted_values - target_data)/target_data)
    ax.plot([0,max(target_data)], [0,0], 'k--', lw = 4)
    ax.set_xlabel('Actual values', fontsize = 20)
    ax.set_ylabel('Relative Error', fontsize = 20)
    plt.title('Relative Error vs. Actual values', fontsize = 20)
    #plt.axis([0,10000,-1200,400])
    plt.show()

    
    
    #----------------------------- Perform t-test -----------------------------
    model = sm.OLS(target_data, training_data)
    results = model.fit()
    print(results.summary())
    '''
    
    #-------------------- Calculate average prediction error ------------------
    total_error = 0.0
    for (actual, predicted) in zip(target_data, cross_predicted_values):
        total_error += abs(actual - predicted)
    print 'Average prediction error for '+hashtag+':',total_error/len(target_data)
#==============================================================================
    
    
def obtain_data_and_perform_cross_validation(hashtag):
    extract_information(hashtag)
    calculate_cross_validation(hashtag)
    
    
    
obtain_data_and_perform_cross_validation('#GoHawks')
obtain_data_and_perform_cross_validation('#GoPatriots')
obtain_data_and_perform_cross_validation('#NFL')
obtain_data_and_perform_cross_validation('#Patriots')
obtain_data_and_perform_cross_validation('#SB49')
obtain_data_and_perform_cross_validation('#SuperBowl')

