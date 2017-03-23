#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:17:17 2017

@author: guanchu
"""


import json
import matplotlib.pyplot as plt


#================================= Plot Function ==============================
def plot_histogram(time_stamps, hashtag):
    hourly_tweet_count = [0] * int((max(time_stamps)-min(time_stamps))/3600+1)
    start_time = min(time_stamps)
    for time_stamp in time_stamps:
        hourly_tweet_count[int((time_stamp-start_time)/3600)] += 1
           
    plt.figure(figsize = (12,8))
    plt.bar([i for i in range(0,len(hourly_tweet_count))], hourly_tweet_count, 1, align='edge', color = 'k')
    hours_passed = float(max(time_stamps)-min(time_stamps))/3600.0
    plt.axis([0,hours_passed+1,0,int(max(hourly_tweet_count)*1.03)])
    plt.xlabel('Time(Hour)', fontsize = 15)
    plt.ylabel('Number of Tweets', fontsize = 15)
    plt.title('Number of Tweets over Time (' + hashtag + ')', fontsize = 20)
    plt.show()        
#==============================================================================


#=================== Function calculating required statistics =================
def calculate(hashtag):
    hashtag_dict = {'#GoHawks' : 'tweets_#gohawks.txt',
                    '#GoPatriots' : 'tweets_#gopatriots.txt',
                    '#NFL' : 'tweets_#nfl.txt',
                    '#Patriots' : 'tweets_#patriots.txt',
                    '#SB49' : 'tweets_#sb49.txt',
                    '#SuperBowl' : 'tweets_#superbowl.txt'}
    
    time_stamps = []
    followers_of_users = dict([])
    num_of_followers = []
    num_of_retweets = []
    
    #----------------------- Extract data from file ---------------------------
    input_file = open('./Original_data/'+hashtag_dict[hashtag])
    for line in input_file:
        data = json.loads(line)
        time_stamps.append(data['citation_date'])
        num_of_retweets.append(data['metrics']['citations']['total'])
        num_of_followers.append(data['author']['followers'])
        
        user_name = data['author']['nick']
        if user_name in followers_of_users:
            followers_of_users[user_name].append(data['author']['followers'])
        else:
            followers_of_users[user_name] = [data['author']['followers']]
    input_file.close()
    
    
    #-------------------- Calculate related parameters ------------------------
    total_number_of_tweets = float(len(time_stamps))
    hours_passed = float(max(time_stamps)-min(time_stamps))/3600.0
    
    
    followers = []
    for user in followers_of_users:
        followers.append(float(sum(followers_of_users[user]))/float(len(followers_of_users[user])))
        
    total_number_of_retweets = float(sum(num_of_retweets))
    
    #---------------------------- Print results -------------------------------
    print '\n############################################################\n'
    print 'Statistics for', hashtag
    print '    Total number of tweets:', len(time_stamps)
    print '    Total number of users:', len(followers)
    print '    Average number of tweets per hour:', total_number_of_tweets/hours_passed
    print '    Average number of followers per user:', sum(followers)/len(followers)
    print '    Average number of followers per tweet:', sum(num_of_followers)/len(num_of_followers)
    print '    Average number of retweets per tweet:', total_number_of_retweets/total_number_of_tweets
    
    if hashtag in ['#NFL', '#SuperBowl']:
        plot_histogram(time_stamps, hashtag)
#==============================================================================


    
calculate('#GoHawks')
calculate('#GoPatriots')
calculate('#NFL')
calculate('#Patriots')
calculate('#SB49')
calculate('#SuperBowl')
    