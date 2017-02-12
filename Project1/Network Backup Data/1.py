
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 23:53:03 2017

@author: guanchu
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('network_backup_dataset.csv')
data.columns = ['week', 'day_of_week_str', 'start_time','work_flow_str','file_name_str','size','duration']

#==============================Replace day of week=============================
map_day = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
day_of_week = []
for day_of_week_item in data['day_of_week_str']:
    day_of_week.append(map_day[day_of_week_item])
data.insert(2, 'day_of_week', day_of_week)
data.drop('day_of_week_str', 1, inplace = True)
#==============================================================================

#===============================Replace work flow==============================
work_flow = []
for work_flow_item in data['work_flow_str']:
    work_flow.append(int(work_flow_item[10:]))
data.insert(3, 'work_flow', work_flow)
data.drop('work_flow_str', 1, inplace = True)
#==============================================================================

#==============================Replace file name===============================
file_name = []
for file_name_item in data['file_name_str']:
    file_name.append(int(file_name_item[5:]))
data.insert(4, 'file_name', file_name)
data.drop('file_name_str', 1, inplace = True)
#==============================================================================

#==============================Add day column==================================
week_column = data['week']
day_of_week_column = data['day_of_week']
day = []
for i in range(0, 18588):
    day.append((week_column[i]-1)*7+day_of_week_column[i])
data.insert(2, 'day', day)
#==============================================================================





#================================Get plot data=================================
size_column = data['size']
work_flow_column = data['work_flow']
plot_day_limit = 20
total_size_per_work_flow_per_day = {0:[0.0]*plot_day_limit, 1:[0.0]*plot_day_limit, 2:[0.0]*plot_day_limit, 3:[0.0]*plot_day_limit, 4:[0.0]*plot_day_limit}
for i in range(0, 18588):
    if day[i] > plot_day_limit:
        break
    total_size_per_work_flow_per_day[work_flow_column[i]][day[i]-1] += size_column[i]
#------------------------------------------------------------------------------
plot_week_limit = 15
fixed_day_of_week = 'Saturday'
start_time_column = data['start_time']
total_size_per_same_day_of_week = []
for i in range(0, 16):
    total_size_per_same_day_of_week.append([0.0]*6)
for i in range(0, 18588):
    if week_column[i] > plot_week_limit:
        break
    if day_of_week[i] == map_day[fixed_day_of_week]:
        total_size_per_same_day_of_week[int(week_column[i]-1)][int((start_time_column[i]-1)/4)] += size_column[i]
#==============================================================================



#================================Plot graphs===================================
plt.figure(figsize = (15,7.5))
style_list = ['s', '8', '^', 'D', '*']
color_list = ['b', 'g', 'r', 'c', 'm']
for i in range(0, 5):
    plt.plot(range(1, plot_day_limit+1), total_size_per_work_flow_per_day[i], style_list[i], c = color_list[i], ms = 9.0, label = 'Work Flow '+str(i))
    plt.plot(range(1, plot_day_limit+1), total_size_per_work_flow_per_day[i], linewidth = 2.0, c = color_list[i])
plt.ylabel('Total Backup Size (GB)', fontsize = 20)
plt.xlabel('Days', fontsize = 20)
plt.title('Total Backup Size   vs.   Days', fontsize = 20)
plt.axis([0,plot_day_limit+1,0,12])
plt.grid(True)
plt.legend(loc = 'upper right', bbox_to_anchor = (0.94, 1), fontsize=15,numpoints = 1)
plt.show()
#------------------------------------------------------------------------------
plt.clf()
plt.figure(figsize = (15,7.5))
for line in total_size_per_same_day_of_week:
    plt.plot([1,5,9,13,17,21], line)
plt.ylabel('Total Backup Size at the Moment (GB)', fontsize = 20)
plt.xlabel('Hours in '+fixed_day_of_week, fontsize = 20)
plt.grid(True)
plt.title('Total Backup Size at Different Moments    vs.   Hours in '+fixed_day_of_week, fontsize = 20)
plt.axis([0,22,0,4])
plt.show()

#==============================================================================




















