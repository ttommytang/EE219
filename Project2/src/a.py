#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 15:37:35 2017

@author: guanchu
"""

from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt

computer_technology_subclasses = [['comp.graphics'], ['comp.os.ms-windows.misc'], ['comp.sys.ibm.pc.hardware'],
                                  ['comp.sys.mac.hardware']]
recreational_activity_subclasses = [['rec.autos'], ['rec.motorcycles'], ['rec.sport.baseball'], ['rec.sport.hockey']]


# ==============================Function definition=============================
# A function that plots a histogram
def plot_histogram(target_set):
    substring_pair = {'all': '(all subsets):', 'train': '(only training subsets)', 'test': '(only testing subsets)'}
    print ''
    print 'Number of documents per topic ' + substring_pair[target_set]
    number_of_documents = []
    number_of_documents_in_comp_tech = 0
    number_of_documents_in_rec_act = 0
    for i in range(4):
        number = len(fetch_20newsgroups(subset=target_set, categories=computer_technology_subclasses[i], shuffle=True,
                                        random_state=42).data)
        number_of_documents.append(number)
        number_of_documents_in_comp_tech += number
    for i in range(4):
        number = len(fetch_20newsgroups(subset=target_set, categories=recreational_activity_subclasses[i], shuffle=True,
                                        random_state=42).data)
        number_of_documents.append(number)
        number_of_documents_in_rec_act += number
    subclasses_of_ducuments = computer_technology_subclasses + recreational_activity_subclasses

    for i in range(8):
        spaces = ''
        for j in range(26 - len(subclasses_of_ducuments[i][0])):
            spaces += ' '
        print spaces + subclasses_of_ducuments[i][0] + ' : ' + str(number_of_documents[i])
    print ''

    print 'Number of documents in Computer Technology: ' + str(number_of_documents_in_comp_tech)
    print 'Number of documents in Recreational Activity: ' + str(number_of_documents_in_rec_act)
    # --------------------------------------------------------------------------
    print ''
    print 'Histogram of the number of documents per topic ' + substring_pair[target_set]

    x_labels = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

    fig, ax = plt.subplots()
    ax.set_xticks([i + 0.5 for i in range(1, 9)])
    ax.set_xticklabels(x_labels, rotation=60, ha='right', fontsize=13)

    rects = plt.bar([i for i in range(1, 9)], number_of_documents, 0.5, align='edge')
    plt.xlabel('Topic Name', fontsize=15)
    plt.ylabel('Number of Documents', fontsize=15)
    plt.title('Number of documents per topic ' + substring_pair[target_set], fontsize=20)
    plt.axis([0.5, 9, 0, 1100])

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1 * height, '%d' % int(height), ha='center', va='bottom')

    plt.show()


# ==============================================================================

plot_histogram('all')
plot_histogram('train')
plot_histogram('test')
