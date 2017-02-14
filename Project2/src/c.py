#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:29:39 2017

@author: guanchu
"""

from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re

stemmer = SnowballStemmer("english")
substring_pair = {'all': 'All subsets included', 'train': 'Only training subsets included',
                  'test': 'Only testing subsets included'}


def ten_most_significant_helper(newsgroup, subset):
    data_list = fetch_20newsgroups(subset=subset, categories=[newsgroup], shuffle=True, random_state=42,
                                   remove=('headers', 'footers', 'quotes')).data
    punctuations = '[! \" # $ % \& \' \( \) \* + , \- \. \/ : ; < = > ? @ \[ \\ \] ^ _ ` { \| } ~]'

    for i in range(len(data_list)):
        data_list[i] = " ".join([stemmer.stem(data) for data in re.split(punctuations, data_list[i])])
        data_list[i] = data_list[i].replace('\n', '').replace('\t', '').replace('\r', '')

    count_vect = CountVectorizer(max_features=10, stop_words='english')
    X_counts = count_vect.fit_transform(data_list)
    print '==================================='
    print 'Newsgroup class:\n    \"' + newsgroup + '\"\n'
    print 'Data subset:\n    ' + substring_pair[subset] + '\n'
    print '10 most significant terms:'

    for (term, count) in zip(count_vect.get_feature_names(), X_counts.toarray().sum(axis=0)):
        spaces = ''
        for i in range(15):
            if 15 - i - len(term) > 0:
                spaces += ' '
        print spaces + '\"' + term + '\" | ' + str(count)

    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    print '\nTFxIDF dimension:\n   ',
    print X_tfidf.shape


def ten_most_significant(newsgroup):
    ten_most_significant_helper(newsgroup, 'all')
    ten_most_significant_helper(newsgroup, 'train')
    ten_most_significant_helper(newsgroup, 'test')


ten_most_significant('comp.sys.ibm.pc.hardware')
ten_most_significant('comp.sys.mac.hardware')
ten_most_significant('misc.forsale')
ten_most_significant('soc.religion.christian')
