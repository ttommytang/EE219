#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:29:39 2017

@author: guanchu
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import re


#=================================Obtain data==================================
computer_technology_subclasses = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
recreational_activity_subclasses = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
  
comp_tech_train = fetch_20newsgroups(subset='train', categories=computer_technology_subclasses, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
rec_act_train = fetch_20newsgroups(subset='train', categories=recreational_activity_subclasses, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
comp_tech_test = fetch_20newsgroups(subset='test', categories=computer_technology_subclasses, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
rec_act_test = fetch_20newsgroups(subset='test', categories=recreational_activity_subclasses, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))

all_data = comp_tech_train.data+comp_tech_test.data+rec_act_train.data+rec_act_test.data

stemmer = SnowballStemmer("english")

classification_train = [1] * len(comp_tech_train.data) + [-1] * len(rec_act_train.data)
classification_test = [1] * len(comp_tech_test.data) + [-1] * len(rec_act_test.data)
#==============================================================================




#===================Remove Punctuation & Stem & Stop Words=====================
punctuations = '[! \" # $ % \& \' \( \) \* + , \- \. \/ : ; < = > ? @ \[ \\ \] ^ _ ` { \| } ~]'
def remove_punctuation_and_stem(data_list):
    for i in range(len(data_list)):
        data_list[i] = " ".join([stemmer.stem(data) for data in re.split(punctuations, data_list[i])])
        data_list[i] = data_list[i].replace('\n','').replace('\t','').replace('\r','')

remove_punctuation_and_stem(all_data)

count_vect = CountVectorizer(min_df=10, stop_words ='english')
X_counts = count_vect.fit_transform(all_data)
#==============================================================================


#============================Feature extraction================================
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

svd = TruncatedSVD(n_components = 50, n_iter = 10,random_state = 42)
svd.fit(X_tfidf)
LSI = svd.transform(X_tfidf)
#==============================================================================



#======================Split training and testing data=========================
split_point_1 = len(comp_tech_train.data)
split_point_2 = split_point_1 + len(comp_tech_test.data)
split_point_3 = split_point_2 + len(rec_act_train.data)

LSI_test = np.concatenate((LSI[split_point_1 : split_point_2], LSI[split_point_3:]))
LSI_train = np.concatenate((LSI[0:split_point_1],LSI[split_point_2:split_point_3]))
#==============================================================================




#============================Predict testing data==============================
classifier = GaussianNB()
classifier.fit(LSI_train, classification_train)

predicted_class = classifier.predict(LSI_test)
actual_class = classification_test
predict_probability = classifier.predict_proba(LSI_test[:])[:,1]


print '                Classification report:'
print '======================================================='
print metrics.classification_report(actual_class, predicted_class, target_names=["Com Tech","Rec Act"]),
print '=======================================================\n'

print 'Confusion Matrix:'
print '=============='
print metrics.confusion_matrix(actual_class, predicted_class) 
print '==============\n'

print 'Total accuracy: '
print np.mean(actual_class == predicted_class)

fpr, tpr, threshold = roc_curve(actual_class, predict_probability)
line = [0, 1]
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1])
plt.ylabel('True Positive Rate', fontsize = 20)
plt.xlabel('False Positive Rate', fontsize = 20)
plt.title('ROC-Curve of Naive Bayes Classification', fontsize = 20)
plt.axis([-0.004, 1, 0, 1.006])
plt.show()
#==============================================================================









