
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn import svm
from sklearn.cross_validation import cross_val_predict
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

classification_all = [1] * (len(comp_tech_train.data)+len(comp_tech_test.data)) + [-1] * (len(rec_act_train.data)+len(rec_act_test.data))
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



for gamma_value in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    classifier = svm.SVC(gamma = gamma_value)
    classifier.fit(LSI, classification_all)
    predicted_class =  cross_val_predict(classifier, LSI, classification_all, cv=5)
    
    
    pound_sign = ''
    spaces = ''
    for i in range(len(str(gamma_value))):
        pound_sign += '#'
        spaces += ' '
    print '#################' + pound_sign +'\n#               ' + spaces + '#'
    print '# Gamma value: ' + str(gamma_value) + ' #'
    print '#               ' + spaces + '#\n#################' + pound_sign
    
    print '\n              Classification report:'
    print '======================================================='
    print metrics.classification_report(classification_all, predicted_class, target_names=["Com Tech","Rec Act"]),
    print '=======================================================\n'
    
    print 'Confusion Matrix:'
    print '=============='
    print metrics.confusion_matrix(classification_all, predicted_class) 
    print '==============\n'

    print 'Total accuracy: '
    print np.mean(predicted_class == classification_all)
    print '\n\n\n\n'



