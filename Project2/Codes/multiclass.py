

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import re
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk import SnowballStemmer
import numpy as np
from sklearn import metrics

def load_dataset(category_list):
    train = fetch_20newsgroups(subset='train',  shuffle=True, random_state=42, categories=category_list)
    test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42, categories=category_list)
    
    return train, test

category = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']
training_data, testing_data = load_dataset(category)    
all_data = training_data.data+testing_data.data
#==============================================================================
    

    



stemmer = SnowballStemmer("english")
punctuations = '[! \" # $ % \& \' \( \) \* + , \- \. \/ : ; < = > ? @ \[ \\ \] ^ _ ` { \| } ~]'    
def preprocess_data(data_list):
    for i in range(len(data_list)):
        data_list[i] = " ".join([stemmer.stem(data) for data in re.split(punctuations, data_list[i])])
        data_list[i] = data_list[i].replace('\n','').replace('\t','').replace('\r','')
    
preprocess_data(all_data)
#==============================================================================






# using CountVectorizer and TFxIDF Transformer
count_vect = CountVectorizer(min_df=10, stop_words ='english')
X_counts = count_vect.fit_transform(all_data)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
#==============================================================================




# apply LSI to TDxIDF matrices
svd = TruncatedSVD(n_components = 50, n_iter = 10,random_state = 42)
svd.fit(X_tfidf)
LSI = svd.transform(X_tfidf)
#==============================================================================

        

LSI_train = LSI[0:len(training_data.data)]
LSI_test = LSI[len(training_data.data):]
#==============================================================================



print("Size of Transformed Training Dataset: {0}".format(LSI_train.shape))
print("Size of Transformed Testing Dataset: {0}".format(LSI_test.shape))
        


def calculate_statistics(target, predicted):
    
    print '\n                       Classification Report:'
    print '=================================================================='
    print metrics.classification_report(target, predicted, target_names=['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']),
    print '==================================================================\n'
    
    print 'Confusion Matrix:'
    print '==================='
    print metrics.confusion_matrix(target, predicted) 
    print '===================\n'
    
    print 'Total Accuracy: '
    print np.mean(target == predicted)
    
    
clf_list = [OneVsOneClassifier(GaussianNB()), OneVsOneClassifier(svm.LinearSVC()), OneVsRestClassifier(GaussianNB()), OneVsRestClassifier(svm.LinearSVC())]
clf_name = ['One vs One Classifier - Naive Bayes', 'One vs One Classifier - SVM','One vs Rest Classifier - Naive Bayes', 'One vs Rest Classifier - SVM']

# perform classification
for clf,clf_n in zip(clf_list,clf_name):
    pound_sign = ''
    spaces = ''
    for i in range(len(clf_n)+2):
        pound_sign += '#'
        spaces += ' '
    print '\n\n\n\n'
    print '#' + pound_sign + '#'
    print '#' + spaces + '#'
    print '# ' + clf_n + ' #'
    print '#' + pound_sign + '#'
    
    clf.fit(LSI_train, training_data.target)
  
    test_predicted = clf.predict(LSI_test)
    calculate_statistics(testing_data.target, test_predicted)
    
    
    
    
    

    
