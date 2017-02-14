"""
Created on 6:54 PM , 2/12/17, 2017

        by Tommy Tang
        
Project2 - f
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics
import re

computer_technology_subclasses = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                                  'comp.sys.mac.hardware']
recreational_activity_subclasses = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

subclasses_of_documents = computer_technology_subclasses + recreational_activity_subclasses

all_train = fetch_20newsgroups(subset='train', categories=subclasses_of_documents, shuffle=True,
                               random_state=42, remove=('headers', 'footers', 'quotes'))
all_test = twenty_train = fetch_20newsgroups(subset='test', categories=subclasses_of_documents, shuffle=True,
                                             random_state=42, remove=('headers', 'footers', 'quotes'))
stemmer = SnowballStemmer("english")


# ========================= Prepare the data(remove punctuation and transform into tf-idf) ===========================
punctuations = '[! \" # $ % \& \' \( \) \* + , \- \. \/ : ; < = > ? @ \[ \\ \] ^ _ ` { \| } ~]'


def remove_punctuation_and_stem(data_list):
    for j in range(len(data_list)):
        data_list[j] = " ".join([stemmer.stem(data) for data in re.split(punctuations, data_list[j])])
        data_list[j] = data_list[j].replace('\n', '').replace('\t', '').replace('\r', '')


remove_punctuation_and_stem(all_train.data)
remove_punctuation_and_stem(all_test.data)

count_vect = CountVectorizer(min_df=10, stop_words='english')
X_counts = count_vect.fit_transform(all_train.data + all_test.data)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
tfidf = X_counts.toarray()
tfidf_train = np.array(tfidf)

svd = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
svd.fit(tfidf)
X = svd.transform(tfidf)

# ==================================================== Soft margin SVM ===============================================

# Separate the documents into two groups using SVM
comp_tech = np.array([0, 1, 2, 3])
rec_act = np.array([4, 5, 6, 7])

for i in range(len(all_train.target)):
    if all_train.target[i] in comp_tech:
        all_train.target[i] = 0
    elif all_train.target[i] in rec_act:
        all_train.target[i] = 1

for i in range(len(all_test.target)):
    if all_test.target[i] in comp_tech:
        all_test.target[i] = 0
    elif all_test.target[i] in rec_act:
        all_test.target[i] = 1

y = np.append(np.array(all_train.target), np.array(all_test.target))


for gamma in np.logspace(-3, 3, 7):
    test_target = np.array([])
    test_predicted = np.array([])

    folds = KFold(n_splits=5, shuffle=False)

    # Prediction using soft margin SVM with 5-fold cross validation
    for train_index, test_index in folds.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = svm.SVC(C=1, gamma=gamma)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        test_target = np.append(test_target, y_test)
        test_predicted = np.append(test_predicted, predicted)

    print 'Gamma = ' + str(gamma)
    print 'Confusion Matrix:'
    print metrics.confusion_matrix(test_target, test_predicted)
    print 'Classification report:'
    print metrics.classification_report(test_target, test_predicted,
                                        target_names=['Computer Tech',  'Recreational Activity'])
    print 'Accuracy = ' + str(np.mean(test_target == test_predicted))

