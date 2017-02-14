"""
Created on 10:13 PM , 2/12/17, 2017

        by Tommy Tang
        
Project2 - h
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import re
import matplotlib.pyplot as plt

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
X_counts_train = count_vect.fit_transform(all_train.data)
X_counts_test = count_vect.fit_transform(all_test.data)

tfidf_transformer = TfidfTransformer()
X_tfidf_train = tfidf_transformer.fit_transform(X_counts_train)
tfidf_train = X_counts_train.toarray()
tfidf_train = np.array(tfidf_train)

X_tfidf_test = tfidf_transformer.fit_transform(X_counts_test)
tfidf_test = X_counts_test.toarray()
tfidf_test = np.array(tfidf_test)

svd = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
svd.fit(tfidf_train)
X_train = svd.transform(tfidf_train)

svd = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
svd.fit(tfidf_test)
X_test = svd.transform(tfidf_test)

# Separate the documents into two groups
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

y_train = np.array(all_train.target)
y_test = np.array(all_test.target)

# ===================================== Classification using Logistic Regression====================================

clf = LogisticRegression(C=1e5)
clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test)
y_scores = clf.decision_function(X_test)

# ============================================= Evaluate the prediction ==============================================
print 'Classification report:'
print metrics.classification_report(y_test, y_predicted,
                                    target_names=['Computer Tech',  'Recreational Activity'])
print 'Confusion matrix:'
print metrics.confusion_matrix(y_test, y_predicted)
print 'Accuracy = ' + str(np.mean(y_test == y_predicted))

fpr, tpr, threshold = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
plt.plot(fpr, tpr, color='green', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-CURVE of Classification using LR')
plt.plot([0, 1], [0, 1], 'k--', lw=1, color='blue')
plt.show()

