"""
Created on 12:27 AM , 2/13/17, 2017

        by Tommy Tang
        
Project2 - i
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import re

# =================================Obtain data==================================
computer_technology_subclasses = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                                  'comp.sys.mac.hardware']
recreational_activity_subclasses = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

comp_tech_train = fetch_20newsgroups(subset='train', categories=computer_technology_subclasses, shuffle=True,
                                     random_state=42, remove=('headers', 'footers', 'quotes'))
rec_act_train = fetch_20newsgroups(subset='train', categories=recreational_activity_subclasses, shuffle=True,
                                   random_state=42, remove=('headers', 'footers', 'quotes'))
comp_tech_test = fetch_20newsgroups(subset='test', categories=computer_technology_subclasses, shuffle=True,
                                    random_state=42, remove=('headers', 'footers', 'quotes'))
rec_act_test = fetch_20newsgroups(subset='test', categories=recreational_activity_subclasses, shuffle=True,
                                  random_state=42, remove=('headers', 'footers', 'quotes'))

all_data = comp_tech_train.data + comp_tech_test.data + rec_act_train.data + rec_act_test.data

stemmer = SnowballStemmer("english")

classification_train = [1] * len(comp_tech_train.data) + [-1] * len(rec_act_train.data)
classification_test = [1] * len(comp_tech_test.data) + [-1] * len(rec_act_test.data)
# ==============================================================================

# ===================Remove Punctuation & Stem & Stop Words=====================
punctuations = '[! \" # $ % \& \' \( \) \* + , \- \. \/ : ; < = > ? @ \[ \\ \] ^ _ ` { \| } ~]'


def remove_punctuation_and_stem(data_list):
    for i in range(len(data_list)):
        data_list[i] = " ".join([stemmer.stem(data) for data in re.split(punctuations, data_list[i])])
        data_list[i] = data_list[i].replace('\n', '').replace('\t', '').replace('\r', '')


remove_punctuation_and_stem(all_data)

count_vect = CountVectorizer(min_df=10, stop_words='english')
X_counts = count_vect.fit_transform(all_data)
# ==============================================================================


# ============================Feature extraction================================
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

svd = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
svd.fit(X_tfidf)
LSI = svd.transform(X_tfidf)
# ==============================================================================

# ======================Split training and testing data=========================
split_point_1 = len(comp_tech_train.data)
split_point_2 = split_point_1 + len(comp_tech_test.data)
split_point_3 = split_point_2 + len(rec_act_train.data)

LSI_test = np.concatenate((LSI[split_point_1: split_point_2], LSI[split_point_3:]))
LSI_train = np.concatenate((LSI[0:split_point_1], LSI[split_point_2:split_point_3]))
# ==============================================================================

# =================Logistic Regression(L1/L2 regularized)=======================
l1_error = np.array([])
l2_error = np.array([])

for i, C in enumerate(np.logspace(-4, 4, 9)):
    clf_l1 = LogisticRegression(C=C, penalty='l1')
    clf_l1.fit(LSI_train, classification_train)
    clf_l2 = LogisticRegression(C=C, penalty='l2')
    clf_l2.fit(LSI_train, classification_train)

    l1_predicted = clf_l1.predict(LSI_test)
    l2_predicted = clf_l2.predict(LSI_test)

    l1_error = np.append(l1_error, mean_squared_error(l1_predicted, classification_test))
    l2_error = np.append(l2_error, mean_squared_error(l2_predicted, classification_test))

plt.plot(np.logspace(-4, 4, 9), l1_error, label='L1', color='blue', lw=2)
plt.plot(np.logspace(-4, 4, 9), l2_error, label='L2', color='green', lw=2)
plt.xscale('log')
plt.xlabel('Regularization Coefficient')
plt.ylabel('MSE of Classification')
plt.legend()
plt.show()












