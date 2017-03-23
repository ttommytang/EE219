#!/usr/bin/env python
import csv
import json
import random
import re

import matplotlib.pyplot as plt
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn import svm
from sklearn.cross_validation import cross_val_predict
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

__author__ = "Tommy Tang"
__email__ = "tomishere7@gmail.com"

# ============================== Dictionaries containing the main city-names in WA and MA ==============================

WA = ["Washington", "WASHINGTON", "WA", "Aberdeen", "ABERDEEN", "Anacortes", "ANACORTES", "Auburn", "AUBURN",
      "Bellevue", "BELLEVUE", "Bellingham", "BELLINGHAM", "Bremerton", "BREMERTON", "Centralia", "CENTRALIA",
      "Coulee Dam", "COULEE DAM", "Coupeville", "COUPEVILLE", "Ellensburg", "ELLENSBURG", "Ephrata", "EPHRATA",
      "Everett", "EVERETT", "Hoquiam", "HOQUIAM", "Kelso", "KELSO", "Kennewick", "KENNEWICK", "Longview", "LONGVIEW",
      "Moses Lake", "MOSES LAKE", "Oak Harbor", "OAK HARBOR", "Olympia", "OLYMPIA", "Pasco", "PASCO", "Point Roberts",
      "POINT ROBERTS", "Port Angeles", "PORT ANGELES", "Pullman", "PULLMAN", "Puyallup", "PUYALLUP", "Redmond",
      "REDMOND", "Renton", "RENTON", "Richland", "RICHLAND", "Seattle", "SEATTLE", "Spokane", "SPOKANE", "Tacoma",
      "TACOMA", "Walla Walla", "WALLA WALLA", "Wenatchee", "WENATCHEE", "Yakima", "YAKIMA"]

MA = ["Massachusetts", "MASSACHUSETTS", "MA", "Abington", "ABINGTON", "Adams", "ADAMS", "Amesbury", "AMESBURY",
      "Amherst", "AMHERST", "Andover", "ANDOVER", "Arlington", "ARLINGTON", "Athol", "ATHOL", "Attleboro", "ATTLEBORO",
      "Barnstable", "BARNSTABLE", "Bedford", "BEDFORD", "Beverly", "BEVERLY", "Boston", "BOSTON", "Bourne", "BOURNE",
      "Braintree", "BRAINTREE", "Brockton", "BROCKTON", "Brookline", "BROOKLINE", "Cambridge", "CAMBRIDGE", "Canton",
      "CANTON", "Charlestown", "CHARLESTOWN", "Chelmsford", "CHELMSFORD", "Chelsea", "CHELSEA", "Chicopee", "CHICOPEE",
      "Clinton", "CLINTON", "Cohasset", "COHASSET", "Concord", "CONCORD", "Danvers", "DANVERS", "Dartmouth",
      "DARTMOUTH", "Dedham", "DEDHAM", "Dennis", "DENNIS", "Duxbury", "DUXBURY", "Eastham", "EASTHAM", "Edgartown",
      "EDGARTOWN", "Everett", "EVERETT", "Fairhaven", "FAIRHAVEN", "Fall River", "FALL RIVER", "Falmouth", "FALMOUTH",
      "Fitchburg", "FITCHBURG", "Framingham", "FRAMINGHAM", "Gloucester", "GLOUCESTER", "Great Barrington",
      "GREAT BARRINGTON", "Greenfield", "GREENFIELD", "Groton", "GROTON", "Harwich", "HARWICH", "Haverhill",
      "HAVERHILL", "Hingham", "HINGHAM", "Holyoke", "HOLYOKE", "Hyannis", "HYANNIS", "Ipswich", "IPSWICH", "Lawrence",
      "LAWRENCE", "Lenox", "LENOX", "Leominster", "LEOMINSTER", "Lexington", "LEXINGTON", "Lowell", "LOWELL", "Ludlow",
      "LUDLOW", "Lynn", "LYNN", "Malden", "MALDEN", "Marblehead", "MARBLEHEAD", "Marlborough", "MARLBOROUGH", "Medford",
      "MEDFORD", "Milton", "MILTON", "Nahant", "NAHANT", "Natick", "NATICK", "New Bedford", "NEW BEDFORD",
      "Newburyport", "NEWBURYPORT", "Newton", "NEWTON", "North Adams", "NORTH ADAMS", "Northampton", "NORTHAMPTON",
      "Norton", "NORTON", "Norwood", "NORWOOD", "Peabody", "PEABODY", "Pittsfield", "PITTSFIELD", "Plymouth",
      "PLYMOUTH", "Provincetown", "PROVINCETOWN", "Quincy", "QUINCY", "Randolph", "RANDOLPH", "Revere", "REVERE",
      "Salem", "SALEM", "Sandwich", "SANDWICH", "Saugus", "SAUGUS", "Somerville", "SOMERVILLE", "South Hadley",
      "SOUTH HADLEY", "Springfield", "SPRINGFIELD", "Stockbridge", "STOCKBRIDGE", "Stoughton", "STOUGHTON",
      "Sturbridge", "STURBRIDGE", "Sudbury", "SUDBURY", "Taunton", "TAUNTON", "Tewksbury", "TEWKSBURY", "Truro",
      "TRURO", "Watertown", "WATERTOWN", "Webster", "WEBSTER", "Wellesley", "WELLESLEY", "Wellfleet", "WELLFLEET",
      "West Bridgewater", "WEST BRIDGEWATER", "West Springfield", "WEST SPRINGFIELD", "Westfield", "WESTFIELD",
      "Weymouth", "WEYMOUTH", "Whitman", "WHITMAN", "Williamstown", "WILLIAMSTOWN", "Woburn", "WOBURN", "Woods Hole",
      "WOODS HOLE", "Worcester", "WORCESTER"]


# ======================================================================================================================

# =========================== Pre-processing the tweets into content list and labels ===================================

tweet_highlight = []
location_set = set([])
location_list = []
tweet_labels = []

input_file = open('./Original_data/tweets_#superbowl.txt')

for line in input_file:
    data = json.loads(line)
    tweet_highlight.append(data['highlight'])
    location_set.add(data['tweet']['user']['location'])
    location_list.append(data['tweet']['user']['location'])

selected_WA = set([])
selected_MA = set([])
WA_set = set(WA)
MA_set = set(MA)

count_set_WA = 0
count_set_MA = 0
location_to_include = WA + MA

for state in location_to_include:
    for location in location_set:
        if state in location:
            if state in WA_set and 'DC' not in location and 'D.C.' not in location:
                selected_WA.add(location)
                count_set_WA += 1
            elif state in MA_set:
                selected_MA.add(location)
                count_set_MA += 1

count_list_WA = 0
count_list_MA = 0
for location in location_list:
    if location in selected_WA:
        count_list_WA += 1
        tweet_labels.append(1)
    elif location in selected_MA:
        count_list_MA += 1
        tweet_labels.append(-1)
    else:
        tweet_labels.append(0)

# Remove the non-related tweets and labels
content_list = []
true_labels = []

for (token, label) in zip(tweet_highlight, tweet_labels):
    if label != 0:
        content_list.append(token)
        true_labels.append(label)

csv_out = open('true_labels.csv', 'wb')
myWriter = csv.writer(csv_out)
myWriter.writerow(true_labels)

# ======================================================================================================================

# ====================================== Remove the punctuations and stems =============================================

stemmer = SnowballStemmer("english")
punctuations = '[! \" # $ % \& \' \( \) \* + , \- \. \/ : ; < = > ? @ \[ \\ \] ^ _ ` { \| } ~]'


def remove_punctuation_and_stem(data_list):
    for i in range(len(data_list)):
        data_list[i] = re.sub(r"https?:\/\/\S+", '', data_list[i])
        data_list[i] = " ".join([stemmer.stem(data) for data in re.split(punctuations, data_list[i])])
        data_list[i] = data_list[i].replace('\n', '').replace('\t', '').replace('\r', '')


remove_punctuation_and_stem(content_list)

# ======================================================================================================================

# ====================== Convert the content list into TF-IDF matrix and build data frame ==============================

count_vect = CountVectorizer(stop_words='english')
X_counts = count_vect.fit_transform(content_list)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)


# ======================================================================================================================
# Save the sparse tf-idf matrix to a separate file and read the file next time to fasten the process.

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


# Helper function to print out the classification report.
def report_classification(truth, prediction, classifier):
    print '\n              Classification report:'
    print '======================================================='
    print 'Classifier: ' + classifier

    print '======================================================='
    print metrics.classification_report(truth, prediction, target_names=["Washington", "Massachusetts"]),
    print '=======================================================\n'

    print 'Confusion Matrix:'
    print '=============='
    print metrics.confusion_matrix(truth, prediction)
    print '==============\n'

    print 'Total accuracy: '
    print np.mean(prediction == truth)
    print '\n\n\n\n'

    return np.mean(prediction == truth)


def plot_roc(truth, score):
    fpr, tpr, threshold = roc_curve(truth, score, pos_label=1)
    line = [0, 1]
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.title('ROC-Curve of Logistic Regression Classification', fontsize=20)
    plt.axis([-0.004, 1, 0, 1.006])
    plt.show()


# ======================================================================================================================

save_sparse_csr('x-tfidf', X_tfidf)

X_tfidf = load_sparse_csr('x-tfidf.npz')
csv_in = open('true_labels.csv', 'r')
true_labels = list(csv.reader(csv_in, delimiter=','))[0]
true_labels = map(int, true_labels)
accuracy = []
sample = random.sample(range(0, 54916), 10000)
X_sample = X_tfidf[sample]
label_sample = [true_labels[i] for i in sample]
# for i, n in zip(range(0, 8), [3, 5, 10, 20, 50, 70, 100, 200]):
svd = TruncatedSVD(n_components=3, n_iter=10, random_state=42, algorithm='arpack')
svd.fit(X_tfidf)
LSI = svd.transform(X_tfidf)

test_set = random.sample(range(0, 54916), 3000)
train_set = [i for i in range(0, 54916) if i not in set(test_set)]
LSI_test = LSI[test_set]
true_label_test = [true_labels[i] for i in test_set]

LSI_train = LSI[train_set]
true_label_train = [true_labels[i] for i in train_set]


# ======================================================================================================================

# ================================= Classification with SVM (5-cross-validation) =======================================

classifier = svm.SVC(gamma=6)
classifier.fit(LSI, true_labels)
predicted_labels = cross_val_predict(classifier, LSI, true_labels, n_jobs=-1, cv=5)
report_classification(true_labels, predicted_labels, 'SVM(Cross Validation)')

# Try the random-generated test/training set.
classifier.fit(LSI_train, true_label_train)
predicted_labels = classifier.predict(LSI_test)
scores = classifier.decision_function(LSI_test)

report_classification(true_label_test, predicted_labels, 'SVM(test set)')
plot_roc(true_label_test, scores)


# ======================================================================================================================

# ============================================== Logistic Regression ===================================================

classifier = LogisticRegression()
classifier.fit(LSI_train, true_label_train)
predicted_labels = classifier.predict(LSI_test)
predict_probability = classifier.predict_proba(LSI_test[:])[:, 1]

accuracy.append(report_classification(true_label_test, predicted_labels, 'Logistic Regression'))
plot_roc(true_label_test, predict_probability)

plt.figure()
y_labels = ['0', '20%', '40%', '60%', '80%', '100%']
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], y_labels, fontsize=13)
plt.ylabel('Total Accuracy of Classification', fontsize=20)
plt.xlabel('Truncated dimension', fontsize=20)
plt.plot([10, 20, 50, 70, 100, 200, 350, 500], accuracy, 'g')
plt.title('Classification accuracy VS # of truncated terms')
plt.show()

# ======================================================================================================================

# =============================== Logistic Regression (Regularization and penalty) =====================================


def classify(regularization, penalization):
    classifier = LogisticRegression(C=regularization, penalty=penalization)
    classifier.fit(LSI_train, true_label_train)

    predicted_class = classifier.predict(LSI_test)
    actual_class = true_label_test
    predict_probability = classifier.predict_proba(LSI_test[:])[:, 1]

    # Only print details of these regularization terms
    if 0.0001 <= regularization <= 1:
        line1 = 'REGULARIZATION TERM: ' + str(regularization)
        line2 = 'PENALTY TYPE: ' + penalization + ' norm regularization'
        spaces = ''
        for i in range(36 - len(line1)):
            spaces += ' '
        print '########################################'
        print '#                                      #'
        print '# ' + line2 + ' #'
        print '#                                      #'
        print '# ' + line1 + spaces + ' #'
        print '########################################\n'

        report_classification(actual_class, predicted_class, 'LR with Regularization & Penalty')
        plot_roc(actual_class, predict_probability)

    # Cross validation
    log = LogisticRegression(C=regularization, penalty=penalization)
    predicted_labels = cross_val_predict(log, LSI, true_labels, n_jobs=-1, cv=5)
    log.fit(LSI, true_labels)
    predict_probability = log.predict_proba(LSI[:])[:, 1]

    report_classification(true_labels, predicted_labels, 'LR(l1 penalty) - Cross Validation')
    plot_roc(true_labels, predict_probability)

    return np.mean(actual_class == predicted_class)

# classify(1, 'l1')


accuracy_l1 = []
accuracy_l2 = []

for i in range(-6, 3):
    accuracy_l1.append(classify(pow(10, i), 'l1'))
    accuracy_l2.append(classify(pow(10, i), 'l2'))

plt.clf()
plt.figure()
x_labels = ['0.000001', '0.00001', '0.0001', '0.001', '0.01', '0.1', '1', '10', '100']
plt.xticks(range(-6, 3), x_labels, fontsize=13, rotation=15)
y_labels = ['0', '20%', '40%', '60%', '80%', '100%']
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], y_labels, fontsize=13)
plt.plot(range(-6, 3), accuracy_l1, 's', label='l1 Norm Regularization', c='b')
plt.plot(range(-6, 3), accuracy_l1, c='b')
plt.plot(range(-6, 3), accuracy_l2, 'D', label='l2 Norm Regularization', c='g')
plt.plot(range(-6, 3), accuracy_l2, c='g')
plt.ylabel('Total Accuracy of Classification', fontsize=20)
plt.xlabel('Regularization Term', fontsize=20)
plt.title('Accuracy   vs.   Regularization Term', fontsize=20)
plt.axis([-7, 4, 0, 1])
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(0.94, 0.8), fontsize=15, numpoints=1)
plt.show()
