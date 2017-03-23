
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
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
def classify(regularization, penalization):
    classifier = LogisticRegression(C= regularization, penalty = penalization)
    classifier.fit(LSI_train, classification_train)
    
    predicted_class = classifier.predict(LSI_test)
    actual_class = classification_test
    predict_probability = classifier.predict_proba(LSI_test[:])[:,1]
    
    
    # Only print details of these regularization terms
    if regularization >= 0.0001 and regularization <= 1:
        line1 = 'REGULARIZATION TERM: ' + str(regularization)
        line2 = 'PENALTY TYPE: ' + penalization +' norm regularization'
        spaces = ''
        for i in range(36-len(line1)):
            spaces += ' '
        print '########################################'
        print '#                                      #'
        print '# ' + line2 + ' #'
        print '#                                      #'
        print '# ' + line1 + spaces + ' #'
        print '########################################\n'
    
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
        plt.figure(figsize=(10,10))
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1])
        plt.ylabel('True Positive Rate', fontsize = 20)
        plt.xlabel('False Positive Rate', fontsize = 20)
        plt.title('ROC-Curve of Logistic Regression Classification', fontsize = 20)
        plt.axis([-0.004, 1, 0, 1.006])
        plt.show()
        
        print '\n\n\n\n'
    return np.mean(actual_class == predicted_class)
#==============================================================================


accuracy_l1 = []
accuracy_l2 = []

for i in range(-6, 7):
    accuracy_l1.append(classify(pow(10,i), 'l1'))
    accuracy_l2.append(classify(pow(10,i), 'l2'))
    
    
    
plt.clf()
plt.figure(figsize = (12,6))
x_labels = ['0.000001', '0.00001', '0.0001', '0.001', '0.01', '0.1', '1', '10', '100', '1000', '10000', '100000', '1000000']
plt.xticks(range(-6, 7), x_labels, fontsize = 13, rotation = 15)
y_labels = ['0', '20%', '40%', '60%', '80%', '100%']
plt.yticks([0,0.2,0.4,0.6,0.8,1], y_labels, fontsize = 13)
plt.plot(range(-6, 7), accuracy_l1, 's', label = 'l1 Norm Regularization', c = 'b')
plt.plot(range(-6, 7), accuracy_l1, c = 'b')
plt.plot(range(-6, 7), accuracy_l2, 'D', label = 'l2 Norm Regularization', c = 'g')
plt.plot(range(-6, 7), accuracy_l2, c = 'g')
plt.ylabel('Total Accuracy of Classification', fontsize = 20)
plt.xlabel('Regularization Term', fontsize = 20)
plt.title('Accuracy   vs.   Regularization Term', fontsize = 20)
plt.axis([-7,7,0,1])
plt.grid(True)
plt.legend(loc = 'upper right', bbox_to_anchor = (0.94, 0.8), fontsize=15,numpoints = 1)
plt.show()
