
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import re

#=================================Obtain data==================================
comp_tech_subclasses = ['comp.graphics', 
                        'comp.os.ms-windows.misc', 
                        'comp.sys.ibm.pc.hardware', 
                        'comp.sys.mac.hardware']
                        
rec_act_subclasses = ['rec.autos', 
                      'rec.motorcycles', 
                      'rec.sport.baseball', 
                      'rec.sport.hockey']
  
dataset = fetch_20newsgroups(subset='all',
                             categories=comp_tech_subclasses+rec_act_subclasses,
                             shuffle=True,
                             random_state=42,
                             remove=('headers', 'footers', 'quotes'))


labels = [1]*len(dataset.data)
for i in range(len(dataset.data)):
    if dataset.target[i] > 3:
        labels[i] = 0
#==============================================================================





#===================Remove Punctuation & Stem & Stop Words=====================
punctuations = '[! \" # $ % \& \' \( \) \* + , \- \. \/ : ; < = > ? @ \[ \\ \] ^ _ ` { \| } ~]'
stemmer = SnowballStemmer("english")
def remove_punctuation_and_stem(data_list):
    for i in range(len(data_list)):
        data_list[i] = " ".join([stemmer.stem(data) for data in re.split(punctuations, data_list[i])])
        data_list[i] = data_list[i].replace('\n','').replace('\t','').replace('\r','')
remove_punctuation_and_stem(dataset.data)
#==============================================================================





#=======================Transform the data into TF-IDF=========================
vectorizer = TfidfVectorizer(max_features=10000,
                             min_df=10,
                             stop_words='english',
                             use_idf=True)
X = vectorizer.fit_transform(dataset.data)
#==============================================================================





#=============================K-Means Clustering===============================
km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
km.fit(X)
#==============================================================================






#================================Print results=================================
print 'Homogeneity:', metrics.homogeneity_score(labels, km.labels_)
print 'Completeness:', metrics.completeness_score(labels, km.labels_)
print 'Adjusted Rand Score:', metrics.adjusted_rand_score(labels, km.labels_)
print 'Adjusted Mutual Info Score:', metrics.adjusted_mutual_info_score(labels, km.labels_)
print '\nConfusion matrix:'
print '=============='
print metrics.confusion_matrix(labels, km.labels_)
print '=============='
#==============================================================================





