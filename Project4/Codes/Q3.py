
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.sparse.linalg import svds
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from math import log
import matplotlib.pyplot as plt
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






#=======================Calculate top singular values==========================
print 'Calculating singular values...'
num_of_singular_values = 1000
u, singular_values, vt = svds(X.toarray(), num_of_singular_values)
singular_values = singular_values[::-1]
print 'Top',num_of_singular_values,'singular values are:'
print singular_values

plt.figure(figsize = (10,6))
plt.plot(range(1,1001), singular_values)
plt.ylabel('Singular Value', fontsize = 20)
plt.xlabel('Index', fontsize = 20)
plt.title('Top 1000 singular values', fontsize = 20)
plt.axis([-1,1001,0,14])
plt.show()
#==============================================================================





def k_means(X_reduced, labels, dim_reduce):
    #=============================K-Means Clustering===============================
    km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
    km.fit(X_reduced)
    #==============================================================================
    
    #================================Print results=================================
    
    print 'Dimension reduction method:', dim_reduce
    print 'Homogeneity:', metrics.homogeneity_score(labels, km.labels_)
    print 'Completeness:', metrics.completeness_score(labels, km.labels_)
    print 'Adjusted Rand Score:', metrics.adjusted_rand_score(labels, km.labels_)
    print 'Adjusted Mutual Info Score:', metrics.adjusted_mutual_info_score(labels, km.labels_)
    print '\nConfusion matrix:'
    print '=============='
    print metrics.confusion_matrix(labels, km.labels_)
    print '=============='
    print '-----------------------------------------------------'
    #==============================================================================






#=========================Reduce Dimensionality (SVD)==========================
print '##############################################################'
for i in range(0,5):
    print 'Performing truncatedSVD...'
    svd = TruncatedSVD(n_components = 165, n_iter = 13,random_state = 42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    
    X_reduced = lsa.fit_transform(X)
    
    k_means(X_reduced, labels, 'truncatedSVD')
#==============================================================================



#=========================Reduce Dimensionality (PCA)==========================
print '##############################################################'
for i in range(0,5):
    print 'Performing PCA...'
    
    pca = PCA(n_components = 300,random_state = 42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(pca, normalizer)
    
    X_reduced = lsa.fit_transform(X.toarray())
    
    k_means(X_reduced, labels, 'PCA')
#==============================================================================





#=========================Reduce Dimensionality (NMF)==========================
print '##############################################################'
print 'Point distribution:'

nmf = NMF(n_components = 2, random_state = 42)
lsa = make_pipeline(nmf)
X_reduced = lsa.fit_transform(X)

plt.clf()
plt.figure(figsize = (10,6))
plt.scatter(X_reduced[:,0], X_reduced[:,1], s=3, c='r', alpha=0.8)
plt.title('Point distribution after NMF', fontsize = 20)
plt.axis([0,0.16,0,0.2])
plt.show()


for i in range(0,5):
    print 'Performing NMF with logarithm transformation...'
    nmf = NMF(n_components = 13, random_state = 42)
    lsa = make_pipeline(nmf)
    
    X_reduced = lsa.fit_transform(X)
    
    #----------------------------Non-linear Transformation-------------------------
    for j in range(X_reduced.shape[0]):
        for k in range(X_reduced.shape[1]):
            if X_reduced[j][k] == 0:
                X_reduced[j][k] = -3.08
            else:
                X_reduced[j][k] = log(X_reduced[j][k], 10)
    #------------------------------------------------------------------------------
      
    k_means(X_reduced, labels, 'NMF')
#==============================================================================
    





