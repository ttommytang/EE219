from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from math import log
import matplotlib.pyplot as plt
import re
import numpy as np

# =================================Obtain data==================================
comp_tech_subclasses = ['comp.graphics',
                        'comp.os.ms-windows.misc',
                        'comp.sys.ibm.pc.hardware',
                        'comp.sys.mac.hardware',
                        'comp.windows.x']

rec_act_subclasses = ['rec.autos',
                      'rec.motorcycles',
                      'rec.sport.baseball',
                      'rec.sport.hockey']

science_subclass = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']

miscellaneous_subclass = ['misc.forsale']

politics_subclass = ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast']

religion_subclass = ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']

dataset = fetch_20newsgroups(subset='all',
                             categories=comp_tech_subclasses + rec_act_subclasses + science_subclass +
                                        miscellaneous_subclass + politics_subclass + religion_subclass,
                             shuffle=True,
                             random_state=42,
                             remove=('headers', 'footers', 'quotes'))

labels = dataset.target
# labels = [1] * len(dataset.data)
# for i in range(len(dataset.data)):
#     if dataset.target[i] > 3:
#         labels[i] = 0
# ==============================================================================


# ===================Remove Punctuation & Stem & Stop Words=====================
punctuations = '[! \" # $ % \& \' \( \) \* + , \- \. \/ : ; < = > ? @ \[ \\ \] ^ _ ` { \| } ~]'
stemmer = SnowballStemmer("english")


def remove_punctuation_and_stem(data_list):
    for i in range(len(data_list)):
        data_list[i] = " ".join([stemmer.stem(data) for data in re.split(punctuations, data_list[i])])
        data_list[i] = data_list[i].replace('\n', '').replace('\t', '').replace('\r', '')


remove_punctuation_and_stem(dataset.data)
# ==============================================================================


# =======================Transform the data into TF-IDF=========================
vectorizer = TfidfVectorizer(max_features=10000,
                             min_df=10,
                             stop_words='english',
                             use_idf=True)
X = vectorizer.fit_transform(dataset.data)
# ==============================================================================

# ==============================================================================


def plot_clusters(actual_labels, clustered_labels, X_2d, centers, reducer):
    # =================================Plot results=================================
    color = ["grey", "lightcoral", "maroon", "mistyrose", "coral", "peachpuff", "darkorange", "orange", "darkgoldenrod",
             "olive", "yellowgreen", "lawngreen", "lightgreen", "g", "mediumseagreen", "mediumturquoise", "c", "cadetblue",
             "skyblue", "dodgerblue"]
    mark = ["o", "+", "v", "^", "s", "p", "*", "D", "x", "d"]
    for i in range(len(labels)):
        plt.scatter(X_2d[i, 0], X_2d[i, 1], s=12, marker=mark[(actual_labels[i] % 10)], color=color[clustered_labels[i]], alpha=0.5)
    for i in range(20):
        plt.scatter(centers[i, 0], centers[i, 1], marker='^', s=100, color='k', alpha=0.6)
    plt.title('Clustering results with ' + reducer)
    plt.show()

# ==============================================================================

# =========================Reduce Dimensionality (SVD)==========================
print '##############################################################'
print 'Performing truncatedSVD...'
svd = TruncatedSVD(n_components=166, n_iter=13, random_state=42)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X_reduced = lsa.fit_transform(X)

# k_means(X_reduced, labels, 'truncatedSVD')
km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
km.fit(X_reduced)
clustered_labels = km.labels_

svd = TruncatedSVD(n_components=2, n_iter=13, random_state=42)
X_2d = svd.fit_transform(X)
km.fit(X_2d)
centers = km.cluster_centers_

plot_clusters(labels, clustered_labels, X_2d, centers, 'TruncatedSVD')

# ==============================================================================

# =========================Reduce Dimensionality (PCA)==========================
print '##############################################################'
print 'Performing PCA...'

pca = PCA(n_components=57, random_state=42)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(pca, normalizer)

X_reduced = lsa.fit_transform(X.toarray())

# k_means(X_reduced, labels, 'PCA')
km.fit(X_reduced)
clustered_labels = km.labels_

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X.toarray())
km.fit(X_2d)
centers = km.cluster_centers_

plot_clusters(labels, clustered_labels, X_2d, centers, 'PCA')
# ==============================================================================

# =========================Reduce Dimensionality (NMF)==========================
print '##############################################################'
print 'Performing NMF with logarithm...'

nmf = NMF(n_components=2, random_state=42)
lsa = make_pipeline(nmf)
X_reduced = lsa.fit_transform(X)

km.fit(X_reduced)
clustered_labels = km.labels_
centers = km.cluster_centers_

plot_clusters(labels, clustered_labels, X_reduced, centers, 'NMF without Logarithm')

print 'Performing NMF with logarithm transformation...'
nmf = NMF(n_components=24, random_state=42)
lsa = make_pipeline(nmf)
X_reduced = lsa.fit_transform(X)

# ----------------------------Non-linear Transformation-------------------------
for j in range(X_reduced.shape[0]):
    for k in range(X_reduced.shape[1]):
        if X_reduced[j][k] == 0:
            X_reduced[j][k] = -3.08
        else:
            X_reduced[j][k] = log(X_reduced[j][k], 10)
# ------------------------------------------------------------------------------

km.fit(X_reduced)
clustered_labels = km.labels_

nmf = NMF(n_components=2, random_state=42)
X_2d = nmf.fit_transform(X)
km.fit(X_2d)
centers = km.cluster_centers_

plot_clusters(labels, clustered_labels, X_2d, centers, 'NMF with Logarithm')
# ==============================================================================
