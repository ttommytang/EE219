from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from math import log
import matplotlib.pyplot as plt
import re

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
#labels = dataset.target
# labels = [1] * len(dataset.data)
# for i in range(len(dataset.data)):
#     if dataset.target[i] > 3:
#         labels[i] = 0
labels = [1] * len(dataset.data)
for i in range(len(dataset.data)):
    if dataset.target[i] <= 4:
        labels[i] = 0
    elif dataset.target[i] <= 8:
        labels[i] = 1
    elif dataset.target[i] <= 12:
        labels[i] = 2
    elif dataset.target[i] <= 13:
        labels[i] = 3
    elif dataset.target[i] <= 16:
        labels[i] = 4
    elif dataset.target[i] <= 19:
        labels[i] = 5
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

# =======================Calculate top singular values==========================
# print 'Calculating singular values...'
# num_of_singular_values = 1000
# u, singular_values, vt = svds(X.toarray(), num_of_singular_values)
# singular_values = singular_values[::-1]
# print 'Top', num_of_singular_values, 'singular values are:'
# print singular_values
#
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 1001), singular_values)
# plt.ylabel('Singular Value', fontsize=20)
# plt.xlabel('Index', fontsize=20)
# plt.title('Top 1000 singular values', fontsize=20)
# plt.axis([-1, 1001, 0, 14])
# plt.show()


# ==============================================================================

def k_means(X_reduced, labels, dim_reduce):
    # =============================K-Means Clustering===============================
    km = KMeans(n_clusters=6, init='k-means++', max_iter=100, n_init=1) #### n_cluster=6
    km.fit(X_reduced)
    # ==============================================================================

    # ================================Print results=================================

    homo = metrics.homogeneity_score(labels, km.labels_)
    complete = metrics.completeness_score(labels, km.labels_)
    rand = metrics.adjusted_rand_score(labels, km.labels_)
    mutual = metrics.adjusted_mutual_info_score(labels, km.labels_)

    print 'Dimension reduction method:', dim_reduce
    # print 'Homogeneity:', metrics.homogeneity_score(labels, km.labels_)
    # print 'Completeness:', metrics.completeness_score(labels, km.labels_)
    # print 'Adjusted Rand Score:', metrics.adjusted_rand_score(labels, km.labels_)
    # print 'Adjusted Mutual Info Score:', metrics.adjusted_mutual_info_score(labels, km.labels_)
    # print '\nConfusion matrix:'
    # print '=============='
    # print metrics.confusion_matrix(labels, km.labels_)
    # print '=============='
    # print '-----------------------------------------------------'

    return homo, complete, rand, mutual
    # ==============================================================================


# =========================Reduce Dimensionality (SVD)==========================
dimension_array = [213, 193, 180, 166, 133, 113, 93, 73, 57, 33, 24, 13, 9, 8, 7, 6, 5, 4, 3, 2]
# dimension_array = range(165, 180)
print '##############################################################'
homos = []
cmplts = []
rands = []
mutuals = []

for d in dimension_array:
    print 'Performing truncatedSVD...reduce dimension to ' + str(d)
    svd = TruncatedSVD(n_components=d, n_iter=13, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X_reduced = lsa.fit_transform(X)

    homo, cmplt, rand, mutual = k_means(X_reduced, labels, 'truncatedSVD')

    homos.append(homo)
    cmplts.append(cmplt)
    rands.append(rand)
    mutuals.append(mutual)

plt.plot(dimension_array, homos, color='r', label='homogeneity_score')
plt.plot(dimension_array, cmplts, color='g', label='completeness_score')
plt.plot(dimension_array, rands, color='y', label='adjusted_rand_score')
plt.plot(dimension_array, mutuals, color='b', label='adjusted_mutual_info_score')
plt.legend()
plt.xlabel('Dimension')
plt.ylabel('Scores')
plt.title('<TruncatedSVD> Clustering performance vs Dimension')
plt.show()

# ==============================================================================

# =========================Reduce Dimensionality (PCA)==========================
# dimension_array = range(57, 61)
print '##############################################################'
homos = []
cmplts = []
rands = []
mutuals = []
for d in dimension_array:
    print 'Performing PCA...reduce dimension to ' + str(d)

    pca = PCA(n_components=d, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(pca, normalizer)

    X_reduced = lsa.fit_transform(X.toarray())

    homo, cmplt, rand, mutual = k_means(X_reduced, labels, 'PCA')

    homos.append(homo)
    cmplts.append(cmplt)
    rands.append(rand)
    mutuals.append(mutual)

plt.plot(dimension_array, homos, color='r', label='homogeneity_score')
plt.plot(dimension_array, cmplts, color='g', label='completeness_score')
plt.plot(dimension_array, rands, color='y', label='adjusted_rand_score')
plt.plot(dimension_array, mutuals, color='b', label='adjusted_mutual_info_score')
plt.legend()
plt.xlabel('Dimension')
plt.ylabel('Scores')
plt.title('<PCA> Clustering performance vs Dimension')
plt.show()
# ==============================================================================


# =========================Reduce Dimensionality (NMF)==========================
print '##############################################################'
print 'Point distribution:'
dimension_array = [73, 57, 33, 24, 13, 9, 8, 7, 6, 5, 4, 3, 2]
# dimension_array = range(13, 33)
nmf = NMF(n_components=2, random_state=42)
lsa = make_pipeline(nmf)
X_reduced = lsa.fit_transform(X)

plt.clf()
plt.figure(figsize=(10, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=3, c='r', alpha=0.8)
plt.title('Point distribution after NMF', fontsize=20)
plt.axis([0, 0.16, 0, 0.2])
plt.show()

homos = []
cmplts = []
rands = []
mutuals = []

homosLog = []
cmpltsLog = []
randsLog = []
mutualsLog = []

for d in dimension_array:
    print 'Performing NMF with/without logarithm transformation...reduce dimension to ' + str(d)
    nmf = NMF(n_components=d, random_state=42)
    lsa = make_pipeline(nmf)

    X_reduced = lsa.fit_transform(X)

    homo, cmplt, rand, mutual = k_means(X_reduced, labels, 'NMF')

    homos.append(homo)
    cmplts.append(cmplt)
    rands.append(rand)
    mutuals.append(mutual)

    # ----------------------------Non-linear Transformation-------------------------
    for j in range(X_reduced.shape[0]):
        for k in range(X_reduced.shape[1]):
            if X_reduced[j][k] == 0:
                X_reduced[j][k] = -3.08
            else:
                X_reduced[j][k] = log(X_reduced[j][k], 10)
    # ------------------------------------------------------------------------------

    homo, cmplt, rand, mutual = k_means(X_reduced, labels, 'NMF_LOG')

    homosLog.append(homo)
    cmpltsLog.append(cmplt)
    randsLog.append(rand)
    mutualsLog.append(mutual)

plt.plot(dimension_array, homos, color='r', label='homogeneity_score')
plt.plot(dimension_array, cmplts, color='g', label='completeness_score')
plt.plot(dimension_array, rands, color='y', label='adjusted_rand_score')
plt.plot(dimension_array, mutuals, color='b', label='adjusted_mutual_info_score')
plt.plot(dimension_array, homosLog, 'r--', label='homogeneity_score LOG')
plt.plot(dimension_array, cmpltsLog, 'g--', label='completeness_score LOG')
plt.plot(dimension_array, randsLog, 'y--', label='adjusted_rand_score LOG')
plt.plot(dimension_array, mutualsLog, 'b--', label='adjusted_mutual_info_score LOG')
plt.legend()
plt.xlabel('Dimension')
plt.ylabel('Scores')
plt.title('<NMF> Clustering performance vs Dimension')
plt.show()
# ==============================================================================
