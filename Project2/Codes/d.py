
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import re

computer_technology_subclasses = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
recreational_activity_subclasses = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
  
training_data = fetch_20newsgroups(subset='train', categories=computer_technology_subclasses+recreational_activity_subclasses, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))

stemmer = SnowballStemmer("english")


#===================Remove Punctuation & Stem & Stop Words=====================
punctuations = '[! \" # $ % \& \' \( \) \* + , \- \. \/ : ; < = > ? @ \[ \\ \] ^ _ ` { \| } ~]'
def remove_punctuation_and_stem(data_list):
    for i in range(len(data_list)):
        data_list[i] = " ".join([stemmer.stem(data) for data in re.split(punctuations, data_list[i])])
        data_list[i] = data_list[i].replace('\n','').replace('\t','').replace('\r','')


remove_punctuation_and_stem(training_data.data)


count_vect = CountVectorizer(min_df=10, stop_words ='english')
X_counts = count_vect.fit_transform(training_data.data)

#==============================================================================

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

svd = TruncatedSVD(n_components = 50, n_iter = 10,random_state = 42)
svd.fit(X_tfidf)
LSI = svd.transform(X_tfidf)









