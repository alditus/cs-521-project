## full implementation of TF-IDF
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import string, re

def preprocessing_txt(dataset):
    stop_words = set(stopwords.words('english'))
    rm_punct = re.compile('[{}]'.format(re.escape(string.punctuation)))
    corpus=[]
    for elm in range(0, len(dataset.index)):
        res=' '.join([i for i in dataset['Statement'][elm].lower().split() if i not in stop_words])
        res=rm_punct.sub(' ', res)
        corpus.append(res)
    return corpus

def sort_coo(coo_matrix):
    tuples=zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_top_words(feature_names, sorted_items, topn=3):
    sorted_items = sorted_items[:topn]
    score_vals,feature_vals,results = [],[],{}
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results
##
def get_keywords_tfidf(dataset):
    keywords=[]
    cv=CountVectorizer(max=0.85, stop_words=None)
    tfidf_trans=TfidfTransformer(smooth_idf=True, use_idf=True)
    txt_corpus = preprocessing_txt(dataset)
    word_count_vector=cv.fit_transform(txt_corpus)
    tfidf_trans.fit(word_count_vector)
    feature_name = cv.get_feature_names()
    for i in range(0, len(txt_corpus)):
        tfidf_vector = tfidf_trans.transform(cv.transform([txt_corpus[i]]))
        sorted_vector=sort_coo(tfidf_vector.tocoo())
        res=extract_top_words(feature_name, sorted_vector,3)
        keywords.append(res)
    return keywords