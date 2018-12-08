## helper function to extract biggram phrase with TF-IDF scheme

## import dependecies
from sklearn.pipeline import Pipeline
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import spacy, en_core_web_sm

nlp=en_core_web_sm.load()

##
def preprocessing_txt(dataset):
    stop_words = set(stopwords.words('english'))
    corpus=[]
    for elm in range(0, len(dataset.index)):
        res=' '.join([i for i in dataset['Statement'][elm].lower().split() if i not in stop_words])
        res=re.sub("</?.*?>"," <> ",dataset['Statement'][elm])    # remove tags
        res=re.sub("(\\d|\\W)+"," ",dataset['Statement'][elm])        # remove special characte
        res=re.sub(r'@([A-Za-z0-9_]+)', "",dataset['Statement'][elm])  # remove twitter handler
        res=re.sub('(\r)+', "", dataset['Statement'][elm])            # remove newline character
        res=re.sub('[^\x00-\x7F]+', "", dataset['Statement'][elm])    # remove non-ascii characters
        res=''.join(x for x in dataset['Statement'][elm] if x not in set(string.punctuation))   ## remove punctuation
        corpus.append(res)
    return corpus

##
corpus=preprocessing_txt(dataset)

def bigphrase_tfidf_feats(corpus):
    lemmetized_sent=[]
    for each_sent in nlp.pipe(corpus, batch_size=50, n_threads=-1):
        if each_sent.is_parsed:
            res=[tok.lemma_ for tok in each_sent if not tok.is_punct or tok.is_space or tok.is_stop or tok.like_num]
            lemmetized_sent.append(res)
        else:
            lemmetized_sent.append(None)
    bigram=Phraser(Phrases(lemmetized_sent))
    bigram_lem=list(bigram[lemmetized_sent])
    parsed=[]
    for k in range(0, len(bigram_lem)):
        joined=' '.join(bigram_lem[k])
        parsed.append(joined)
    return parsed