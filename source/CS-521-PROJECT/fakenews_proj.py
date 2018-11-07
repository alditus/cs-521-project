## This project is dedicated to course project of statistical Natural language processing
## Title: fake news project
## Project Member: Jurat, Aldo

import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
import nltk
import nltk.corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.word2vec import Word2Vec
from spacy import load
import en_core_web_sm
from spacy import displacy
from string import punctuation
##
nlp=load("en")
tokenier=Tokenizer(nlp.vocab)


## data cleaning and pre-processing
def read_file(txt):
    with open(txt, 'r') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for line in rd:
            print(line)

df=read_file('train.tsv')
df=pd.DataFrame(df)
df.columns=['statement_ID','label','statement','subject','speaker','job_title',
           'state_info','pantry_affiliation','tot_credit_hist_cnt','barely_true_cnt','false_cnt',
           'half_true_cnt','mostly_true_cnt','pants_on_fire_cnt']

df.head()
df.tail()
df.shape[0]
list(df.columns.values)
df.index

## remove all stopwords
stop=stopwords.words('english')
df['statement']=df['statement'].apply(lambda x: "".join([word for word in x.split() if word not in (stop)]))

## helper function to get bag of words
def tokenize_sent(sentence):
    token=word_tokenize(sentence)
    return(token)

def sentence_tokens(df):
    bag_of_words = []
    for line in range(len(df.index)):
        sentence = re.sub(ur"[0-9]+|\p{P}+", u"", line)
        sentence = tokenize_sent(df['statement'][i])
        bag_of_words.append(sentence)
    return bag_of_words

## helper function to remove stop words
from string import punctuation

def rm_punct(sentence):
    flushed_punct=set(string.punctuation)
    res=''.join(x for x in sentence if x not in flushed_punct)
    return res

## helper function to lemmatize the token
def lemmetize_token(sentence):
    lemmati=WordNetLemmatizer()
    lemmatized = [[lemmati.lemmatize(word) for word in word_tokenize(s)] for s in sentence]
    return(list(lemmatized))

## helper function to reomve white space in sentence with lower case character
def sent_lower_case(df):
    sent_arr=[]
    for i in range(0, len(df.index)):
        patt=re.sub(r'[^a-zA-Z]','', df['statement'][i])
        sent='.'.join(x.lower() for x in patt.split('.'))
        sent_arr.append(sent)
    return sent_arr

## helper function for depndencing parsing on fake news statement
def dep_par_sents(df):
    nlp=en_core_web_sm.load()
    corpus=sent_lower_case(df)
    dep_pars_res=[]
    for line in range(0, len(df.index)):
        docs=nlp(unicode(corpus[line],"utf-8"))
        dep_pars_res.append(docs)

# to visualize parsing tree
docs=[]
for each in range(0, len(df.index)):
    displacy.serve(df[each], style='dep')


## helper function to filter out tags
## where we want to filter the word with tag- noun,adjective,verb,adverb

def filterTag(tagged_fakenews_statement):
    final_text_list=[]
    for text_list in tagged_fakenews_statement:
        final_text=[]
        for word,tag in text_list:
            if tag in ['NN','NNS','NNP','NNPS','RB','RBR','RBS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ']:
                final_text.append(word)
        final_text_list.append(' '.join(final_text))
    return final_text_list