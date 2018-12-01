## helper function to extract key target phrases in fake news statement
from __future__ import unicode_literals
import spacy, en_core_web_sm, textacy
import pandas as pd
import numpy as np
nlp=en_core_web_sm.load()

## read dataset
df=pd.read_csv('train.tsv',delimiter='\t',encoding='utf-8')
df.columns=['ID','Label','Statement','Subject','speaker','job_title',
           'state_info','pantry_affiliation','barely_true_cnt','false_cnt',
           'half_true_cnt','mostly_true_cnt','pants_on_fire_cnt','Context']

## remove stop words
stop=stopwords.words('english')
df['statement']=df['statement'].apply(lambda x: "".join([word for word in x.split() if word not in (stop)]))

## keep major columns that we are interested in
dataset = df.filter(items=['ID','Statement', 'Subject', 'Label','Context'])
dataset.head()

### define key value pair
data_dict=dict()
data_dict['ids'] = train_data[['ID']].values[:,0]
data_dict['labels'] = train_data[['Label']].values[:,0]
data_dict['statements'] = train_data[['Statement']].values[:,0]
data_dict['subjects'] = train_data[['Subject']].values[:,0]
data_dict['contexts'] = train_data[['Context']].values[:,0]

## extract multiple key phrases from fake news sentence

## get_diff_chunking function to extract multiple phrase except noun phrase
def get_diff_chunking(sent):
    nlp=en_core_web_sm.load()
    parsed=nlp(sent)
    pps, adjs=[],[]
    for token in doc:
        if token.pos_=='ADP': ## find out propositional phrase
            pp=' '.join([tok.orth_ for tok in token.subtree])
            pps.append(pp)
        if token.pos_=='ADJ': ## find out adjective phrase
            adj=' '.join([tok.orth_ for tok in token.subtree])
            adjs.append(adj)
    return pps, adjs

## get_noun_adj helper function to extract noun_adjective pairs from sentence
import spacy
import en_core_web_sm
nlp=en_core_web_sm.load()

def get_noun_adj_pairs(sent):
    parsed=nlp(sent)
    noun_adj_pairs_res=[]
    for i, tok in enumerate(parsed):
        if tok.pos_ not in ('NOUN','PRON'):
            continue
        for j in range(i+1, len(parsed)):
            if parsed[j].pos_=='ADJ':
                noun_adj_pairs_res.append(tok, parsed[j])
                break
    return noun_adj_pairs_res

## This helper function can extract different type of phrases from fake news sentence
def get_phrases(sent, which_phrase=('verb','propositional','noun')):
    parsed=nlp(sent)
    res=[]
    if which_phrase=='verb':
        verb_patt=r'<VERB>?<ADV>*<VERB>+'
        matched=textacy.extract.pos_regex_matches(parsed, verb_patt)
        verbs=[vb.text for vb in matched]
        res.append(verbs)
    if which_phrase=='propositional':
        prop_patt=r'<PREP> <DET>? (<NOUN>+<ADP>)* <NOUN>+'
        matched=textacy.extract.pos_regex_matches(parsed, prop_patt)
        props=[pp.text for pp in matched]
        res.append(props)
    if which_phrase=='noun':
        nn_patt=r'<DET>?(<NOUN>+ <ADP|CONJ>)*<NOUN>+'
        matched=textacy.extract.pos_regex_matches(parsed,  nn_patt)
        nnp=[nn.text for nn in matched]
        res.append(nnp)
    return res


####
import spacy, en_core_web_sm
nlp=en_core_web_sm.load()

def get_compound_nn_adj(doc):
    compounds_nn_pairs = []
    parsed=nlp(doc)
    compounds = [token for token in sent if token.dep_ == 'compound']
    compounds = [nc for nc in compounds if nc.i == 0 or sent[nc.i - 1].dep_ != 'compound']
    if compounds:
        for token in compounds:
            pair_1, pair_2 = (False, False)
            noun = sent[token.i:token.head.i + 1]
            pair_1 = noun
            if noun.root.dep_ == 'nsubj':
                adj_list = [rt for rt in noun.root.head.rights if rt.pos_ == 'ADJ']
                if adj_list:
                    pair_2 = adj_list[0]
            if noun.root.dep_ == 'dobj':
                verb_root = [vb for vb in noun.root.ancestors if vb.pos_ == 'VERB']
                if verb_root:
                    pair_2 = verb_root[0]
            if pair_1 and pair_2:
                compounds_nn_pairs.append(pair_1, pair_2)
    return compounds_nn_pairs