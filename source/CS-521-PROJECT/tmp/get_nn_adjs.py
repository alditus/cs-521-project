## helper function to find closest adjective in the sentence

from nltk.tokenize import MWETokenizer
import nltk

def tokenizer_sent(dataset):
    tokenizer=MWETokenizer()
    aspect_tokenized=[]
    sentence_tokenized=[]
    for i in range(0, len(dataset.index)):
        aspect_split=tuple(dataset['aspect_term'][i].lower().split())
        res=tokenizer.add_mwe(aspect_split)
        aspect_tokenized.append(res)
    for j in range(0, len(dataset.index)):
        tok=nltk.pos_tag(tokenizer.tokenize(dataset['text'][i].lower().split()))
        sentence_tokenized.append(tok)