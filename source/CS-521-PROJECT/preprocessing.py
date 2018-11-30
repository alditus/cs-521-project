import nlp_util
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
# return POS grouped by unigrams, bigrams, trigrams using a dictionary

## read dataset
df=pd.read_csv('train.tsv',delimiter='\t',encoding='utf-8')
df.columns=['ID','Label','Statement','Subject','speaker','job_title',
           'state_info','pantry_affiliation','barely_true_cnt','false_cnt',
           'half_true_cnt','mostly_true_cnt','pants_on_fire_cnt','Context']

#---------------------------------------------------#
## data cleaning phase

stop=set(stopwords.words('english'))
text_no_stops=[]
for elm in range(0, len(dataset.index)):
    res=' '.join([i for i in dataset['Statement'][elm].lower().split() if i not in stop])
    text_no_stops.append(res)

## remove the punctuation
def rm_punct(sentence):
    flushed_punct = set(string.punctuation)
    res = ''.join(x for x in sentence if x not in flushed_punct)
    return res

text_no_punct=[]
for i in range(0, len(dataset.index)):
    res=rm_punct(dataset['Statement'][i])
    text_no_punct.append(res)

## here is how to clean up non-letter symbols in statement columns
all_text = []
for i in range(0, len(dataset.index)):
    patt = re.sub('[^a-zA-Z]', ' ', dataset['Statement'][i])
    res = ' '.join(str(patt).lower().split())
    all_text.append(res)

#-----------------------------------------#

def extract_POS(statements):
	corenlp = nlp_util.NLP_Task()
	print('Extracting POS Tags')
	pos_tags = corenlp.POS_tagging(statements,return_word_tag_pairs=False)
	bigrams_pos = corenlp.POS_groupping(pos_tags, grams=2)
	trigrams_pos = corenlp.POS_groupping(pos_tags, grams=3)
	#For experimenting
	print("Stringify")
	pos_tags = [" ".join(x) for x in pos_tags]
	bigrams_pos = [" ".join(x) for x in bigrams_pos]
	trigrams_pos = [" ".join(x) for x in trigrams_pos]
	print('Finished')
	return pos_tags,bigrams_pos,trigrams_pos

# return labels for multiclass classification
# label_values dictionary representing the different values for every label in labels
def create_labels(labels, label_values):
	n_labels = list()
	for label in labels:
		n_labels.append(label_values[label])
	return n_labels

# return number of word by sentence
def word_counts(statements):
	corenlp = nlp_util.NLP_Task()
	#getting tokens by sentences
	print('Extracting tokens by sentences')
	tbs = corenlp.TokensBySentence(statements) # tokens by sentence
	print('Counting tokens')
	wc = [len(x) for x in tbs]
	print('Finished')
	return wc

# return sentences vectors for pos unigrams
def pos_vectors(vector_dictionary, pos_tags):
	# One hot version of POS tags
	occurrence_vector = np.zeros((len(pos_tags),len(vector_dictionary)))
	# Frequency vector
	frequency_vector = np.zeros((len(pos_tags),len(vector_dictionary)))
	print('Processing POS tags and creating vectors')
	for index, pos_t in enumerate(pos_tags):
		for each_pos in pos_t:
			if each_pos in vector_dictionary:
				# get the index of the tags vector
				v_index = [i for i,x in enumerate(vector_dictionary) if x == each_pos][0]
				occurrence_vector[index][v_index] = 1
				frequency_vector[index][v_index] +=1
	print('Finished')
	return occurrence_vector, frequency_vector



