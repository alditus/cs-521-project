import nlp_util
import numpy as np
# return POS grouped by unigrams, bigrams, trigrams using a dictionary
def extract_POS(statements):
	corenlp = nlp_util.NLP_Task()
	print('Extracting POS Tags')
	pos_tags = corenlp.POS_tagging(statements,return_word_tag_pairs=False)
	bigrams_pos = corenlp.POS_groupping(pos_tags, grams=2)
	trigrams_pos = corenlp.POS_groupping(pos_tags, grams=3)
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



