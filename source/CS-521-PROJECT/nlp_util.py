from pycorenlp import StanfordCoreNLP
import numpy as np

class NLP_Task:
	"""
		Initialize StanfordCoreNLP
	"""
	def __init__(self):
			self.core_nlp = StanfordCoreNLP('http://localhost:9000')
			print("NLP_Task ready to use.")

	"""
		return POS tags ngram wise
	"""
	def POS_tagging(self, statements, return_word_tag_pairs = False):
		POS_tags = list()
		for statement in statements:
			statement_tags = list()
			annotations = self.core_nlp.annotate(statement, properties={
			  'annotators': 'tokenize,pos',
			  'outputFormat': 'json'
			  })
			for output in annotations['sentences']:
				statement_tags.append('<s>')
				for token in output['tokens']:
					if return_word_tag_pairs:
						statement_tags.append(token['word']+'/'+token['pos'])
					else:
						statement_tags.append(token['pos'])
			POS_tags.append(statement_tags)
		return POS_tags

	"""
		return POS grouped by number of grams
	"""
	def POS_groupping(self, sentences_pos,grams=1):
		result = list()
		for sentence_tags in sentences_pos:
			tag_group = list()
			for index, each_tag in enumerate(sentence_tags):
				if index < len(sentence_tags)-grams and len(sentence_tags)>=grams:
					format_str = str()
					for i in range(0,grams):
						format_str += sentence_tags[index+i]
						if i<grams-1:
							format_str += ' '
					tag_group.append(format_str)
			result.append(tag_group)
		return result

	"""
		return the list of tokens per statement
	"""
	def TokensBySentence(self, sentences):
		tbs =  list()
		for sentence in sentences:
			token_list = list()
			output = self.core_nlp.annotate(sentence, properties={
			  'annotators': 'tokenize',
			  'outputFormat': 'json'
			  })
			for t in output['tokens']:
				token_list.append(t['word'])
			tbs.append(token_list)
		return tbs