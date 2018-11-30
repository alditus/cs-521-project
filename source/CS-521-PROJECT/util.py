import pandas as pd
import numpy as np

"""
	Load tsv files.
	return: train, validation and testing files
"""
def load_files(path='..\\dataset\\{0}', file_names=['train.tsv','valid.tsv','test.tsv']):
	# define columns names
	column_names = ['Id', 'Label','Statement','Subject','Speaker','Speaker Job','State Info','Party','BT','FC','HT','MT','PF','Context']

	# load each file
	train_file = pd.read_csv(path.format(file_names[0]), sep='\t', header=None, encoding='utf-8')
	validation_file = pd.read_csv(path.format(file_names[1]), sep='\t',header=None, encoding='utf-8')
	testing_file = pd.read_csv(path.format(file_names[2]), sep='\t',header=None, encoding='utf-8')

	# set columns names
	train_file.columns = column_names
	validation_file.columns = column_names
	testing_file.columns = column_names

	return train_file, validation_file, testing_file

def load_file(path, file_name):
	# define columns names
	#column_names = ['Id', 'Label','Statement','Subject','Speaker','Speaker Job','State Info','Party','BT','FC','HT','MT','PF','Context']

	# load each file
	uploading_file = pd.read_csv(path.format(file_name), encoding='utf-8')
	#validation_file = pd.read_csv(path.format(file_names[1]), sep='\t',header=None, encoding='utf-8')
	#testing_file = pd.read_csv(path.format(file_names[2]), sep='\t',header=None, encoding='utf-8')

	# set columns names
	#train_file.columns = column_names
	#validation_file.columns = column_names
	#testing_file.columns = column_names

	return uploading_file


"""
	Convert tsv file columns into a dictionary
	return: dictionary with specified columns
"""
def tsv_to_dict(tsv_file,columns=None):
	# columns to return 
	if columns==None:
		columns = tsv_file.columns

	# dictionary to return
	data_dict = dict()

	for each_column in columns:
		d_key = each_column.lower().strip().replace(' ','-')
		data_dict[d_key] = tsv_file[[each_column]].values[:,0] 

	return data_dict

"""
	return the most frequent element in a sequence
"""
def mode(seq):
  if len(seq) == 0:
    return 1.
  else:
    cnt = {}
    for item in seq:
      if item in cnt:
        cnt[item] += 1
      else:
        cnt[item] = 1
    maxItem = seq[0]
    for item,c in cnt.items():
      if c > cnt[maxItem]:
        maxItem = item
    return maxItem
