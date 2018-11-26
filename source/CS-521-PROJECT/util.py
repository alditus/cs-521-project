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
	train_file = pd.read_csv(path.format(file_names[0]), sep='\t', header=None)
	validation_file = pd.read_csv(path.format(file_names[1]), sep='\t',header=None)
	testing_file = pd.read_csv(path.format(file_names[2]), sep='\t',header=None)

	# set columns names
	train_file.columns = column_names
	validation_file.columns = column_names
	testing_file.columns = column_names

	return train_file, validation_file, testing_file

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
