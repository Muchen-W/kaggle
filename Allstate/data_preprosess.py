import pandas as pd
import numpy as np
from sklearn import preprocessing 

def data_preprocess(fread, fwrite):
	df_tr = pd.read_csv(fread)
	for key in df_tr.keys():
		if 'cat' in key:
			for item in df_tr[key].unique():
				df_tr[key+'_'+item] = df_tr[key].apply(lambda x: 1 if x == item else 0)
			df_tr.drop(key, axis=1, inplace=True)
	
	#key_drop = ['cat'+str(i+1) for i in range(116)]
	#df_tr.drop(key_drop, axis=1, inplace=True)
	df_tr.to_csv(fwrite, index=False)
	return df_tr

def sample_trial():
	fread = 'train_sample.csv'
	fwrite = 'train_sample_preprocessed.csv'
	df_tr = data_preprocess(fread, fwrite)
	print(df_tr.describe())

def complete_train_data():
	fread = 'train.csv'
	fwrite = 'train_preprocessed.csv'
	df_tr = data_preprocess(fread, fwrite)

def split_train_validate_test(fread, fwrite):
	df = pd.read_csv(fread)
	df_train = df.sample(frac=0.6, random_state=1).sort_values('id')
	df.drop(df_train.index, inplace=True)
	df_validate = df.sample(frac=0.5, random_state=2).sort_values('id')
	df_test = df.drop(df_validate.index).sort_values('id')
	for fw in fwrite:
		if 'validate' in fw:
			df_validate.to_csv(fw, index=False)
		elif 'test' in fw:
			df_test.to_csv(fw, index=False)
		else:
			df_train.to_csv(fw, index=False)
	#print(df_sample.describe())

def split_training_dataset():
	#fread = 'train_sample_preprocessed.csv'
	#fwrite = ['train_sample_split.csv', 'validate_sample_split.csv', 'test_sample_split.csv']
	fread = 'train_preprocessed.csv'
	fwrite = ['train_split.csv', 'validate_split.csv', 'test_split.csv']
	split_train_validate_test(fread, fwrite)

def normalize():
	# original data has been normalized
	fread = 'train_sample_preprocessed.csv'
	fwrite = 'train_sample_preprocessed_normalized.csv'
	df = pd.read_csv(fread)
	#min_max_scaler = preprocessing.MinMaxScaler()
	df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
	df.to_csv(fwrite, index=False)

def split_train_for_experiment():
	fread = 'train_split.csv'
	fwrite = 'train_split_sample.csv'
	df_train = pd.read_csv(fread)
	df_sample = df_train.sample(frac=0.02, random_state=3).sort_values('id')
	df_sample.to_csv(fwrite, index=False)

if __name__ == '__main__':
	#sample_trial()
	#complete_train_data()
	#split_training_dataset()
	split_train_for_experiment()