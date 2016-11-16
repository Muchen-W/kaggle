import pandas as pd

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

if __name__ == '__main__':
	sample_trial()
	#complete_train_data()