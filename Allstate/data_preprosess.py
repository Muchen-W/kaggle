import pandas as pd

data_tr = pd.read_csv('train.csv')

for key in data_tr.keys():
	if 'cat' in key:
		for item in data_tr[key].unique():
			data_tr[key+'_'+item] = 0
		data_tr.drop(key, axis=1, inplace=True)

