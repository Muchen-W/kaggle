import pandas as pd
import numpy as np
from sklearn import linear_model, kernel_ridge, svm, ensemble, grid_search
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

import xgboost as xgb


def split_train_validate(fread, fwrite):
	df = pd.read_csv(fread)
	df_train = df.sample(frac=0.7, random_state=1).sort_values('id')
	df.drop(df_train.index, inplace=True)
	for fw in fwrite:
		if 'train' in fw:
			df_train.to_csv(fw, index=False)
		else:
			df.to_csv(fw, index=False)
	#print(df_sample.describe())

def split_training_dataset():
	#fread = 'train_sample_preprocessed.csv'
	#fwrite = ['train_sample_split.csv', 'validate_sample_split.csv']
	fread = 'train_preprocessed.csv'
	fwrite = ['train_split.csv', 'validate_split.csv']
	split_train_validate(fread, fwrite)


def data_preprocess(fread, fwrite):
	df = pd.read_csv(fread)
	for key in df.keys():
		if 'cat' in key:
			for item in df[key].unique():
				df[key+'_'+item] = df[key].apply(lambda x: 1 if x == item else 0)
			df.drop(key, axis=1, inplace=True)

	df.to_csv(fwrite, index=False)
	return df


def load_file(fread):
	for fr in fread:
		if 'train' in fr:
			df_train = pd.read_csv(fr)
		else:
			df_validate = pd.read_csv(fr)
	return df_train, df_validate


def data_label_separation(df, y_key='loss', x_drop_key=['id', 'loss']):
	'''
	remove columns of 'id' and 'loss' for X;
	keep 'loss' column as label/ground truth
	'''
	X = df.drop(x_drop_key, axis=1)
	y = df[y_key]
	return X, y


def model_initialization(name, **kwargs):
	if 'gradient_boost_regression' in name:
		l = kwargs['loss'] if 'loss' in kwargs else 'ls'
		r = kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.1
		e = kwargs['n_estimators'] if 'n_estimators' in kwargs else 100
		model = ensemble.GradientBoostingRegressor(loss=l, learning_rate=r, n_estimators=e)
	elif 'xgboost_regression' in name:
		d = kwargs['max_depth'] if 'max_depth' in kwargs else 3
		ne = kwargs['n_estimators'] if 'n_estimators' in kwargs else 100
		model = xgb.XGBRegressor(max_depth=d, n_estimators=ne)
		print('max_depth:', d, 'n_e:', ne)
	return model


def train_model(model, df_train):
	x_train, y_train = data_label_separation(df_train)
	model.fit(x_train, y_train)
	return model


def test_model(model, df_test):
	x_test, y_test = data_label_separation(df_test, y_key=['id', 'loss'])
	y_predict = model.predict(x_test)
	return y_predict, y_test
	

if __name__ == '__main__':
	#split_training_dataset()
	#data_preprocess('test.csv', 'test_preprocessed.csv')