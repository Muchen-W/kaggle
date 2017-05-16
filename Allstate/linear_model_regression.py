import pandas as pd
import numpy as np
from sklearn import linear_model, kernel_ridge, svm, ensemble, grid_search
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

import xgboost as xgb


def load_file(fread):
	#fread = 'train_split.csv'
	for fr in fread:
		if 'train' in fr:
			df_train = pd.read_csv(fr)
		elif 'validate' in fr:
			df_validate = pd.read_csv(fr)
		else:
			df_test = pd.read_csv(fr)
	return df_train, df_validate, df_test


def data_label_separation(df, y_key='loss', x_drop_key=['id', 'loss']):
	'''
	remove columns of 'id' and 'loss' for X;
	keep 'loss' column as label/ground truth
	'''
	X = df.drop(x_drop_key, axis=1)
	y = df[y_key]
	return X, y


def model_initialization(name, **kwargs):
	if name == 'linear_regression':
		model = linear_model.LinearRegression()
	elif 'ridge_regression' in name:
		alpha = kwargs['alpha']
		model = linear_model.Ridge(alpha)
	elif 'lasso_regression' in name:
		alpha = kwargs['alpha']
		model = linear_model.Lasso(alpha)
	elif 'elastic_net' in name:
		alpha = kwargs['alpha'] if 'alpha' in kwargs else 1
		r = kwargs['l1_ratio'] if 'l1_ratio' in kwargs else 0.5
		model = linear_model.ElasticNet(alpha=alpha, l1_ratio=r, max_iter=2000)
	elif 'SGD_regression' in name:
		l = kwargs['loss'] if 'loss' in kwargs else 'squared_loss'
		p = kwargs['penalty'] if 'penalty' in kwargs else 'l2'
		alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.0001
		r = kwargs['l1_ratio'] if 'l1_ratio' in kwargs else 0.15
		model = linear_model.SGDRegressor(loss=l, penalty=p, alpha=alpha, l1_ratio=r, n_iter=5)
	elif 'kernel_ridge' in name:
		kernel = kwargs['kernel']
		model = kernel_ridge.KernelRidge(kernel=kernel)
	elif 'support_vector_regression' in name:
		kernel = kwargs['kernel']
		c = kwargs['C'] if 'C' in kwargs else 1
		d = kwargs['degree'] if 'degree' in kwargs else 3
		g = kwargs['gamma'] if 'gamma' in kwargs else 'auto'
		model = svm.SVR(kernel=kernel, C=c, degree=d, gamma=g, cache_size=1000, max_iter=2000)
	elif 'gradient_boost_regression' in name:
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
	#print(x_train.dtypes, y_train.dtypes)
	model.fit(x_train, y_train)
	#print(model.best_score_, model.best_params_)
	return model


def validate_model(model, df_validate):
	x_validate, y_validate = data_label_separation(df_validate)
	score = model.score(x_validate, y_validate)
	y_predict = model.predict(x_validate)
	mae = mean_absolute_error(y_validate, y_predict)
	return score, mae


def test_model(model, df_test):
	x_test, y_test = data_label_separation(df_test, y_key=['id', 'loss'])
	y_predict = model.predict(x_test)
	return y_predict, y_test


def test_model_nolabel(model, df_test):
	x_test = df_test.drop('id', axis=1)
	y_predict = model.predict(x_test)
	df_predict = pd.DataFrame(y_predict, columns=['loss'])
	df_predict = pd.concat([df_test['id'], df_predict], axis=1)
	return df_predict


def complete_procedure(name, output=False, **kwargs):
	'''
	model initialization; load files; train model; validate model; test model.
	'''
	model = model_initialization(name, **kwargs)
	#fread = ['train_sample_split.csv', 'validate_sample_split.csv', 'test_sample_split.csv']
	fread = ['train_split.csv', 'validate_split.csv', 'test_split.csv']
	df_train, df_validate, df_test = load_file(fread)
	model = train_model(model, df_train)
	score, mae = validate_model(model, df_validate)
	y_predict, y_test = test_model(model, df_test)

	if output:
		output_test_prediction(name, y_predict, y_test)
		output_validation_score(name, score, mae)
	
	#return model, score, mae
	return model, score, mae, y_predict, y_test['loss']


def complete_model_train_test(name, output=False, **kwargs):
	'''
	model initialization; load files; train model; validate model; test model.
	'''
	model = model_initialization(name, **kwargs)
	#fread = ['train_sample_split.csv', 'validate_sample_split.csv', 'test_sample_split.csv']
	fread = ['train_preprocessed.csv', 'validate_split.csv', 'test_preprocessed.csv']
	#fread = ['train_split.csv', 'validate_split.csv', 'test_preprocessed.csv']
	df_train, df_validate, df_test = load_file(fread)
	df_test_reindex = df_test.reindex_axis(df_train.keys().drop('loss'), axis=1).fillna(value=0, axis=1)

	model = train_model(model, df_train)
	score, mae = validate_model(model, df_validate)
	df_predict = test_model_nolabel(model, df_test_reindex)

	if output:
		fwrite = 'test_prediction_' + name + '.csv'
		df_predict.to_csv(fwrite, index=False, float_format='%.2f')
		output_validation_score(name, score, mae)
	
	#return model, score, mae
	return model, score, mae, df_predict


def output_test_prediction(name, y_predict, y_test):
	fwrite = 'test_prediction_' + name + '.csv'
	y_predict = pd.DataFrame(y_predict, columns=['predict_loss'])
	result = pd.concat([y_test, y_predict], axis=1)
	result.to_csv(fwrite, index=False, float_format='%.2f')


def output_validation_score(name, score, mae):
	fw = 'validation_score_mae_' + name + '.txt'
	f = open(fw, 'w')
	f.write('score:' + str(score) + '\n')
	f.write('mae:' + str(mae) + '\n')
	f.close()


def model_explore():
	#model_name = 'linear_regression'
	#model, score, mae, y_predict, y_test = complete_procedure(model_name, output=True)
	
	#model_name = 'ridge_regression_alpha100'
	#model, score, mae, y_predict, y_test = complete_procedure(model_name, output=False, alpha=100)
	
	#model_name = 'lasso_regression_alpha0.1'
	#model, score, mae, y_predict, y_test = complete_procedure(model_name, output=False, alpha=0.1)

	#model_name = 'elastic_net_alpha1e-2'
	#model, score, mae, y_predict, y_test = complete_procedure(model_name, output=True, alpha=1e-2, l1_ratio=0.5)
	
	#model_name = 'SGD_regression_squared_loss_r0'
	#model, score, mae, y_predict, y_test = complete_procedure(model_name, output=True, loss='squared_loss', l1_ratio=0)

	# MemoryError
	#model_name = 'kernel_ridge'
	#model, score, mae, y_predict, y_test = complete_procedure(model_name, output=False, kernel='linear')

	# linear, poly, rbf, sigmoid
	#model_name = 'support_vector_regression_linear_C1e1'
	#model, score, mae, y_predict, y_test = complete_procedure(model_name, output=True, kernel='linear', C=1e2)

	#model_name = 'gradient_boost_regression_ls_r0.2_ne300'
	#model, score, mae, y_predict, y_test = complete_procedure(model_name, output=True, loss='ls', learning_rate=0.2, n_estimators=300)	

	model_name = 'xgboost_regression_linear_d9_ne300'
	model, score, mae, y_predict, y_test = complete_procedure(model_name, output=True, max_depth=9, n_estimators=300)

	print('score:', score, 'mae:', mae)
	#plt.scatter(y_predict, y_test)
	#plt.show()


if __name__ == '__main__':
	#model_explore()

	#model_name = 'xgboost_regression_linear_d9_ne600_submission'
	#model, score, mae, df_predict = complete_model_train_test(model_name, output=True, max_depth=9, n_estimators=600)
	model_name = 'gradient_boost_regression_ls_r0.1_ne900_submission'
	model, score, mae, y_predict, y_test = complete_model_train_test(model_name, output=True, loss='ls', learning_rate=0.1, n_estimators=900)	
	print('score:', score, 'mae:', mae)