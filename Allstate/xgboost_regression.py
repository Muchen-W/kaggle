import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


def load_file(fread):
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


def train_xgb_model(df_train):
	x_train, y_train = data_label_separation(df_train)
	#param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
	#num_round = 2
	bst = xgb.XGBRegressor(max_depth=3, n_estimators=100, silent=True, objective='reg:linear')
	bst.fit(x_train, y_train)
	#print(x_train.dtypes, y_train.dtypes)
	return bst


def validate_xgb_model(model, df_validate):
	x_validate, y_validate = data_label_separation(df_validate)
	score = model.score(x_validate, y_validate)
	y_predict = model.predict(x_validate)
	mae = mean_absolute_error(y_validate, y_predict)
	return score, mae


if __name__ == '__main__':
	#fread = ['train_sample_split.csv', 'validate_sample_split.csv', 'test_sample_split.csv']
	fread = ['train_split.csv', 'validate_split.csv', 'test_split.csv']
	df_train, df_validate, df_test = load_file(fread)
	#x_train, y_train = data_label_separation(df_train)
	model = train_xgb_model(df_train)
	score, mae = validate_xgb_model(model, df_validate)
	print('score:', score, 'mae:', mae)