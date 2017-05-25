import pandas as pd
import pickle

dataDir = './data/'
file_train = 'train.csv'
file_test = 'test.csv'
file_train_wo = 'train_without_noise.csv'

df_train = pd.read_csv(dataDir + file_train)
df_test = pd.read_csv(dataDir + file_test)
df_train_wo = pd.read_csv(dataDir + file_train_wo)

with open(dataDir + file_train[:file_train.find('.')] + '.dump', 'wb') as fw_train:
	pickle.dump(df_train, fw_train)
with open(dataDir + file_test[:file_test.find('.')] + '.dump', 'wb') as fw_test:
	pickle.dump(df_test, fw_test)
with open(dataDir + file_train_wo[:file_train_wo.find('.')] + '.dump', 'wb') as fw_train_wo:
	pickle.dump(df_train_wo, fw_train_wo)