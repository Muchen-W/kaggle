import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle

dataDir = './data/'
# load original train/test data
#file_train = open(dataDir + 'train.dump', 'rb')
#file_test = open(dataDir + 'test.dump', 'rb')

#df_train = pickle.load(file_train)
#df_test = pickle.load(file_test)

file_train_wo = open(dataDir + 'train_without_noise.dump', 'rb')
df_train_wo = pickle.load(file_train_wo)
df_train = df_train_wo

miss_cnt = df_train.apply(lambda x: sum(pd.isnull(x)))
miss_pct = miss_cnt / df_train.shape[0]
miss_pct = miss_pct[miss_pct > 0]

miss_pct_sort = miss_pct.sort_values(ascending=False)
miss_names = np.array(miss_pct_sort.index)

ax = sns.barplot(miss_names, miss_pct_sort, color='b')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.show()

file_train_wo.close()
#file_train.close()
#file_test.close()