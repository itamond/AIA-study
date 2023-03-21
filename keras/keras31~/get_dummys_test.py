import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

plt.style.use('seaborn')
sns.set(font_scale=1.5)

import warnings
warnings.filterwarnings("ignore")


바나나, 1, 사과, 123


#1. 데이터
os.listdir("./_data/kaggle_house/")


df_train = pd.read_csv('./_data/kaggle_house/train.csv')
df_test = pd.read_csv('./_data/kaggle_house/test.csv')
print(df_train.head(30))


df_train = pd.get_dummies(df_train, dummy_na=True)
print(df_train.head(30))

# print(df_train.shape, df_test.shape)    #(1460, 80) (1459, 79)

# print(df_train.head())
# print(type(df_train))  #<class 'pandas.core.frame.DataFrame'>

# numfeat = df_train.dtypes[df_train.dtypes !='object']
# # print('숫자형 피쳐 :', len(numerical_feats))

# catfeat = df_train.dtypes[df_train.dtypes =='object']

# print('numfeat :',numfeat)
# print('catfeat :',catfeat)    

# print(type(numfeat))  #<class 'pandas.core.indexes.base.Index'>
# print(type(catfeat))  #<class 'pandas.core.indexes.base.Index'>


# print(type(numfeat))  #<class 'pandas.core.frame.DataFrame'>  
# print(type(catfeat))  #<class 'pandas.core.frame.DataFrame'>