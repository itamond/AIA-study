#랜포 디폴트와 LDA 적용된 값 비교

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_covtype
from sklearn.datasets import load_iris, load_wine
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# 1. 데이터

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_set = pd.read_csv(path + 'train.csv',index_col=0)
test_set = pd.read_csv(path + 'test.csv',index_col=0)

x=train_set.drop(['Outcome'], axis=1)
y=train_set['Outcome']


lda=LinearDiscriminantAnalysis() 

scaler=StandardScaler()

scaler.fit(x)
x=scaler.transform(x)

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=337)

model1 = RandomForestClassifier()
model2 = RandomForestClassifier()
model1.fit(x_train,y_train)

x_lda = lda.fit_transform(x, y)
x_l_train,x_l_test,y_l_train,y_l_test = train_test_split(x_lda,y,train_size=0.8, shuffle=True, random_state=337)
model2.fit(x_lda,y)
print('========================================')
print('데이콘 디아벳', ':', x.shape, '->', x_lda.shape)
print('LDA적용 전 ACC :',model1.score(x_test, y_test))
print('LDA적용 후 ACC :',model2.score(x_l_test, y_l_test))    
print('========================================')


# ========================================
# 아이리스 : (150, 4) -> (150, 2)
# LDA적용 전 ACC : 0.9333333333333333
# LDA적용 후 ACC : 1.0
# ========================================
# 캔서 : (569, 30) -> (569, 1)
# LDA적용 전 ACC : 0.9649122807017544
# LDA적용 후 ACC : 1.0
# ========================================
# 와인 : (178, 13) -> (178, 2)
# LDA적용 전 ACC : 0.9444444444444444
# LDA적용 후 ACC : 1.0
# ========================================
# 디짓스 : (1797, 64) -> (1797, 9)
# LDA적용 전 ACC : 0.9694444444444444
# LDA적용 후 ACC : 1.0
# ========================================
# ========================================
# 데이콘 와인 : (5497, 11) -> (5497, 6)
# LDA적용 전 ACC : 0.69
# LDA적용 후 ACC : 1.0
# ========================================
# ========================================
# 데이콘 디아벳 : (652, 8) -> (652, 1)
# LDA적용 전 ACC : 0.7709923664122137
# LDA적용 후 ACC : 1.0
# ========================================



