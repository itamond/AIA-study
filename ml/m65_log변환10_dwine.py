from sklearn.datasets import  load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer #scaling 
# :: QuantileTransformer, RobustScaler ->이상치에 자유로움
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt


#1. 데이터
path = './_data/dacon_wine/'
path_save = './_save/dacon/wine/'
train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

# Remove rows with single class label
single_class_label = train_csv['quality'].nunique() == 1
if single_class_label:
    train_csv = train_csv[train_csv['quality'] != train_csv['quality'].unique()[0]]

# Label encode 'type'
le = LabelEncoder()
train_csv['type'] = le.fit_transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

# Split data
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']-3
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state = 337, shuffle = True,)

import pandas as pd

# scaler = StandardScaler()
# x_trian = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = LogisticRegression()
model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
results = r2_score(y_test, y_predict)
print("기본 결과 : ", round(results,4))


################### 로그 변환!!! ######################
# df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
# print(x.describe())

# df.plot.box()
# plt.title("wine")
# plt.xlabel('Feature')
# plt.ylabel('data values')
# plt.show()

#print(df['B'].head())
x = np.log1p(x)
# y = np.log1p(y)
#print(df['B'].head())


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123   
)

# scaler = StandardScaler()
# x_trian = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = LogisticRegression()
model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
results = accuracy_score(y_test, y_predict)
print("LOG 변환 후 결과 : ", round(results,4))



'''
기본 결과 :  0.4183
LOG 변환 후 결과 :  0.6482
'''