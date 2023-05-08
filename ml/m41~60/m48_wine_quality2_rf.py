import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

#1. 데이터
path = './_data/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                 

test_csv = pd.read_csv(path + 'test.csv',
                        index_col=0)                 
print(train_csv.shape)
# (5497, 13)

print(test_csv.shape)
# (1000, 12)

# print(train_csv['quality'].value_counts())
# 6    2416
# 5    1788
# 7     924
# 4     186
# 8     152
# 3      26
# 9       5

# print(train_csv.columns)
# Index(['quality', 'fixed acidity', 'volatile acidity', 'citric acid',
#        'residual sugar', 'chlorides', 'free sulfur dioxide',
#        'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
#        'type'],
#       dtype='object')

# print(test_csv.columns)
# Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#        'pH', 'sulphates', 'alcohol', 'type'],
#       dtype='object')


le = LabelEncoder()
le.fit(train_csv['type'])

aaa = le.transform(train_csv['type'])
train_csv['type'] = aaa
test_csv['type'] = le.transform(test_csv['type'])

x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']

# 



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=3377,
                                                    shuffle=True,
                                                    stratify= y
                                                    )

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(y_train.shape)
print(y)

# 2. 모델구성

model = RandomForestClassifier(random_state=3377)

# 3. 컴파일, 훈련


model.fit(x_train, y_train,)

#4. 평가, 예측
# print("최상의 매개변수 :", model.best_params_)

print('최상의 점수 :', model.score(x_test,y_test))
# results = model.score(x_test, y_test)
# print("최종점수 : ", results)

y_pred =model.predict(test_csv)

submission = pd.read_csv('./_data/dacon_wine/sample_submission.csv', index_col=0)

submission['quality'] = y_pred

submission.to_csv('./_data/dacon_wine/sub.csv')