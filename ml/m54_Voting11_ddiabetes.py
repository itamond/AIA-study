# [실습] 각종 모델 넣기
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import all_estimators
from sklearn.ensemble import BaggingClassifier, VotingClassifier, VotingRegressor
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes]

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_set = pd.read_csv(path + 'train.csv',index_col=0)
test_set = pd.read_csv(path + 'test.csv',index_col=0)

x=train_set.drop(['Outcome'], axis=1)
y=train_set['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8, shuffle=True, stratify=y)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 2. 모델
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
dt = DecisionTreeClassifier()


model = VotingClassifier(
    estimators=[('LR',lr),('KNN', knn),('DT', dt)],
    voting = 'soft',verbose=0)


model.fit(x_train, y_train)
# 4. 평가, 예측
y_pred = model.predict(x_test)

print('acc :', model.score(x_test, y_test))
print('acc :', accuracy_score(y_test, y_pred))


classifiers = [lr, knn, dt]
for model2 in classifiers :
    model2.fit(x_train,y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_predict)
    class_name = model2.__class__.__name__  # 
    print("{0}정확도 :{1:.4f}".format(class_name, score2))#{0}정확도 :{1:.4f}를 출력하겠다. 중괄호 안에 변수 가능. 뒤에 지정한 class_name, score2가 들어감

# acc : 0.6793893129770993
# acc : 0.6793893129770993
# LogisticRegression정확도 :0.7634
# KNeighborsClassifier정확도 :0.7023
# DecisionTreeClassifier정확도 :0.6870