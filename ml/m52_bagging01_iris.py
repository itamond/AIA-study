# [실습] 각종 모델 넣기
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.ensemble import BaggingClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes]

for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8, shuffle=True)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 2. 모델
    if i <5:
        allalgorithms = all_estimators(type_filter='classifier')
    else:
        allalgorithms = all_estimators(type_filter='regressor')
    for (name, algorithm) in allalgorithms:
        try:
            model = BaggingClassifier(algorithm(), n_estimators=10, n_jobs=-1, random_state=337, bootstrap=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            # 4. 평가, 예측
            print(data_list[i].__name__,name, 'score : ', model.score(x_test, y_test))
            print(data_list[i].__name__,name, 'acc : ', accuracy_score(y_test, y_pred))
        except:
            continue