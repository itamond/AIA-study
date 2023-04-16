import numpy as np
from sklearn.datasets import fetch_covtype, load_iris, load_breast_cancer, load_wine, load_digits
import warnings
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
warnings.filterwarnings(action='ignore')
from sklearn.metrics import r2_score, accuracy_score

data_list = [load_iris(return_X_y=True), 
             load_breast_cancer(return_X_y=True), 
             load_wine(return_X_y=True), 
             load_digits(return_X_y=True),
             ]

model_list = [LinearSVC(),
              LogisticRegression(),
              DecisionTreeClassifier(),
              RandomForestClassifier(),
              ]

data_name_list = ['아이리스 :',
                  '브레스트 캔서 :',
                  '와인 :',
                  '디짓스 :']


model_name_list = ['LinearSVC :',
                   'LogisticRegression :',
                   'DecisionTreeClassifier :',
                   'RandomForestClassifier :']

#각 데이터셋마다 모델을 적용하고 점수를 출력하는 포문 생성

for i, value in enumerate(data_list):
    x, y = value
    print('########################')
    print(data_name_list[i])
    for j, value2 in enumerate(model_list):
        model = value2
        model.fit(x,y)
        result = model.score(x,y)
        y_predict = model.predict(x)
        acc = accuracy_score(y, y_predict)
        print(model_name_list[j], result)
        print('acc :', acc)
        
        