
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
                  '당뇨병 :']

model_name_list = ['LinearSVC :',
                   'LogisticRegression :',
                   'DecisionTreeClassifier :',
                   'RandomForestClassifier :']

#2. 모델
for i,value in enumerate(data_list):
    x, y = value                #에뉴머레이트 하면 순서가 i 값은 v여서 x, y 는 v로 지정
    # print(x.shape, y.shape)
    print("=============================================")
    print(data_name_list[i])
    
    for j, value2 in enumerate(model_list):
        model = value2
        #컴파일, 훈련
        model.fit(x,y)
        #평가, 예측
        result = model.score(x,y)
        print(model_name_list[j], result)
        y_predict = model.predict(x)
        acc=accuracy_score(y, y_predict)
        print('accuracy_score :',acc)
        