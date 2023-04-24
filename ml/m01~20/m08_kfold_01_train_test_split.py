#분류 싹 모아서 테스트

import numpy as np
from sklearn.datasets import fetch_covtype, load_iris, load_breast_cancer, load_wine, load_digits
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


datasets = [load_iris(return_X_y=True),
            load_digits(return_X_y=True),
            load_wine(return_X_y=True),
            load_breast_cancer(return_X_y=True),
            ]
dname = ['아이리스', '디지트', '와인', '캔서']


scalers = [MinMaxScaler(),
           RobustScaler(),
           StandardScaler(), 
           MaxAbsScaler()]

scaler_names = ['민맥스',
                '로버스트',
                '스탠다드',
                '맥샙스']


n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle=True, random_state=123)
allAlgorithms = all_estimators(type_filter='classifier')
max_score = 0
max_name =  '최고'

for i, v in enumerate(datasets):
    x, y = v
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=33, shuffle=True)
    allAlgorithms = all_estimators(type_filter='classifier')    
    
    
    for (name, algorithm) in allAlgorithms:
        try:
            model = algorithm()
            
            scores= cross_val_score(model,x_train,y_train, cv=kfold,)
            results= round(np.mean(scores),4)
            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)   # cross_val_score에서 predict 하는법

            acc = round(accuracy_score(y_test, y_predict),4)
            
            if max_score <results :
                max_score = results
                max_name = name
        except :
            continue   #에러 무시하고 계속 for문 돌려라, break = for문 중단해라
        
        
    print("===========", dname[i],"==========")
    print("최고모델 :",max_name, '\nScore:',max_score, '\nACC:',acc)
    print("=================================")


# =========== 아이리스 ==========
# 최고모델 LinearDiscriminantAnalysis
# Score: 0.975
# ACC: 0.7333
# =================================
# =========== 디지트 ==========
# 최고모델 SVC
# Score: 0.9868
# ACC: 0.95
# =================================
# =========== 와인 ==========
# 최고모델 QuadraticDiscriminantAnalysis
# Score: 0.9931
# ACC: 0.6944
# =================================
# =========== 캔서 ==========
# 최고모델 QuadraticDiscriminantAnalysis
# Score: 0.9931
# ACC: 0.8509
# =================================



# print('최고모델 짱짱모델 :', max_name, max_score)
            
            
        

