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
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


datasets = [load_iris(return_X_y=True),
            load_digits(return_X_y=True),
            load_wine(return_X_y=True),
            load_breast_cancer(return_X_y=True),
            ]
dname = ['아이리스', '캔서', '와인', '디지트']

# models = [RandomForestClassifier(),
#           DecisionTreeClassifier(),
#           LogisticRegression(),
#           LinearSVC()]

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
    
    allAlgorithms = all_estimators(type_filter='classifier')    
    
    
    for (name, algorithm) in allAlgorithms:
        try:
            model = algorithm()
            
            scores= cross_val_score(model,x,y, cv=kfold,)
            results= round(np.mean(scores),4)
            
            if max_score <results :
                max_score = results
                max_name = name
        except :
            continue   #에러 무시하고 계속 for문 돌려라, break = for문 중단해라
        
        
    print("===========", dname[i],"==========")
    print("최고모델",max_name, max_score)
    print("=================================")
    
# for i, v in enumerate(datasets):
#     x, y = v
#     for j in scalers :
#         scaler = j
#         x = scaler.fit_transform(x)
#         for (name, algorithm) in allAlgorithms :
#             try :
#                 model = algorithm()
#                 scores = cross_val_score(model, x, y, cv=kfold)
#                 mean = round(np.mean(scores),4)

#                 if max_score < mean :
#                     max_score = mean
#                     max_name = name
#             except :
#                 continue
            
            
print('최고모델 짱짱모델 :', max_name, max_score)
            
            
        


# allAlgorithms = all_estimators(type_filter='classifier')


#튜플 안에 첫번째는 스트링 형태의 모델, 두번째는 클래스로 정의된 모델



# =========== 아이리스 ==========
# 최고모델 LinearDiscriminantAnalysis 0.98
# =================================
# =========== 캔서 ==========
# 최고모델 SVC 0.9866
# =================================
# =========== 와인 ==========
# 최고모델 QuadraticDiscriminantAnalysis 0.9889
# =================================
# =========== 디지트 ==========
# 최고모델 QuadraticDiscriminantAnalysis 0.9889
# =================================