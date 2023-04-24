#분류 싹 모아서 테스트

import numpy as np
from sklearn.datasets import fetch_covtype, load_iris, load_breast_cancer, load_wine, load_digits, load_diabetes
from sklearn.datasets import fetch_california_housing
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import all_estimators



path_ddarung = './_data/ddarung/'
path_kaggle = './_data/kaggle_bike/'
ddarung_train = pd.read_csv(path_ddarung + 'train.csv', index_col=0).dropna()
kaggle_train = pd.read_csv(path_kaggle + 'train.csv', index_col=0).dropna()
data_list = [ load_diabetes, fetch_california_housing, ddarung_train, kaggle_train]
dname = [ 'load_diabetes', 'fetch_california_housing', 'ddarung_train', 'kaggle_train']


models = [DecisionTreeRegressor(), RandomForestRegressor()]
n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle=True, random_state=123)
allAlgorithms = all_estimators(type_filter='regressor')
max_score = 0
max_name =  '최고'

for i, v in enumerate(range(len(data_list))):
    if i<2:
        x, y = data_list[i](return_X_y=True)
    elif i==2:
        x = data_list[i].drop(['count'], axis=1)
        y = data_list[i]['count']
    else:
        x = data_list[i].drop(['casual', 'registered', 'count'], axis=1)
        y = data_list[i]['count']
    
    allAlgorithms = all_estimators(type_filter='regressor')    
    
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
    print("==================================")
    