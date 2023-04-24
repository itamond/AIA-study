#분류 싹 모아서 테스트

import numpy as np
from sklearn.datasets import fetch_covtype, load_iris, load_breast_cancer, load_wine, load_digits
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score



datasets = [load_iris(return_X_y=True), load_digits(return_X_y=True), load_wine(return_X_y=True), load_breast_cancer(return_X_y=True),fetch_covtype(return_X_y=True)]
model = RandomForestClassifier()
scaler = MinMaxScaler()

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle=True, random_state=123)




for i, datasets in enumerate(datasets):
    x,y = datasets
    print(f"\n데이터셋 {i+1}:")
    scores = cross_val_score(model, x, y, cv=kfold, n_jobs=-1)   # n_jobs= 사용할 코어 갯수
    print('ACC :', scores, '\ncross_val_score 평균 : ', round(np.mean(scores), 4))
