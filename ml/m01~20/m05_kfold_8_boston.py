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



path_ddarung = './_data/ddarung/'
path_kaggle = './_data/kaggle_bike/'
ddarung_train = pd.read_csv(path_ddarung + 'train.csv', index_col=0).dropna()
kaggle_train = pd.read_csv(path_kaggle + 'train.csv', index_col=0).dropna()
data_list = [fetch_california_housing, load_diabetes, ddarung_train, kaggle_train]
name_list = ['fetch_california_housing', 'load_diabetes', 'ddarung_train', 'kaggle_train']


model = RandomForestRegressor()

n_splits = 5

kfold = KFold(n_splits = n_splits, shuffle=True, random_state=123)

for i in range(len(data_list)):
    if i<2:
        x, y = data_list[i](return_X_y=True)
    elif i==2:
        x = data_list[i].drop(['count'], axis=1)
        y = data_list[i]['count']
    else:
        x = data_list[i].drop(['casual', 'registered', 'count'], axis=1)
        y = data_list[i]['count']
    print(f"\n데이터셋 {i+1}:")
    scores = cross_val_score(model, x, y, cv=kfold, n_jobs=-1)   # n_jobs= 사용할 코어 갯수
    print('ACC :', scores, '\ncross_val_score 평균 : ', round(np.mean(scores), 4))


# 데이터셋 1: 캘리포니아
# ACC : [0.81269898 0.82441359 0.80980714 0.79361004 0.80495617] 
# cross_val_score 평균 :  0.8091

# 데이터셋 2: 디아벳스
# ACC : [0.51650115 0.40464892 0.44651967 0.49944982 0.26372522] 
# cross_val_score 평균 :  0.4262

# 데이터셋 3: 따릉이
# ACC : [0.79484812 0.76819041 0.77809954 0.80816321 0.74793512] 
# cross_val_score 평균 :  0.7794

# 데이터셋 4: 캐글
# ACC : [0.27543799 0.29039524 0.27920683 0.32344764 0.32939805] 
# cross_val_score 평균 :  0.2996
