import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, fetch_covtype
from sklearn.preprocessing import MinMaxScaler,LabelEncoder, QuantileTransformer, PowerTransformer,MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
import pandas as pd

#1 데이터

d_wine = 'c:/AIA/AIA-study/_data/dacon_wine/'
d_train = pd.read_csv(d_wine + 'train.csv', index_col= 0)

single_class_label = d_train['quality'].nunique() == 1
if single_class_label:
    d_train = d_train[d_train['quality'] != d_train['quality'].unique()[0]]
le = LabelEncoder()
d_train['type'] = le.fit_transform(d_train['type'])

x1 = d_train.drop(['quality'], axis=1)
y1 = d_train['quality']-3


d_diabete = 'c:/AIA/AIA-study/_data/dacon_diabetes/'
dia_train = pd.read_csv(d_diabete + 'train.csv',index_col=0)

x2=dia_train.drop(['Outcome'], axis=1)
y2=dia_train['Outcome']





data_list={'iris' : load_iris,
           'cancer' : load_breast_cancer,
           'wine' : load_wine,
           'digits' : load_digits,
        #    'covtype' : fetch_covtype,
           'd_wine' : (x1, y1),
           'ddiabete' : (x2, y2)
           }

scaler_list = {'MinMax' : MinMaxScaler(),
               'Max' : MaxAbsScaler(),
               'Standard' : StandardScaler(),
               'Robust' : RobustScaler(),
               'QuantileTransformer' : QuantileTransformer(n_quantiles=10),
               'PowerTransformer' : PowerTransformer(method='yeo-johnson')}


for d in data_list:
    max_ACC = -1
    max_scaler = None
    if d == 'd_wine' or d == 'ddiabete':
        x, y = data_list[d]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)
    elif d == 'iris' or d == 'cancer' or d == 'wine' or d == 'digits' :
        x, y = data_list[d](return_X_y = True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)
    
    for s in scaler_list:
        scaler = scaler_list[s]
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        acc = np.round(accuracy_score(y_test, y_predict),2)
        if acc > max_ACC:
            max_ACC = acc
            max_scaler = s
        print(f'데이터 : {d}, 스케일러 : {s}, 정확도 : {acc}')
    print(f'데이터 : {d}, 가장 높은 결정 계수 : {max_ACC}, 가장 높은 결정 계수 스케일러 : {max_scaler}')

# 데이터 : iris, 스케일러 : MinMax, 정확도 : 0.96
# 데이터 : iris, 스케일러 : Max, 정확도 : 0.96
# 데이터 : iris, 스케일러 : Standard, 정확도 : 0.98
# 데이터 : iris, 스케일러 : Robust, 정확도 : 0.96
# 데이터 : iris, 스케일러 : QuantileTransformer, 정확도 : 0.96
# 데이터 : iris, 스케일러 : PowerTransformer, 정확도 : 0.96
# 데이터 : iris, 가장 높은 결정 계수 : 0.98, 가장 높은 결정 계수 스케일러 : Standard
# 데이터 : cancer, 스케일러 : MinMax, 정확도 : 0.92
# 데이터 : cancer, 스케일러 : Max, 정확도 : 0.92
# 데이터 : cancer, 스케일러 : Standard, 정확도 : 0.94
# 데이터 : cancer, 스케일러 : Robust, 정확도 : 0.92
# 데이터 : cancer, 스케일러 : QuantileTransformer, 정확도 : 0.94
# 데이터 : cancer, 스케일러 : PowerTransformer, 정확도 : 0.94
# 데이터 : cancer, 가장 높은 결정 계수 : 0.94, 가장 높은 결정 계수 스케일러 : Standard
# 데이터 : wine, 스케일러 : MinMax, 정확도 : 0.96
# 데이터 : wine, 스케일러 : Max, 정확도 : 0.96
# 데이터 : wine, 스케일러 : Standard, 정확도 : 0.96
# 데이터 : wine, 스케일러 : Robust, 정확도 : 0.96
# 데이터 : wine, 스케일러 : QuantileTransformer, 정확도 : 0.96
# 데이터 : wine, 스케일러 : PowerTransformer, 정확도 : 0.96
# 데이터 : wine, 가장 높은 결정 계수 : 0.96, 가장 높은 결정 계수 스케일러 : MinMax
# 데이터 : digits, 스케일러 : MinMax, 정확도 : 0.97
# 데이터 : digits, 스케일러 : Max, 정확도 : 0.97
# 데이터 : digits, 스케일러 : Standard, 정확도 : 0.97
# 데이터 : digits, 스케일러 : Robust, 정확도 : 0.97
# 데이터 : digits, 스케일러 : QuantileTransformer, 정확도 : 0.97
# 데이터 : digits, 스케일러 : PowerTransformer, 정확도 : 0.96
# 데이터 : digits, 가장 높은 결정 계수 : 0.97, 가장 높은 결정 계수 스케일러 : MinMax
# 데이터 : d_wine, 스케일러 : MinMax, 정확도 : 0.67
# 데이터 : d_wine, 스케일러 : Max, 정확도 : 0.65
# 데이터 : d_wine, 스케일러 : Standard, 정확도 : 0.66
# 데이터 : d_wine, 스케일러 : Robust, 정확도 : 0.66
# 데이터 : d_wine, 스케일러 : QuantileTransformer, 정확도 : 0.67
# 데이터 : d_wine, 스케일러 : PowerTransformer, 정확도 : 0.67
# 데이터 : d_wine, 가장 높은 결정 계수 : 0.67, 가장 높은 결정 계수 스케일러 : MinMax
# 데이터 : ddiabete, 스케일러 : MinMax, 정확도 : 0.79
# 데이터 : ddiabete, 스케일러 : Max, 정확도 : 0.76
# 데이터 : ddiabete, 스케일러 : Standard, 정확도 : 0.79
# 데이터 : ddiabete, 스케일러 : Robust, 정확도 : 0.79
# 데이터 : ddiabete, 스케일러 : QuantileTransformer, 정확도 : 0.79
# 데이터 : ddiabete, 스케일러 : PowerTransformer, 정확도 : 0.81
# 데이터 : ddiabete, 가장 높은 결정 계수 : 0.81, 가장 높은 결정 계수 스케일러 : PowerTransformer
