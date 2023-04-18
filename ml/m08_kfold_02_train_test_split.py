from sklearn.utils import all_estimators
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import QuantileRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, r2_score
warnings.filterwarnings(action='ignore')

path_ddarung = './_data/ddarung/'
path_kaggle = './_data/kaggle_bike/'

ddarung_train = pd.read_csv(path_ddarung + 'train.csv', index_col=0).dropna()
kaggle_train = pd.read_csv(path_kaggle + 'train.csv', index_col=0).dropna()

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes, ddarung_train, kaggle_train]

algorithms_classifier = all_estimators(type_filter='classifier')
algorithms_regressor = all_estimators(type_filter='regressor')

max_score = 0
max_name = ''
max_acc = 0
max_acc_name = ''
max_r2 = 0
max_r2_name = ''

scaler_list = [RobustScaler(), StandardScaler(), MinMaxScaler(), MaxAbsScaler()]
n_split = 10
kf = KFold(n_splits=n_split, shuffle=True, random_state=123)

for i in range(len(data_list)):
    # if i<4:
    #     x, y = data_list[i](return_X_y=True)
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)
    #     for j in scaler_list:
    #         scaler = j
    #         x = scaler.fit_transform(x)
    #         for name, algorithm in algorithms_classifier:
    #             try:
    #                 model = algorithm()
    #                 results = cross_val_score(model, x_train, y_train, cv=kf)
    #                 if max_score<np.mean(results):
    #                     max_score=np.mean(results)
    #                     max_name = name
    #                 y_predict = cross_val_predict(model, x_test, y_test)
    #                 acc = accuracy_score(y_test, y_predict)
    #                 print(type(j).__name__, ' - ', data_list[i].__name__, name, 'predict acc : ', acc)
    #                 if max_acc<acc:
    #                     max_acc=acc
    #                     max_acc_name = name
    #                 # print(type(j).__name__, data_list[i].__name__, name, 'acc :', results, 'mean of cross_val_score : ', round(np.mean(results), 5))
    #             except:
    #                 # print(type(j).__name__, data_list[i].__name__, name, 'set default value first')
    #                 continue
    #         print('\n', type(j).__name__, ' - ', data_list[i].__name__, 'max_score :', max_name, max_score)
    #         print('\n', type(j).__name__, ' - ', data_list[i].__name__, 'max_predict_acc :', max_acc_name, max_acc, '\n')
    if 4<=i<6:
        x, y = data_list[i](return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)
        for j in scaler_list:
            scaler = j
            x = scaler.fit_transform(x)
            for name, algorithm in algorithms_regressor:
                try:
                    model = algorithm()
                    results = cross_val_score(model, x, y, cv=kf)
                    if max_score<np.mean(results):
                        max_score=np.mean(results)
                        max_name=name
                    y_predict = cross_val_predict(model, x_test, y_test)
                    r2 = r2_score(y_test, y_predict)
                    print('predict r2 : ', r2)
                    if max_r2<r2:
                        max_r2=r2
                        max_r2_name = name
                    # print(type(j).__name__, data_list[i].__name__, name, 'acc :', results, 'mean of cross_val_score : ', round(np.mean(results), 5))
                except:
                    # print(type(j).__name__, data_list[i].__name__, name, 'set default value first')
                    continue
            print('\n', type(j).__name__, ' - ', data_list[i].__name__, 'max_score :', max_name, max_score)
            print('\n', type(j).__name__, ' - ', data_list[i].__name__, 'max_predict_r2 :', max_r2_name, max_r2, '\n')  
    elif i==6:
        x = data_list[i].drop(['count'], axis=1)
        y = data_list[i]['count']
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)
        for j in scaler_list:
            scaler = j
            x = scaler.fit_transform(x)
            for name, algorithm in algorithms_regressor:
                try:
                    model = algorithm()
                    results = cross_val_score(model, x, y, cv=kf)
                    if max_score<np.mean(results):
                        max_score=np.mean(results)
                        max_name=name
                    y_predict = cross_val_predict(model, x_test, y_test)
                    r2 = r2_score(y_test, y_predict)
                    print('predict r2 : ', r2)
                    if max_r2<r2:
                        max_r2=r2
                        max_r2_name = name
                    # print(type(j).__name__, 'ddarung', name, 'acc :', results, 'mean of cross_val_score : ', round(np.mean(results), 5))
                except:
                    # print(type(j).__name__, 'ddarung', name, 'set deault value first')
                    continue
            print('\n', type(j).__name__, ' - ', 'ddarung max_score :', max_name,  max_score)
            print('\n', type(j).__name__, ' - ', data_list[i].__name__, 'max_predict_r2 :', max_r2_name, max_r2, '\n')
    elif i==7:
        x = data_list[i].drop(['casual', 'registered', 'count'], axis=1)
        y = data_list[i]['count']
        for j in data_list:
            scaler = j
            x = scaler.fit_transform
            for name, algorithm in algorithms_regressor:
                try:
                    model = algorithm()
                    results = cross_val_score(model, x, y, cv=kf)
                    if max_score<np.mean(results):
                        max_score=np.mean(results)
                        max_name=name
                    y_predict = cross_val_predict(model, x_test, y_test)
                    r2 = r2_score(y_test, y_predict)
                    print('predict r2 : ', r2)
                    if max_r2<r2:
                        max_r2=r2
                        max_r2_name = name
                    # print(type(j).__name__, 'kaggle', name, 'acc :', results, 'mean of cross_val_score : ', round(np.mean(results), 5))
                except:
                    # print(type(j).__name__, 'kaggle', name, 'set deault value first')
                    continue
            print('\n', type(j).__name__, ' - ', 'kaggle max_score :', max_name, max_score)
            print('\n', type(j).__name__, ' - ', data_list[i].__name__, 'max_predict_r2 :', max_r2_name, max_r2, '\n')
