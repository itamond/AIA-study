import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostClassifier
import optuna

#0. fix seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)


scaler_list = [
            #    MinMaxScaler(),
            #    MaxAbsScaler(), 
               StandardScaler(), 
            #    RobustScaler(),
               ]
#1. 데이터
path = './dacon_crime/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
save_path = './dacon_crime/'
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv')
x = train_csv.drop(['TARGET'], axis = 1)
y = train_csv['TARGET']

# 범주형 변수 리스트
qual_col = ['요일', '범죄발생지']

# 원-핫 인코딩
x = pd.get_dummies(x, columns=qual_col)
test = pd.get_dummies(test_csv, columns=qual_col)

# train, test 분리

print(x.shape, test.shape)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=91345)


min_rmse=1


for k in range(100):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=91345, stratify=y)


    for i in scaler_list:
        scaler = i
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        test = scaler.transform(test)

        def objective(trial, x_train, y_train, x_test, y_test, acc):
            param = {
                'iterations': trial.suggest_int('iterations', 5000, 13000),
                'depth': trial.suggest_int('max_depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate',  0.0001,0.1),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.001, 10),
                'one_hot_max_size' : trial.suggest_int('one_hot_max_size',24, 64),
                # 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0, 1),
                'bagging_temperature': trial.suggest_int('bagging_temperature', 0.5, 1),
                'random_strength': trial.suggest_float('random_strength', 0.5, 1),
                # 'border_count': trial.suggest_int('border_count', 64, 128),
                    }
            model = CatBoostClassifier(**param, verbose=0)
            
            model.fit(x_train, y_train)
            val_y_pred = model.predict(x_test)
            accuracy = -(accuracy_score(y_test, val_y_pred))
            # print(accuracy)
            y_pred = model.predict(test)
            y_pred = np.round(y_pred, 3)
            # print(y_pred)
            
            best_accuracy = float('-inf')
            sample_submission_csv['TARGET'] = y_pred
            # submission.to_csv('./_data/dacon_air/Monday2.csv', index=True)           
            if -accuracy > best_accuracy:
                # Update the best accuracy and save the model to disk
                best_accuracy = -accuracy
                pacc = np.round(best_accuracy*10000)
                print(best_accuracy)
                print(type(best_accuracy))
                print(pacc)
                sample_submission_csv.to_csv(save_path+ f'{pacc}' + 'monday1.csv', index=False)
                # Save the predictions to a CSV file
            return accuracy
        opt = optuna.create_study(direction='minimize')
        opt.optimize(lambda trial: objective(trial, x_train, y_train, x_test, y_test, min_rmse), n_trials=10000)
        print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)
