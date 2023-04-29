import random
import os
import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import datetime
import optuna


scaler_list = [MinMaxScaler(), MaxAbsScaler(), StandardScaler(), RobustScaler()]
model_list = [CatBoostClassifier()]


cat = CatBoostClassifier()




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Fixed Seed

def csv_to_parquet(csv_path, save_name):
    df = pd.read_csv(csv_path)
    df.to_parquet(f'./{save_name}.parquet')
    del df
    gc.collect()
    print(save_name, 'Done.')

csv_to_parquet('./_data/dacon_air/train.csv', 'train')
csv_to_parquet('./_data/dacon_air/test.csv', 'test')

train = pd.read_parquet('./train.parquet')
test = pd.read_parquet('./test.parquet')
sample_submission = pd.read_csv('./_data/dacon_air/sample_submission.csv', index_col = 0)

# Replace variables with missing values except for the label (Delay) with the most frequent values of the training data
# 컬럼의 누락된 값은 훈련 데이터에서 해당 컬럼의 최빈값으로 대체됩니다.
NaN_col = ['Origin_State','Destination_State','Airline','Estimated_Departure_Time', 'Estimated_Arrival_Time','Carrier_Code(IATA)','Carrier_ID(DOT)']

for col in NaN_col:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    
    if col in test.columns:
        test[col] = test[col].fillna(mode)
print('Done.')

# Quantify qualitative variables
# 정성적 변수는 LabelEncoder를 사용하여 숫자로 인코딩됩니다.
qual_col = ['Origin_Airport', 'Origin_State', 'Destination_Airport', 'Destination_State', 'Airline', 'Carrier_Code(IATA)', 'Tail_Number']

for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train[i])
    train[i] = le.transform(train[i])
    
    for label in np.unique(test[i]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    test[i] = le.transform(test[i])
print('Done.')

# Remove unlabeled data
# 훈련 세트에서 레이블이 지정되지 않은 데이터가 제거되고 숫자 레이블 열이 추가됩니다.
train = train.dropna()

column_number = {}
for i, column in enumerate(sample_submission.columns):
    column_number[column] = i
    
def to_number(x, dic):
    return dic[x]

train.loc[:, 'Delay_num'] = train['Delay'].apply(lambda x: to_number(x, column_number))
print('Done.')

train_x = train.drop(columns=['ID', 'Delay', 'Delay_num'])
train_y = train['Delay_num']
test = test.drop(columns=['ID'])


# Cross-validation with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=337)

# Model and hyperparameter tuning using GridSearchCV

min_rmse=1

for k in range(10):
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=337, stratify=train_y)


    for i in scaler_list:
        scaler = i
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        test = scaler.transform(test)

        def objective(trial, x_train, y_train, x_test, y_test, min_rmse):
            param = {
                'iterations': trial.suggest_int('iterations', 800, 2000),
                'depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate',  0.001,0.1),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 0, 15),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 1),
                'bagging_temperature': trial.suggest_int('bagging_temperature', 0, 10),
                'random_strength': trial.suggest_int('random_strength', 0, 10),
                'border_count': trial.suggest_int('border_count', 64, 256),
                    }
            model = CatBoostClassifier(**param, verbose=0)
            valid_cv = KFold(n_splits = 5,
                shuffle = True)
            # for train_idx, valid_idx in valid_cv.split(x_train):
            
            #     train_x , test_x = x_train.iloc[train_idx], y_train.iloc[train_idx]
            #     valid_x , valid_y = x_train.iloc[valid_idx], y_train.iloc[valid_idx]
            
            # 모델 학습
            grid = GridSearchCV(model,
                param,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1)
            
            model.fit(x_train, y_train)
            # best_model = grid.best_estimator_
            val_y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, val_y_pred)
            f1 = f1_score(y_test, val_y_pred, average='weighted')
            precision = precision_score(y_test, val_y_pred, average='weighted')
            recall = recall_score(y_test, val_y_pred, average='weighted')
            
            print(f'Accuracy: {accuracy}')
            print(f'F1 Score: {f1}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            
            y_pred = model.predict_proba(test)
            # y_pred = np.round(y_pred, 5)

            submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
            submission.to_csv('./_data/dacon_air/SB4.csv', index=True)
            # if rmse < 0.3:
            #     submit_csv['Calories_Burned'] = model.predict(test_csv)
            #     date = datetime.datetime.now()
            #     date = date.strftime('%m%d_%H%M%S')
            #     submit_csv.to_csv(path_save + date + str(round(rmse, 5)) + '.csv')
            #     # if rmse < min_rmse:
            #     #     min_rmse = rmse
            #     #     submit_csv.to_csv(path_save_min + date + str(round(rmse, 5)) + '.csv')
            # return rmse
        opt = optuna.create_study(direction='minimize')
        opt.optimize(lambda trial: objective(trial, x_train, y_train, x_test, y_test, min_rmse), n_trials=10000)
        print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)























# 'n_estimators' : [100, 200, 300, 400, 500, 1000] 디폴트 100 / 1~inf / 정수
# 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3 / 0~1 /
# 'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~inf/ 정수
# 'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100] 디폴트 0 / 0~inf
# 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] 디폴트1 
# 'subsample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트1 / 0~1
# 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트1 / 0~1
# 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트1 / 0~1
# 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트0 / 0~inf / L1 절대값 가중치 규제
# 'reg_lambda': [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트0 / 0~inf / L2 제곱 가중치 규제



# param_grid = {'n_estimators' : [5,100],
#     'learning_rate': [0.00001,0.000001],
#     'max_depth': [5],
    
# }

# grid = GridSearchCV(model,
#                     param_grid,
#                     cv=cv,
#                     scoring='accuracy',
#                     n_jobs=-1,
#                     verbose=1)

# grid.fit(train_x, train_y)

# best_model = grid.best_estimator_

# # Model evaluation
# val_y_pred = best_model.predict(val_x)
# accuracy = accuracy_score(val_y, val_y_pred)
# f1 = f1_score(val_y, val_y_pred, average='weighted')
# precision = precision_score(val_y, val_y_pred, average='weighted')
# recall = recall_score(val_y, val_y_pred, average='weighted')

# print(f'Accuracy: {accuracy}')
# print(f'F1 Score: {f1}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')

# # 하이퍼파라미터 튜닝 결과를 바탕으로 최적의 모델을 선택하고 테스트 세트의 목표 변수를 예측하는 데 사용합니다.
# # Model prediction
# y_pred = best_model.predict_proba(test_x)
# y_pred = np.round(y_pred, 5)

# submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
# submission.to_csv('./_data/dacon_air/SB4.csv', index=True)

# print(best_model)
