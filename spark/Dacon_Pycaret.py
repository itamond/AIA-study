from pycaret.classification import *
#파이캐럿 분류 회귀는 pycaret.regression


import random
import os
import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer, log_loss
from xgboost import XGBClassifier
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(805) # Fixed Seed


train = pd.read_csv('./_data/dacon_air/train.csv',index_col=0)
test = pd.read_csv('./_data/dacon_air/test.csv',index_col=0)
sample_submission = pd.read_csv('./_data/dacon_air/sample_submission.csv', index_col = 0)


# Replace variables with missing values except for the label (Delay) with the most frequent values of the training data
# 컬럼의 누락된 값은 훈련 데이터에서 해당 컬럼의 최빈값으로 대체됩니다.
NaN_col = ['Origin_State','Destination_State','Airline','Estimated_Departure_Time', 'Estimated_Arrival_Time','Carrier_Code(IATA)','Carrier_ID(DOT)']

for col in NaN_col:
    mode = train[col].mode()[0]
    # median = train[col].median()
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

# train.loc[:, 'Delay_num'] = train['Delay'].apply(lambda x: to_number(x, column_number))
# train_x = train.drop(columns=['Delay', 'Delay_num'],axis=1)
# train_y = train['Delay_num']

# train = train_x
clf = setup(data=train, target='Delay', train_size=0.8, use_gpu=True, normalize=True,)
add_metric('log_loss','Log_loss',log_loss,greater_is_better=True,target='pred_proba')
add_metric('ACC','ACC',accuracy_score,greater_is_better=True,target='pred_proba')

best2 = compare_models(fold=5, sort='log_loss', n_select=2, exclude=['svm','ridge'])

tuned_best2 = [tune_model(i, optimize='log_loss') for i in best2]


blend_best2 = blend_models(estimator_list=tuned_best2, fold=5, optimize='log_loss')

prep_pipe = get_config('prep_pipe')

final_model = finalize_model(blend_best2)
prep_pipe = get_config('prep_pipe')
prep_pipe.steps.append(['trained_model', final_model])
# pred = prep_pipe.predict_proba(test)

# final_model = finalize_model(blend_best2)
# pred = predict_model(final_model, data=test)
y_pred = prep_pipe.predict_proba(test)

# y_pred = np.round(y_pred, 5)

submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('./_data/dacon_air/SB4.csv', index=True)