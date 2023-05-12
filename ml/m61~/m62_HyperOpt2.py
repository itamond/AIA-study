from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')
import time


def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state = 337, shuffle = True,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

search_space = {
    'learning_rate' : hp.uniform('learning_rate', 0.001, 0.2), #q유니폼은 정수형태, 일반 유니폼은 정규분포의 실수 형태이다.
    'max_depth' : hp.quniform('max_depth', 3, 16, 1),
    'num_leaves' : hp.quniform('num_leaves', 24, 64, 1), 
    'min_child_samples' : hp.quniform('min_child_samples',10, 200, 1),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 50, 1),
    'subsample' : hp.uniform('subsample', 0.5, 1),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
    # 'max_bin' : hp.quniform('max_bin', 2, 500, 10),
    'reg_lambda' : hp.uniform('reg_lambda', 0.001, 10),
    'reg_alpha' : hp.uniform('reg_alpha', 0.01, 50)
}

# hp.quniform(label, low, high, q) : 최소부터 최대까지 q 간격
# hp.uniform(label, low, high) : 최소부터 최대까지
# hp.randint(label,upper) : 0부터 최대값 upper까지 random한 정수값
# hp.loguniform(label, low, high) : exp(uniform(low, high)) #엄청 큰 값을 쓸때 사용한다... 잘 사용안하는데 큰 값이면 사용함. exp는 지수변환 하는 함수




# print(search_space)
#최대값을 찾는 베이시안 옵티마이져이기 때문에, r2 스코어를 써도 되지만 rmse같은 로스 펑션에 마이너스를 붙여서 사용해도 좋다.

def lgb_hamsu(search_space):
    params = {
        'n_estimators' : 1000,
        'learning_rate' : search_space['learning_rate'],
        'max_depth' : int(search_space['max_depth']), #무조건 정수형
        'num_leaves' : int(search_space['num_leaves']), #무조건 정수형
        'min_child_samples' : int(search_space['min_child_samples']), #무조건 정수형
        'min_child_weight' : int(search_space['min_child_weight']), #무조건 정수형
        'subsample' : search_space['subsample'], #0에서 1사이의 값이다. min,1을 사용해서 1보다 작은 수만 뽑음. max,0을 사용하여 0보다 큰 수만 뽑음
        'colsample_bytree' : search_space['colsample_bytree'], 
        # 'max_bin' :int(search_space['max_bin']), #무조건 10 이상만 나와야한다. 따라서 Max ,10을 사용하여 10과 비교하여 더 높은값을 뽑게한다.
        'reg_lambda' : search_space['reg_lambda'],   # 0과 비교하여 더 높은 값을 뽑는 함수. 따라서 양수만 나오게 된다.
        'reg_alpha' : search_space['reg_alpha']
    }
    
    model = LGBMRegressor(**params)
    
    model.fit(x_train,y_train,
              eval_set = [(x_train,y_train),(x_test,y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    result = rmse(y_test, y_predict)
    
    return result

trial_val = Trials()  #hist

best = fmin(
    fn = lgb_hamsu,
    space = search_space,
    algo = tpe.suggest,  #묻지도 따지지도 말고 이거 쓰삼
    max_evals = 50,       #epochs
    trials = trial_val,
    rstate=np.random.default_rng(seed=10)
)
print('best', best)


results_val = pd.DataFrame(trial_val.vals)
results_df = pd.DataFrame(trial_val.results)

results = pd.concat([results_df, results_val], axis=1)
results = results.drop(['status'],axis=1)


min_loss_idx = results['loss'].idxmin()
min_loss_row = results.loc[min_loss_idx]
# print(results)
print(results.min())
# print(min_loss_row)

# best
# loss                  0.192716
# colsample_bytree      0.510664
# learning_rate         0.178106
# max_depth             7.000000
# min_child_samples    44.000000
# min_child_weight     45.000000
# num_leaves           43.000000
# reg_alpha             0.196900
# reg_lambda            5.808490
# subsample             0.590541