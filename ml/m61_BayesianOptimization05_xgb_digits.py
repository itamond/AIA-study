from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor,LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris, load_digits, load_wine, fetch_covtype, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
import time

x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state = 337, shuffle = True,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

bayesian_params = {
    'learning_rate' : (0.001, 0.2),
    'max_depth' : (3, 16),
    # 'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 500), #무조건 10 이상
    'reg_lambda' : (-0.001, 10), #무조건 양수만 빼야한다
    'reg_alpha' : (0.01, 50)    
}

#최대값을 찾는 베이시안 옵티마이져이기 때문에, r2 스코어를 써도 되지만 rmse같은 로스 펑션에 마이너스를 붙여서 사용해도 좋다.

def lgb_hamsu(learning_rate,max_depth,
              min_child_weight,
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 1000,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)), #무조건 정수형
        # 'min_child_samples' : int(round(min_child_samples)), #무조건 정수형
        'min_child_weight' : int(round(min_child_weight)), #무조건 정수형
        'subsample' : max(min(subsample, 1),0), #0에서 1사이의 값이다. min,1을 사용해서 1보다 작은 수만 뽑음. max,0을 사용하여 0보다 큰 수만 뽑음
        'colsample_bytree' : float(colsample_bytree), 
        'max_bin' : max(int(round(max_bin)),10), #무조건 10 이상만 나와야한다. 따라서 Max ,10을 사용하여 10과 비교하여 더 높은값을 뽑게한다.
        'reg_lambda' : max(reg_lambda, 0),   # 0과 비교하여 더 높은 값을 뽑는 함수. 따라서 양수만 나오게 된다.
        'reg_alpha' : float(reg_alpha)
    }
    
    model = XGBClassifier(**params)
    
    model.fit(x_train,y_train,
              eval_set = [(x_train,y_train),(x_test,y_test)],
              eval_metric='merror',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    result = accuracy_score(y_test, y_predict)
    
    return result

optimizer = BayesianOptimization(f = lgb_hamsu,
                                 pbounds=bayesian_params, #사용할 파라미터 범위
                                 random_state=337,
                                 )

n_iter = 100
stt = time.time()
optimizer.maximize(init_points=5, n_iter = 100)  #init_points 초기 시작점 갯수
ett = time.time()

print(optimizer.max)
print(n_iter, '번 걸린 시간 :',round(ett-stt,2))

# |   iter    |  target   | colsam... | learni... |  max_bin  | max_depth | min_ch... | reg_alpha | reg_la... | subsample |
# | 2         | 0.8176    | 0.5819    | 0.1489    | 441.1     | 6.288     | 8.758     | 18.59     | 7.092     | 0.8917    |