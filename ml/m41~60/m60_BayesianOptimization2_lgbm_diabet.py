from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

from bayes_opt import BayesianOptimization
bayesian_params = {
    'max_depth' : (3, 16),
    'num_leaves' : (24, 64),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 500),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50)    
}

#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델

def L_function(max_depth, num_leaves, min_child_samples, min_child_weight,
               subsample, colsample_bytree,max_bin, reg_lambda, reg_alpha) :
    params = {
        'n_estimators' : 500,
        'learning_rate' : 0.15,
        'max_depth' : int(max_depth),
        'num_leaves' : int(num_leaves),
        'min_child_samples' : int(min_child_samples),
        'min_child_weight' : int(min_child_weight),
        'subsample' : float(subsample),
        'colsample_bytree' : float(colsample_bytree),
        'max_bin' : int(max_bin),
        'reg_lambda' : float(reg_lambda),
        'reg_alpha' : float(reg_alpha)
    }
    
    model = LGBMRegressor(**params)
    
    model.fit(x_train, y_train,
              eval_set = [(x_train,y_train),(x_test,y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50,
              )
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    
    return results

optimizer = BayesianOptimization(
    f = L_function,
    pbounds = bayesian_params,
    random_state=337
)

optimizer.maximize(init_points = 5,
                   n_iter = 500)

print(optimizer.max)

# |  target   | colsam... |  max_bin  | max_depth | min_ch... | min_ch... | num_le... | reg_alpha | reg_la... | subsample |
# | 0.6321    | 0.8042    | 463.5     | 16.0      | 82.96     | 31.19     | 64.0      | 19.95     | 8.815     | 0.5       |
# | 0.6371    | 0.5       | 327.8     | 14.0      | 10.0      | 32.73     | 64.0      | 21.63     | 10.0      | 1.0       |

# {'target': 0.6370850340211294, 'params': {'colsample_bytree': 0.5, 
# 'max_bin': 327.75338206756277, 'max_depth': 14.0, 'min_child_samples': 10.0, 
# 'min_child_weight': 32.73146367151851, 'num_leaves': 64.0, 'reg_alpha': 21.63233082725312, 'reg_lambda': 10.0, 'subsample': 1.0}}