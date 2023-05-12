from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor,LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris, load_digits, load_wine, fetch_covtype, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import time

path = './_data/dacon_wine/'
path_save = './_save/dacon/wine/'
train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

# Remove rows with single class label
single_class_label = train_csv['quality'].nunique() == 1
if single_class_label:
    train_csv = train_csv[train_csv['quality'] != train_csv['quality'].unique()[0]]

# Label encode 'type'
le = LabelEncoder()
train_csv['type'] = le.fit_transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

# Split data
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']-3
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state = 337, shuffle = True,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import time

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from catboost import CatBoostClassifier


search_space = {
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),
    'depth': hp.quniform('depth', 3, 16, 1),
    'one_hot_max_size': hp.quniform('one_hot_max_size', 24, 64, 1),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 200, 1),
    'bagging_temperature': hp.uniform('bagging_temperature', 0.5, 1),
    'random_strength': hp.uniform('random_strength', 0.5, 1),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.001, 10)
}


def lgb_hamsu(search_space):
    params = {
        'iterations': 10,
        'learning_rate': search_space['learning_rate'],
        'depth': int(search_space['depth']),
        'l2_leaf_reg': search_space['l2_leaf_reg'],
        'bagging_temperature': search_space['bagging_temperature'],
        'random_strength': search_space['random_strength'],
        'one_hot_max_size': int(search_space['one_hot_max_size']),
        'min_data_in_leaf': int(search_space['min_data_in_leaf']),
        'task_type': 'CPU',
        'logging_level': 'Silent'
    }
    
    
    model = CatBoostClassifier(**params)

    model.fit(
        x_train, y_train,
        eval_set=(x_test, y_test),
        verbose=0,
        early_stopping_rounds=50
    )

    y_predict = model.predict(x_test)
    result = accuracy_score(y_test, y_predict)

    return result


trial_val = Trials()

best = fmin(
    fn=lgb_hamsu,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
)
print('best', best)


results_val = pd.DataFrame(trial_val.vals)
results_df = pd.DataFrame(trial_val.results)

results = pd.concat([results_df, results_val], axis=1)
results = results.drop(['status'], axis=1)


min_loss_idx = results['loss'].idxmin()
min_loss_row = results.loc[min_loss_idx]
print(results)
print(results.min())


# loss                    0.933333
# bagging_temperature     0.503025
# depth                   3.000000
# l2_leaf_reg             0.151843
# learning_rate           0.003265
# min_data_in_leaf       13.000000
# one_hot_max_size       24.000000
# random_strength         0.500350
# dtype: float64