import pandas as pd
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, HalvingRandomSearchCV, HalvingGridSearchCV
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.utils import all_estimators
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import optuna
import datetime
import warnings
warnings.filterwarnings('ignore')

scaler_list = [MinMaxScaler(), MaxAbsScaler(), StandardScaler(), RobustScaler()]
model_list = [RandomForestRegressor(), DecisionTreeRegressor()]

param_r = [
    {'n_estimators':[100, 200, 500], 'max_depth':[10,20,50], 'min_samples_leaf':[5,10,15],'min_samples_split':[5,10]}, 
]

param_d = [
    {'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 'splitter':['best', 'random'], 'max_depth':[10,20,50],'min_samples_split':[5,10]}
]

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

# n_splits = 10
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
# regressor = all_estimators(type_filter='regressor')

path = './_data/open/'
path_save = './_save/open/'
path_save_min = './_save/open/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0).drop(['Weight_Status'], axis=1)
test_csv = pd.read_csv(path + 'test.csv', index_col=0).drop(['Weight_Status'], axis=1)
submit_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv['Calories_Burned']

x['Height(Feet)'] = 12*x['Height(Feet)']+x['Height(Remainder_Inches)']
x['Height(Remainder_Inches)'] = 703*x['Weight(lb)']/x['Height(Feet)']**2

test_csv['Height(Feet)'] = 12*test_csv['Height(Feet)']+test_csv['Height(Remainder_Inches)']
test_csv['Height(Remainder_Inches)'] = 703*test_csv['Weight(lb)']/test_csv['Height(Feet)']**2

le = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
test_csv['Gender'] = le.transform(test_csv['Gender'])

min_rmse = 1

for k in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=337)

    for i in scaler_list:
        scaler = i
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        test_csv = scaler.transform(test_csv)
        def objective(trial, x_train, y_train, x_test, y_test, min_rmse):
            alpha = trial.suggest_loguniform('alpha', 0.0001, 1)
            n_restarts_optimizer  = trial.suggest_int('n_restarts_optimizer', 3, 10)
            optimizer = trial.suggest_categorical('optimizer', ['fmin_l_bfgs_b', 'Powell', 'CG'])

            model = GaussianProcessRegressor(
                alpha=alpha,
                n_restarts_optimizer=n_restarts_optimizer,
                optimizer=optimizer,
            )
            
            model.fit(x_train, y_train)
            
            print('GPR result : ', model.score(x_test, y_test))
            
            y_pred = model.predict(x_test)
            rmse = RMSE(y_test, y_pred)
            print('GPR RMSE : ', rmse)
            if rmse < 1:
                submit_csv['Calories_Burned'] = model.predict(test_csv)
                date = datetime.datetime.now()
                date = date.strftime('%m%d_%H%M%S')
                submit_csv.to_csv(path_save + str(round(rmse, 5)) + '.csv')
                if rmse < min_rmse:
                    min_rmse = rmse
                    submit_csv.to_csv(path_save + str(round(rmse, 5)) + '.csv')
            return rmse
        opt = optuna.create_study(direction='minimize')
        opt.optimize(lambda trial: objective(trial, x_train, y_train, x_test, y_test, min_rmse), n_trials=100)
        print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)
        

# model = Sequential()
# model.add(Dense(32, input_shape=(x.shape[1],)))
# model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(32, activation='selu'))
# model.add(Dropout(0.1))
# model.add(Dense(32, activation='swish'))
# model.add(Dense(32, activation='swish'))
# model.add(Dense(1, activation='swish'))

# model.compile(loss='mae', optimizer='adam')
# es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)
# model.fit(x_train, y_train, epochs=1000, validation_split=0.2, batch_size=3, callbacks=[es])