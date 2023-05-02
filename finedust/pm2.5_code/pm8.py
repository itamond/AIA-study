import os
import numpy as np
import pandas as pd
import time
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input,Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)        

imputer = IterativeImputer(LGBMRegressor())

from preprocess import load_aws_and_pm
awsmap, pmmap = load_aws_and_pm()

label = LabelEncoder()

awsmap['Location'] = label.fit_transform(awsmap['Location'])
pmmap['Location'] = label.fit_transform(pmmap['Location'])

awsmap = awsmap.sort_values(by='Location')
pmmap = pmmap.sort_values(by='Location')

from preprocess import load_distance_of_pm_to_aws
distance_of_pm_to_aws = load_distance_of_pm_to_aws(awsmap, pmmap)

from preprocess import scaled_score
result, min_i = scaled_score(distance_of_pm_to_aws, pmmap)
distance_of_pm_to_aws = distance_of_pm_to_aws.values
result = result.values

train_pm_path = './_data/finedust/TRAIN/'
train_aws_path = './_data/finedust/TRAIN_AWS/'
test_pm_path = './_data/finedust/TEST_INPUT/'
test_aws_path = './_data/finedust/TEST_AWS/'

def bring(path:str)->np.ndarray:
    file_list = os.listdir(path)
    data_list = []
    for file_name in file_list:
        if file_name.endswith(".csv"):
            file_path = os.path.join(path, file_name)
            data = pd.read_csv(file_path).values
            data_list.append(data)
    data_array = np.array(data_list)
    return data_array

train_pm = bring(train_pm_path)
train_aws = bring(train_aws_path)
test_pm = bring(test_pm_path)
test_aws = bring(test_aws_path)

for i in range(train_pm.shape[0]):
    train_pm[i, :, 3] = imputer.fit_transform(train_pm[i, :, 3].reshape(-1, 1)).reshape(-1,)
# print(pd.DataFrame(train_pm.reshape(-1,4)).isna().sum())

for j in range(train_aws.shape[2]-3):
    for i in range(train_aws.shape[0]):
        train_aws[i, :, j+3] = imputer.fit_transform(train_aws[i, :, j+3].reshape(-1, 1)).reshape(-1,)
# print(pd.DataFrame(train_aws.reshape(-1,8)).isna().sum())

for j in range(test_aws.shape[2]-3):
    for i in range(test_aws.shape[0]):
        test_aws[i, :, j+3] = imputer.fit_transform(test_aws[i, :, j+3].reshape(-1, 1)).reshape(-1,)

train_pm = train_pm.reshape(-1, 4)[:, 2:]
test_pm =  test_pm.reshape(-1, 4)[:, 2:]
train_aws =  train_aws.reshape(-1, 8)[:, 2:]
test_aws =  test_aws.reshape(30, -1, 8)[:, :, 2:]


for i in range(test_aws.shape[0]):
    test_aws[i, :, 0] = pd.DataFrame(test_aws[i, :, 0]).ffill().values.reshape(-1,)
test_aws = test_aws.reshape(-1, 6)

train_pm[:, 0] = label.fit_transform(train_pm[:, 0])
train_aws[:, 0] = label.fit_transform(train_aws[:, 0])
test_pm[:, 0] = label.fit_transform(test_pm[:, 0])
test_aws[:, 0] = label.fit_transform(test_aws[:, 0])

train_pm = train_pm.reshape(17, -1, 2)
test_pm = test_pm.reshape(17, -1, 2)
train_aws = train_aws.reshape(30, -1, 6)
test_aws = test_aws.reshape(30, -1, 6)

train_pm_aws = []
for i in range(train_pm.shape[0]):
    train_pm_aws.append(train_aws[min_i[i, 0], :, 1:]*result[0, 0] + train_aws[min_i[i, 1], :, 1:]*result[0, 1] + train_aws[min_i[i, 2], :, 1:]*result[0, 2])

train_pm_aws = np.array(train_pm_aws)
train_data = np.concatenate([train_pm, train_pm_aws], axis=2)

test_pm_aws = []
for i in range(test_pm.shape[0]):
    test_pm_aws.append(test_aws[min_i[i, 0], :, 1:]*result[0, 0] + test_aws[min_i[i, 1], :, 1:]*result[0, 1] + test_aws[min_i[i, 2], :, 1:]*result[0, 2])

test_pm_aws = np.array(test_pm_aws)

train_pm_reverse = np.flip(train_pm, axis=1)
train_pm_aws_reverse = np.flip(train_pm_aws, axis=1)
test_pm_reverse = np.flip(test_pm, axis=1)
test_pm_aws_reverse = np.flip(test_pm_aws, axis=1)

train_rev_data = np.concatenate([train_pm_reverse, train_pm_aws_reverse], axis=2)


def split_x(dt, ts):
    a = []
    for j in range(dt.shape[0]):
        b = []
        for i in range(dt.shape[1]-ts):
            c = dt[j, i:i+ts, :]
            b.append(c)
        a.append(b)
    return np.array(a)

timesteps = 10

x = split_x(train_data, timesteps).reshape(-1, timesteps, train_data.shape[2])
x_rev = split_x(train_rev_data, timesteps).reshape(-1, timesteps, train_rev_data.shape[2])

y = []
for i in range(train_data.shape[0]):
    y.append(train_data[i, timesteps:, 1].reshape(train_data.shape[1]-timesteps,))
y = np.array(y).reshape(-1,)

y_rev=[]
for i in range(train_rev_data.shape[0]):
    y_rev.append(train_rev_data[i, timesteps:, 1].reshape(train_rev_data.shape[1]-timesteps,))
y_rev = np.array(y_rev).reshape(-1,)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)
x_rev_train, x_rev_test, y_rev_train, y_rev_test = train_test_split(x_rev, y_rev, train_size=0.7, random_state=123, shuffle=True)

scaler = MinMaxScaler()

x_train = x_train.reshape(-1, 7)
x_test = x_test.reshape(-1, 7)
x_rev_train = x_rev_train.reshape(-1, 7)
x_rev_test = x_rev_test.reshape(-1, 7)

x_train[:, 2:], x_test[:, 2:] = scaler.fit_transform(x_train[:, 2:]), scaler.transform(x_test[:, 2:])
x_rev_train[:, 2:], x_rev_test[:, 2:] = scaler.fit_transform(x_rev_train[:, 2:]), scaler.transform(x_rev_test[:, 2:])

x_train=x_train.reshape(-1, timesteps, 7).astype(np.float32)
x_test=x_test.reshape(-1, timesteps, 7).astype(np.float32)
y_train=y_train.astype(np.float32)
y_test=y_test.astype(np.float32)

x_rev_train=x_rev_train.reshape(-1, timesteps, 7).astype(np.float32)
x_rev_test=x_rev_test.reshape(-1, timesteps, 7).astype(np.float32)
y_rev_train=y_rev_train.astype(np.float32)
y_rev_test=y_rev_test.astype(np.float32)

input1 = Input(shape=(timesteps,7))
lstm1 = LSTM(256, activation='relu', name='lstm1')(input1)
drop2 = Dropout(0.2)(lstm1)
dense1 = Dense(128, activation='relu', name='dense1')(drop2)
dense2 = Dense(64, activation='relu', name='dense2')(dense1)
dense3 = Dense(32, activation='relu', name='dense3')(dense2)
dense4 = Dense(16, activation='relu', name='dense4')(dense3)
output1 = Dense(1, name='output1')(dense4)

model1 = Model(inputs=input1, outputs=output1)
model2 = Model(inputs=input1, outputs=output1)

model1.compile(loss='mae', optimizer='adam')
model2.compile(loss='mae', optimizer='adam')

es = EarlyStopping(monitor='val_loss',
                   restore_best_weights=True,
                   patience=20
                   )
rl = ReduceLROnPlateau(monitor='val_loss',
                       patience=4,
                       )

stt = time.time()
model1.fit(x_train, y_train, batch_size=512, epochs=200,
          callbacks=[es,rl],
          validation_split=0.2)

model2.fit(x_rev_train, y_rev_train, batch_size=512, epochs=200,
          callbacks=[es,rl],
          validation_split=0.2)







# print(x_train.shape,y_train.shape)

test_pm = np.array(test_pm)
test_pm_aws = np.array(test_pm_aws)
# print(pd.DataFrame(test_pm.reshape(-1,2)).isna().sum())
submission = pd.read_csv('./_data/pm2.5/answer_sample.csv', index_col=0)


a=np.zeros(submission.shape[0])

# print(test_pm[0, 204+1-84:204+11-84, :])
# print(test_pm_aws[0, 204+1-84:204+11-84, :])
# print(np.concatenate([test_pm[0, 204+1-84:204+11-84, :],test_pm_aws[0, 204+1-84:204+11-84, :]],axis=1))
# print(np.concatenate([test_pm[0, 204+1-84:204+11-84, :],test_pm_aws[0, 204+1-84:204+11-84, :]],axis=1).shape)
l=[]
for j in range(17):
    for k in range(64):
        for i in range(120):
            if np.isnan(test_pm[j, 120*k+i, 1]) and i<84:
                test_pm[j, 120*k+i, 1] = model1.predict(np.concatenate([test_pm[j, 120*k+i-11:120*k+i-1, :], test_pm_aws[j, 120*k+i-11:120*k+i-1, :]], axis=1).reshape(-1,timesteps,7).astype(np.float32))
            elif i>=84:
                test_pm[j, 120*k+204-i, 1] = model2.predict(np.flip(np.concatenate([test_pm[j, 120*k+204-i:120*k+204-i+10, :], test_pm_aws[j, 120*k+204-i:120*k+204-i+10, :]], axis=1), axis=0).reshape(-1,timesteps,7).astype(np.float32))
            print(f'model1 변환 진행중{j}의 {k}의 {i}번')
        l.append(test_pm[j, 120*k+48:120*k+120, 1])

l = np.array(l).reshape(-1,)
print(l)
print(l.shape)


submission['PM2.5']=l
submission.to_csv('./_data/pm2.5/Aiur_Submit_3.csv')
ett = time.time()
print('걸린시간 :', np.round((ett-stt),2),'초')
model1.save("./_save/Airu_Submit.h5")



# Traceback (most recent call last):
#   File "C:\AIA\AIA-study\finedust\pm2.5_code\pm8.py", line 247, in <module>
#     submission['PM2.5']=np.round(l,3)
#   File "<__array_function__ internals>", line 5, in round_
#   File "C:\Users\Administrator\anaconda3\envs\tf274gpu\lib\site-packages\numpy\core\fromnumeric.py", line 3739, in round_
#     return around(a, decimals=decimals, out=out)
#   File "<__array_function__ internals>", line 5, in around
#   File "C:\Users\Administrator\anaconda3\envs\tf274gpu\lib\site-packages\numpy\core\fromnumeric.py", line 3314, in around
#     return _wrapfunc(a, 'round', decimals=decimals, out=out)
#   File "C:\Users\Administrator\anaconda3\envs\tf274gpu\lib\site-packages\numpy\core\fromnumeric.py", line 66, in _wrapfunc
#     return _wrapit(obj, method, *args, **kwds)
#   File "C:\Users\Administrator\anaconda3\envs\tf274gpu\lib\site-packages\numpy\core\fromnumeric.py", line 43, in _wrapit
#     result = getattr(asarray(obj), method)(*args, **kwds)
# TypeError: loop of ufunc does not support argument 0 of type numpy.ndarray which has no callable rint method