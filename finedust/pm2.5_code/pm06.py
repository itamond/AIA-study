import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from haversine import haversine
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from typing import Tuple

def load_aws_and_pm()->Tuple[pd.DataFrame,pd.DataFrame]:
    path='./_data/finedust'
    path_list=os.listdir(path)
    print(f'datafolder_list:{path_list}')

    meta='/'.join([path,path_list[1]])
    meta_list=os.listdir(meta)

    print(f'META_list:{meta_list}')
    awsmap=pd.read_csv('/'.join([meta,meta_list[0]]))
    awsmap=awsmap.drop(awsmap.columns[-1],axis=1)
    pmmap=pd.read_csv('/'.join([meta,meta_list[1]]))
    pmmap=pmmap.drop(pmmap.columns[-1],axis=1)
    return awsmap,pmmap

def load_distance_of_pm_to_aws(awsmap:pd.DataFrame,pmmap:pd.DataFrame)->pd.DataFrame:
    '''pm과 ams관측소 사이의 거리들을 프린트해준다'''
    a = []
    for i in range(pmmap.shape[0]):
        b=[]
        for j in range(awsmap.shape[0]):
            b.append(haversine((np.array(pmmap)[i, 1], np.array(pmmap)[i, 2]), (np.array(awsmap)[j, 1], np.array(awsmap)[j, 2])))
        a.append(b)    
    distance_of_pm_to_aws = pd.DataFrame(np.array(a),index=pmmap['Location'],columns=awsmap['Location'])
    return distance_of_pm_to_aws

def scaled_score(distance_of_pm_to_aws:pd.DataFrame,pmmap:pd.DataFrame,near:int=3)->Tuple[pd.DataFrame,np.ndarray]:
    '''pm으로부터 가까운 상위 near개의 환산점수'''
    min_i=[]
    min_v=[]
    for i in range(distance_of_pm_to_aws.shape[0]):
        min_i.append(np.argsort(distance_of_pm_to_aws.values[i,:])[:near])
        min_v.append(distance_of_pm_to_aws.values[i, min_i[i]])

    min_i = np.array(min_i)
    min_v = pd.DataFrame(np.array(min_v),index=distance_of_pm_to_aws.index)
    
    for i in range(pmmap.shape[0]):
        for j in range(near):
            min_v.values[i, j]=min_v.values[i, j]**2
            
    sum_min_v = np.sum(min_v, axis=1)

    recip=[]
    for i in range(pmmap.shape[0]):
        recip.append(sum_min_v[i]/min_v.values[i, :])
    recip = np.array(recip)
    recip_sum = np.sum(recip, axis=1)
    coef = 1/recip_sum

    result = []
    for i in range(pmmap.shape[0]):
        result.append(recip[i, :]*coef[i])
    result = pd.DataFrame(np.array(result),index=distance_of_pm_to_aws.index)
    return result,min_i
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     except RuntimeError as e:
#         print(e)        

imputer = IterativeImputer(XGBRegressor())

awsmap, pmmap = load_aws_and_pm()

label = LabelEncoder()

awsmap['Location'] = label.fit_transform(awsmap['Location'])
pmmap['Location'] = label.fit_transform(pmmap['Location'])

awsmap = awsmap.sort_values(by='Location')
pmmap = pmmap.sort_values(by='Location')

distance_of_pm_to_aws = load_distance_of_pm_to_aws(awsmap, pmmap)

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

y = []
for i in range(train_data.shape[0]):
    y.append(train_data[i, timesteps:, 1].reshape(train_data.shape[1]-timesteps,))

y = np.array(y).reshape(-1,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=False)
scaler = MinMaxScaler()

x_train = x_train.reshape(-1, 7)
x_test = x_test.reshape(-1, 7)

x_train[:, 2:], x_test[:, 2:] = scaler.fit_transform(x_train[:, 2:]), scaler.transform(x_test[:, 2:])

x_train=x_train.reshape(-1, timesteps, 7).astype(np.float32)
x_test=x_test.reshape(-1, timesteps, 7).astype(np.float32)
y_train=y_train.astype(np.float32)
y_test=y_test.astype(np.float32)

model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, 7)))
model.add(Dense(16))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, batch_size=128, epochs=10)

print(x_train.shape,y_train.shape)

test_pm = np.array(test_pm)
test_pm_aws = np.array(test_pm_aws)
print(pd.DataFrame(test_pm.reshape(-1,2)).isna().sum())
submission = pd.read_csv('./finedust/answer_sample.csv', index_col=0)
a=np.zeros(submission.shape[0])
k=0
for j in range(17):
    for i in range(test_pm.shape[1]):
        if np.isnan(test_pm[j, i, 1]):
            test_pm[j, i, 1] = model.predict(np.concatenate([test_pm[j, i-11:i-1, :], test_pm_aws[j, i-11:i-1, :]], axis=1).reshape(-1,timesteps,7).astype(np.float32))
            a[k]=test_pm[j, i, 1]
            k+=1
        print(f'변환 진행중{j}번 {np.round(100*i/test_pm.shape[1],1)}%')
print(pd.DataFrame(test_pm.reshape(test_pm.shape[1],-1)).isna().sum())
print(a[:20])

submission['PM2.5']=a
submission.to_csv('./finedust/Aiur_Submit_2.csv')