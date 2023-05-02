import os
import numpy as np
import pandas as pd
from haversine import haversine
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

imputer = IterativeImputer(XGBRegressor())
le = OrdinalEncoder()

from preprocess import load_aws_and_pm
awsmap,pmmap=load_aws_and_pm()

from preprocess import load_distance_of_pm_to_aws
distance_of_pm_to_aws=load_distance_of_pm_to_aws(awsmap,pmmap)

from preprocess import min_dist_from_pm
result,min_i=min_dist_from_pm(distance_of_pm_to_aws,pmmap)
distance_of_pm_to_aws=distance_of_pm_to_aws.values
result=result.values
print(result)


########################################################## 2부 
train_pm_path = './_data/pm2.5/TRAIN/'
train_aws_path = './_data/pm2.5/TRAIN_AWS/'
test_pm_path = './_data/pm2.5/TEST_INPUT/'
test_aws_path = './_data/pm2.5/TEST_AWS/'

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

# print(train_pm.shape)

for i in range(train_pm.shape[0]):
    train_pm[i, :, 3] = imputer.fit_transform(train_pm[i, :, 3].reshape(-1, 1)).reshape(-1,)
# print(pd.DataFrame(train_pm.reshape(-1,4)).isna().sum())

for j in range(train_aws.shape[2]-3):
    for i in range(train_aws.shape[0]):
        train_aws[i, :, j+3] = imputer.fit_transform(train_aws[i, :, j+3].reshape(-1, 1)).reshape(-1,)
# print(pd.DataFrame(train_aws.reshape(-1,8)).isna().sum())

train_pm = train_pm.reshape(-1, 4)[:, 2:]
test_pm =  test_pm.reshape(-1, 4)[:, 2:]
train_aws =  train_aws.reshape(-1, 8)[:, 2:]
test_aws =  test_aws.reshape(30, -1, 8)[:, :, 2:]

# print(train_pm.shape)

def label(x):
    label_dict = {}
    labels = []
    for i in x:
        if i not in label_dict:
            label_dict[i] = len(label_dict)
        labels.append(label_dict[i])
    return labels


train_pm[:, 0] = label(train_pm[:, 0])
train_aws[:, 0] = label(train_aws[:, 0])
test_pm[:, 0] = label(test_pm[:, 0])

train_pm = train_pm.reshape(17, -1, 2)
test_pm = test_pm.reshape(17, -1, 2)
train_aws = train_aws.reshape(30, -1, 6)

for i in range(test_aws.shape[0]):
    test_aws[i, :, 0] = pd.DataFrame(test_aws[i, :, 0]).ffill().values.reshape(-1,)
test_aws = test_aws.reshape(-1, 6)
test_aws[:, 0] = label(test_aws[:, 0])
test_aws = test_aws.reshape(30, -1, 6)

# print(train_pm.shape)
# print(train_aws.shape)
# print(test_pm.shape)
# print(test_aws.shape)





# train_aws = train_aws.astype(np.float32)

train_pm_aws=[]
for i in range(train_pm.shape[0]):
    train_pm_aws.append(train_aws[min_i[i, 0], :, 1:]*result[0, 0] + train_aws[min_i[i, 1], :, 1:]*result[0, 1] + train_aws[min_i[i, 2], :, 1:]*result[0, 2])

# print(train_pm_aws)
train_pm_aws = np.array(train_pm_aws)









train_data = np.concatenate([train_pm, train_pm_aws], axis=2)
print(train_data)
print(train_data.shape)

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
# print(x)
print(x.shape)

y = []
for i in range(train_data.shape[0]):
    y.append(train_data[i, timesteps:, 1].reshape(train_data.shape[1]-timesteps,))

y = np.array(y).reshape(-1,)
# print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)


scaler = MinMaxScaler()
print(x_train.shape)
print(x_test.shape)

x_train= x_train.reshape(417142,70)
x_test= x_test.reshape(178776,70)
x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)

x_train=x_train.reshape(417142,10,7).astype(np.float32)
x_test=x_test.reshape(178776,10,7).astype(np.float32)
y_train=y_train.astype(np.float32)
y_test=y_test.astype(np.float32)

# train_aws = train_aws.astype(np.float32)
# train_aws = train_aws.astype(np.float32)



model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, 7)))
model.add(Dense(16))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, batch_size=128, epochs=5)

result = model.evaluate(x_test, y_test)

print('결과 :', result)

