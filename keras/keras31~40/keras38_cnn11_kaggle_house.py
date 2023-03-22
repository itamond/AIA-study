import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import null
from tensorflow.python.keras.models import Sequential,load_model, Model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten, Input
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import time
import datetime

#1. 데이터
path = './_data/kaggle_house/' 
path_save = './_save/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)
drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
test_set.drop(drop_cols, axis = 1, inplace =True)
submission = pd.read_csv(path + 'sample_submission.csv',
                       index_col=0)
print(train_set)

print(train_set.shape) 
print(type(train_set))
train_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond',
                'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])

print(type(train_set))

# print(test_set)
# print(train_set.shape) #(1460,76)
# print(test_set.shape) #(1459, 75)

# print(train_set.columns)
# print(train_set.info()) 
# print(train_set.describe()) 

# print(train_set.isnull().sum())
train_set = train_set.fillna(train_set.median())
# print(train_set.isnull().sum())
# print(train_set.shape)
test_set = test_set.fillna(test_set.median())
print(type(train_set))

x = train_set.drop(['SalePrice'],axis=1) 
# print(x.columns)
# print(x.shape) #(1460, 75)
print(type(x))

y = train_set['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.90, shuffle = True, random_state = 687)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

print(x_train.shape, x_test.shape)   #(1314, 75) (146, 75)
x_train = np.array(x_train)

x_train = x_train.reshape(-1, 15,5,1)
x_test = np.array(x_test)
x_test = x_test.reshape(-1, 15,5,1)

date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
filepath = './_save/MCP/kaggle_house/'
filename = '{epoch:04d}-{val_loss:.2f}.hdf5'


start_time = time.time()
#2. 모델구성


input1 = Input(shape=(15, 5, 1))
conv1 = Conv2D(20, (2,2),
               padding = 'same',
               activation='relu')(input1)
conv2 = Conv2D(20, (2,2),
               padding = 'same',
               activation='relu')(conv1)
# mp1 = MaxPooling2D()
# pooling1 = mp1(conv2)
flat1 = Flatten()(conv2)
dense1 = Dense(10, activation='relu')(flat1)
output1 = Dense(1)(dense1)

model=Model(inputs=input1, outputs=output1)



# #3. 컴파일,훈련

es = earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=100,
                              mode='auto', 
                              verbose=1,
                              restore_best_weights=True)

model.compile(loss='mae', optimizer='adam', metrics=['mae'])

mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      filepath=''.join([filepath+'k36_1_'+date+'_'+filename])
                      )

hist = model.fit(x_train,y_train,
                 epochs=5000,
                 batch_size=256, 
                 validation_split=0.2,
                 callbacks = [es],
                 verbose=1)


end_time = time.time()

            
#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
test_set = np.array(test_set)
test_set = test_set.reshape(-1, 15,5,1)

submit = model.predict(test_set)
submission['SalePrice'] = submit
submission.to_csv('./_save/house_price/subtest3'+date+'.csv')