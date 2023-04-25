from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.experimental import enable_iterative_imputer 
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer #결측치에 대한 책임을 돌린다
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/kaggle_house/'
path_save = './_save/kaggle_house/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.4 라벨인코딩( object 에서 )
le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype=='object':
        train_csv[i] = le.fit_transform(train_csv[i])
        test_csv[i] = le.fit_transform(test_csv[i])
        
        
x = train_csv.drop(['SalePrice','LotFrontage'], axis=1)
# x = train_csv.drop(['SalePrice'], axis=1)
y = train_csv['SalePrice']
test_csv = test_csv.drop(['LotFrontage'], axis=1)

imputer = IterativeImputer(estimator=XGBRegressor())
# imputer = IterativeImputer(estimator=DecisionTreeRegressor())
x = imputer.fit_transform(x)
test_csv = imputer.transform(test_csv)



# 1.5 x, y 분리


print(x.shape)

# 1.6 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=640874, shuffle=True)

# 1.7 Scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# 2. 모델구성
# model = Sequential()
# model.add(Dense(32, input_dim=8))
# model.add(Dense(64))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(8))
# model.add(Dense(1))

input1 = Input(shape=(78,))
dense1 = Dense(64, activation='relu')(input1) 
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(32, activation='relu')(drop1)  
drop2 = Dropout(0.2)(dense2)                                                          
dense3 = Dense(64,activation='relu')(drop2)     
drop3 = Dropout(0.3)(dense3)                                                          
output1 = Dense(1)(drop3)
model = Model(inputs=input1, outputs=output1) 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min',
                   restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=5000, 
                 batch_size=128, verbose=1, validation_split=0.1, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

#'mse'->rmse로 변경
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)



# r2 :  0.6926327792464559
# RMSE :  49908.933301794204

#########보간 적용##########
# loss :  795607616.0
# r2 :  0.8696091007958218
# RMSE :  28206.51739039371