#따릉이 데이터를 이용하여 es va plt scaler 넣어 모델링 해~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping


#1. 데이터

path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

# print(train_set.columns)

#Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
    #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
    #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
    
train_set = train_set.dropna()

x= train_set.drop(['count'], axis=1)
y= train_set['count']

# print(x.shape, y.shape)   #(1328, 9) (1328,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=59111,
    # stratify=y,
    # verbose=1,
    shuffle=True,
)

ohe = MaxAbsScaler()
ohe.fit(x_train) # ohe fit 범위 지정
x_train = ohe.transform(x_train) 
x_test = ohe.transform(x_test)
test_set = ohe.transform(test_set)
print(np.min(x_test), np.max(x_test))


#2. 모델 구성

model = Sequential()
model.add(Dense(20, input_dim=9))
model.add(Dense(16, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

es = EarlyStopping(monitor='val_mae',
                   mode='min',
                   patience=200,
                   verbose=1,
                   restore_best_weights=True,
                   )

hist = model.fit(x_train,y_train,
                 epochs = 5000,
                 validation_split=0.2,
                 verbose=1,
                 batch_size = 64,
                 callbacks=[es],
                 )

#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result :', result)
y_predict= model.predict(x_test)
r2 = r2_score(y_test, y_predict)

def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_test, y_predict)
print('rmse :', rmse)


submit = model.predict(test_set)


submission['count']=submit
submission.to_csv(path_save+'submission_dda_scaler_MAS.csv')


# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9,6))
# plt.title('따릉이')
# plt.plot(hist.history['val_loss'], c='red', marker='.', label='발_로쓰')
# plt.plot(hist.history['loss'], c='blue', marker='.', label='로쓰')
# plt.grid()
# plt.legend()
# plt.show()






#배치 사이즈 32로 테스트하자.