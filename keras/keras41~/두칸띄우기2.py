import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, MaxPooling1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터

path = './_data/kaggle_jena/'

datasets = pd.read_csv(path+'jena_climate_2009_2016.csv', index_col=0)
x= datasets.drop(["T (degC)"], axis=1)
y= datasets["T (degC)"]

scaler = StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    shuffle=False,
                                                    )

x_test, x_pred, y_test, y_pred = train_test_split(x_test,y_test,
                                                  train_size=0.67,
                                                  shuffle=False,
                                                  )


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)


timesteps = 10

def split_x(datasets, timesteps) : 
    aaa = []
    for i in range(len(datasets)-timesteps-1):
        subset = datasets[i : (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

x_train = split_x(x_train, timesteps)
x_test = split_x(x_test, timesteps)
x_pred = split_x(x_pred, timesteps)
print(x_pred)

y_train = y_train[(timesteps+1) :]
y_test = y_test[(timesteps+1) :]
y_pred = y_pred[(timesteps+1) :]


print(x_train.shape)    #(294374, 10, 13)
model = Sequential()
model.add(LSTM(128, input_shape=(10,13)))
model.add(Dense(1,activation='relu'))


#3.컴파일, 훈련
es = EarlyStopping(monitor='val_loss',
                   patience=10,
                   restore_best_weights=True,
                   verbose=1)
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=5,
          batch_size=512,
          callbacks=[es],
          validation_split=0.2)


#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result :', result)

pred = model.predict(x_pred)

r2 = r2_score(y_pred, pred)
print('r2:',r2)
def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a, b))
rmse = RMSE(y_pred, pred)
print('rmse =', rmse)
# result : 1.013287901878357
# r2: 0.9853355923685364
# rmse = 0.9460746699089344

print('y_pred :', pred)