from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=335,
    train_size=0.9)

# scaler=MinMaxScaler()
scaler=MaxAbsScaler()
# scaler=StandardScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성


# model=Sequential()
# model.add(Dense(10, input_dim=10))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))

input1 = Input(shape=(10,))
Dense1 = Dense(10)(input1)
Dense2 = Dense(100, activation='relu')(Dense1)
Dense3 = Dense(200, activation='relu')(Dense2)
Dense4 = Dense(100, activation='relu')(Dense3)
Dense5 = Dense(10, activation='relu')(Dense4)
output1 = Dense(1)(Dense5)

model = Model(inputs=input1, outputs=output1)



#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss',
                   patience=50,
                   mode='min',
                   verbose=1,
                   restore_best_weights=True)

hist = model.fit(x_train,y_train, epochs=1000, batch_size=10, validation_split=0.2,callbacks=[es])

print(hist.history)



#4. 평가, 예측

loss= model.evaluate(x_test, y_test)
print("loss :", loss)

y_predict=model.predict(x_test)

r2=r2_score(y_test , y_predict)

print("r2 :", r2)




# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9, 6))
# plt.title('디아벳스')
# plt.plot(hist.history['loss'], c='red', marker='.', label='로스')
# plt.plot(hist.history['val_loss'], c='blue', marker='.', label='발_로스')
# plt.grid()  #격자
# plt.legend()  #선에 대한 주석
# plt.show()

# loss : 2619.177978515625
# r2 : 0.6809021467058503

# loss : 2664.231689453125
# r2 : 0.6754132455800013


# loss : 2294.4931640625
# r2 : 0.7204589368666832    스케일러 적용