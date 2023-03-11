from sklearn.datasets import fetch_california_housing
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorflow.python.keras.callbacks import EarlyStopping


#1. 데이터


datasets= fetch_california_housing()
x = datasets.data
y = datasets.target



x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.9, random_state=12)


#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(12, activation='relu'))
model.add(Dense(27, activation='relu'))
model.add(Dense(38, activation='relu'))
model.add(Dense(42, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

#es -> EarlyStopping에 대한 정의

es = EarlyStopping(monitor='val_loss',     #발로스를 주시할거다
                   patience=50,             #50번 참아라
                   mode='min',               #최소값으로
                   verbose=1,                  #텍스트로 출력해라
                   restore_best_weights=True     #최고의 w값을 저장해라
                   )


hist = model.fit(x_train, y_train, epochs=1000, batch_size= 100,
          validation_split=0.2, callbacks=[es])

#4. 평가, 예측

loss = model.evaluate(x_test,y_test)
print("loss :", loss)


y_predict= model.predict(x_test)



r2=r2_score(y_test, y_predict)
print("r2 스코어 :", r2)


#loss : 0.6160849928855896
# r2 스코어 : 0.5407027611021253

#relu, es 첨가
# loss : 0.4392976760864258
# r2 스코어 : 0.6724994342047677

#plt 이용한 시각화

# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.title='캘리포니아'
# plt.grid()
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='로쓰', marker='.')
# plt.plot(hist.history['val_loss'], c='blue', label='발_로쓰', marker='.')
# plt.legend()
# plt.show()


# loss : 0.4312790036201477
# r2 스코어 : 0.6784774127262672