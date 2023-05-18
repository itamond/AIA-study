#ReduceLRonPlateau : 개선이 없으면 Learningrate 반으로 줄인다.


import numpy as np

from sklearn.datasets import fetch_california_housing, fetch_covtype
from sklearn.model_selection import train_test_split as tts


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target



x_train, x_test, y_train, y_test = tts(x,y,
                                       train_size=0.8,
                                       shuffle=True,
                                       random_state=337,
                                       )

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=8))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate,)
model.compile(loss='mse', optimizer=optimizer)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor = 'val_loss',
                   patience = 20, mode = 'min',
                   verbose = 1)

rlr = ReduceLROnPlateau(monitor = 'val_loss',
                        patience = 2,
                        mode='auto',
                        verbose = 1,
                        factor=0.5)

model.fit(x_train, y_train,
          epochs = 200, 
          batch_size = 32, 
          validation_split=0.2, 
          verbose=1, 
          callbacks=[es, rlr])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)

print("loss :", results)


# loss : 0.650985836982727