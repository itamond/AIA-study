#사이킷런 와인 데이터를 이용한 애큐러시 모델 만들기

import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score


#1. 데이터

data_sets = load_wine()
x= data_sets.data
y= data_sets.target

# print(x.shape, y.shape)   #(178, 13) (178,)

y = pd.get_dummies(y)

# print(y.shape)

x_train, x_test, y_train, y_test =train_test_split(x, y,
                                                   train_size= 0.8,
                                                   stratify=y,
                                                   random_state=392,
                                                   )

#2. 모델구성

model=Sequential()
model.add(Dense(20, input_dim=13))
model.add(Dense(10,activation='relu'))
model.add(Dense(3,activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_acc',
                   patience=50,
                   mode='min',
                   verbose=1)

hist = model.fit(x_train,y_train,
                 epochs = 400,
                 batch_size=10,
                 validation_split=0.2,
                 callbacks=[es]
                 )

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result :', result)

y_predict= model.predict(x_test)
print(y_predict)
# acc = accuracy_score(y_test, y_predict)

print('acc :', acc)
