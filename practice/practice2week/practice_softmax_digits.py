# 사이킷런 load_digits 데이터로 다중분류 모델 만들고
# acc 값 구하기.
# 다중 카테고리컬 소프트


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_digits

#1. 데이터

datasets = load_digits()
x=datasets.data
y=datasets['target']


# print(x.shape, y.shape)   (1797, 64) (1797,)

y = to_categorical(y)

# print(y.shape)   #(1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=335,
    stratify=y,
)

#2. 모델구성

model = Sequential()
model.add(Dense(128, input_dim=64))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer = 'adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_acc',
                   mode='auto',
                   restore_best_weights=True,
                   patience=50,
                   verbose=1)

hist = model.fit (x_train, y_train,
                  epochs =1000,
                  batch_size=32,
                  callbacks=[es],
                  verbose=1,
                  validation_split=0.2
                  )

#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result :', result)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis = 1)
y_test = np.argmax(y_test, axis =1)

acc = accuracy_score(y_test, y_predict)

print('acc :', acc)