# 와인 데이터로 모델 구성하기


import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

#1. 데이터

datasets = load_wine()
x = datasets.data   #(178, 13)
y = datasets['target']
# print(datasets.DESCR)
# print(np.unique(y, return_counts=True))

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split (
    x, y,
    train_size=0.8,
    random_state=33,
    stratify=y,    
)

# print(x_train.shape, x_test.shape)    #(142, 13) (36, 13)
# print(y_train.shape, y_test.shape)    #(142, 3) (36, 3)



#2. 모델 구성
model=Sequential()
model.add(Dense(20, activation='relu', input_dim=13))
model.add(Dense(30,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(3,activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc',
                   mode='auto',
                   patience=100,
                   verbose=1)

hist = model.fit(x_train,y_train,
                 epochs=3000,
                 batch_size=4,
                 validation_split=0.2,
                 verbose=1,
                 callbacks=[es])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result :', result)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict)
print('acc :', acc)



#모델링

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.title('개쩌는 와인 데이터')
plt.plot(hist.history['val_loss'], c='red', marker='.', label='발로쓰')
plt.plot(hist.history['loss'], c='blue', marker='.', label='로쓰')
plt.grid()
plt.legend()
plt.show()