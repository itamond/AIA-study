# 패치 코브타입 데이터로 모델링하기.



import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_covtype

#1. 데이터

a = fetch_covtype()
x = a.data
y = a['target']

# print(np.unique(y, return_counts=True))

y = to_categorical(y)
y = np.delete(y, 0, axis=1)

x_train,x_test,y_train,y_test = train_test_split(
    x, y,
    train_size= 0.8,
    random_state=33,
    stratify=y,
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape, x_test.shape)     #(464809, 54) (116203, 54)
print(y_train.shape, y_test.shape)     #(464809, 7) (116203, 7)

#2. 모델 구성

model=Sequential()
model.add(Dense(54,activation='relu', input_dim = 54))
model.add(Dense(108,activation='relu'))
model.add(Dense(216,activation='relu'))
model.add(Dense(108,activation='relu'))
model.add(Dense(54,activation='relu'))
model.add(Dense(7,activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam',
              metrics = ['acc'],
              )

es = EarlyStopping(monitor='val_acc',
                   mode='max',
                   patience=150,
                   verbose=1,
                   restore_best_weights=True,
                   )

hist = model.fit(x_train,y_train,
                 epochs=2000,
                 callbacks=[es],
                 validation_split=0.2,
                 batch_size=3200,
                 verbose=1,
                 )


#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result :', result)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)

print('acc :', acc)