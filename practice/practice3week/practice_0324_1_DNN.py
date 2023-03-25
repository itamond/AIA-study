from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Input, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, GRU, LSTM, SimpleRNN
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import datetime
import time

#1. 데이터

dataset = fetch_covtype()

x= dataset.data
y= dataset.target

# print(dataset.DESCR)
# print(np.unique(y, return_counts=True))
# print(x.shape, y.shape)   #(581012, 54) (581012,)
y = pd.get_dummies(y)
# print(y.shape)   #(581012, 7)

print(np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=335,
                                                    stratify=y,
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x_train),np.max(x_train))
print(x_train.shape)    #(464809, 54)
print(y_train.shape)    #(464809, 7)




#2. 모델구성

input1 = Input(shape = (54,))
dense1 = Dense(50,activation='relu')(input1)
dense2 = Dense(30,activation='relu')(dense1)
drop1 = Dropout(0.5)(dense2)
dense3 = Dense(10, activation='relu')(drop1)
output1 = Dense(7, activation='softmax')(dense3)

model = Model(inputs=input1, outputs=output1)

stt = time.time()

#3. 컴파일, 훈련
# mcp = ModelCheckpoint(monitor='val_acc',
#                       save_best_only=True,
#                       verbose=1,
#                       )

es = EarlyStopping(monitor = 'val_acc',
                   patience=30,
                   verbose=1,
                   restore_best_weights=True,
                   )

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam', 
              metrics='acc')

hist = model.fit(x_train, y_train,
                 epochs=3000,
                 batch_size=256,
                 callbacks = [es],
                 validation_split=0.2)

ett = time.time()

#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('loss :', result[0])
print('acc :', result[1])

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print('acc :', acc)