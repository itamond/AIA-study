# 사이킷런 load_digits 데이터 


import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import tensorflow as tf
from sklearn.datasets import load_digits
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
#1. 데이터
datasets=load_digits()
x=datasets['data']
y=datasets.target
print(x.shape,y.shape) 
# print(np.unique(y)) #[1 2 3 4 5 6 7]
print(np.unique(y,return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
                                    # dtype=int64)) 
# y=to_categorical(y)
# print(y) 
# print(x.shape,y.shape) #(581012, 54) (581012, 8) -> 투카테고리얼을 쓰면 

y=pd.get_dummies(y)
print(x.shape,y.shape) #(581012, 54) (581012, 7)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)



#2. 모델구성
model=Sequential()
model.add(Dense(120,input_dim=64))
model.add(Dense(200,activation='relu'))
model.add(Dense(160,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(80,activation='linear'))
model.add(Dense(10,activation='softmax'))

#3. 컴파일, 훈련
es=EarlyStopping(monitor='val_loss',patience=100,mode='auto',verbose=1,restore_best_weights=True) #
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

hist = model.fit(x_train, y_train,
                 epochs = 250,
                 validation_split=0.2,
                 batch_size=300,
                 verbose=1,
                 callbacks=[es])


#4. 평가, 예측

result=model.evaluate(x_test,y_test)
# print('loss:',result[0])
# print('accuracy:',result[1])
# print("===================================")
y_predict=model.predict(x_test)
y_predict=np.argmax(y_test,axis=1)
y_test=np.argmax(y_test,axis=1)
print(y_predict)
print(y_test)
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)
