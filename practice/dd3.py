import tensorflow as tf
import pandas as pd
import random
import matplotlib as mpl
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
# 0. seed initialization
# 1. data prepare
datasets = load_wine()
x=datasets.data
y=datasets.target

print(datasets.DESCR)
# print(x.shape,y.shape)
print(f'y의 라벨 값  : {np.unique(y)}')
y=pd.Categorical(y)
y=np.array(pd.get_dummies(y,prefix='number'))
# print(x.shape,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)

# 2. model build
model=Sequential()
model.add(Dense(50,input_dim=x.shape[1],activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(y.shape[1],activation='softmax'))

# 3. compile, training
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_loss',mode='min',patience=30,verbose=True,restore_best_weights=True)
hist=model.fit(x_train,y_train,batch_size=len(x_train),validation_split=0.2,verbose=True,epochs=1000)

# 4. predict,evaluate
y_predict=np.array(model.predict(x_test))
y_predict=np.argmax(y_predict,axis=1)
y_test=np.argmax(y_test,axis=1)
print(f'accuracy : {accuracy_score(y_test,y_predict)}')

import matplotlib.pyplot as plt
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['loss'])
plt.show()