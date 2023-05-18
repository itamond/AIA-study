#ReduceLRonPlateau : 개선이 없으면 Learningrate 반으로 줄인다.


import numpy as np

from sklearn.datasets import fetch_california_housing,load_wine
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical

#1. 데이터
datasets = load_wine()
x = datasets.data
y1 = datasets.target

print(np.unique(y1))

y = to_categorical(y1)

x_train, x_test, y_train, y_test = tts(x,y,
                                       train_size=0.8,
                                       shuffle=True,
                                       random_state=337,
                                       )

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=x.shape[1]))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(len(np.unique(y1)),activation='softmax'))


#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate,)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

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

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict)

print('acc :', acc)

# acc : 0.8611111111111112
