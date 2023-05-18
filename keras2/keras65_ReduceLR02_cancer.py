#ReduceLRonPlateau : 개선이 없으면 Learningrate 반으로 줄인다.


import numpy as np

from sklearn.datasets import fetch_california_housing, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error, r2_score


#1. 데이터
datasets = load_breast_cancer()
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
model.add(Dense(64, input_dim=x.shape[1]))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1,activation = 'sigmoid'))

#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate,)
model.compile(loss='binary_crossentropy', optimizer=optimizer)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor = 'val_loss',
                   patience = 30, mode = 'min',
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

y_predict= np.round(model.predict(x_test))

from sklearn.metrics import accuracy_score, r2_score
acc = accuracy_score(y_test, y_predict)
print('acc :', acc)


# acc : 0.8771929824561403