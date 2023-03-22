import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.utils import to_categorical
from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import datetime
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")



#1. 데이터
(x_train, y_train), (x_test, y_test)= cifar100.load_data()
# print(np.unique(y_train, return_counts=True))

print(x_train.shape, x_test.shape)   #(50000, 32, 32, 3) (10000, 32, 32, 3)

# x_train = x_train.reshape()

x_train = x_train.reshape(50000, 32*32*3)/255.
x_test = x_test.reshape(10000, 32*32*3)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2. 모델구성
start_time = time.time()
model = Sequential()
model.add(Dense(512, input_shape=(32*32*3,)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc',
                   mode='auto',
                   restore_best_weights=True,
                   verbose=1,
                   patience=40)


hist = model.fit(x_train,y_train,
                 epochs=3000,
                 batch_size=256,
                 verbose=1,
                 callbacks=[es],
                 validation_split=0.2,
                 )

end_time = time.time()

#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred=model.predict(x_test)
y_pred=np.argmax(y_pred, axis=1)
y_true=np.argmax(y_test, axis=1)
acc = accuracy_score(y_pred, y_true)
print('acc :', acc)
print('걸린시간 :', round(end_time - start_time, 2))