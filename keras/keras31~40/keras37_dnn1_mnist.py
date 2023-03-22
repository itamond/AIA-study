import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import datetime
import time

date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
filepath = './_save/dnn/dnntest/'
filename = '{epoch:04d}-{val_acc:.2f}.hdf5'



#[실습] 맹그러
# 목표 : cnn성능보다 좋게 만들자


#1. 데이터
# 리쉐이프 해주면 되겠지!!!
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler= MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)




#2. 모델구성
model =Sequential()
model.add(Dense(64, input_shape=(28*28,)))
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))


#3. 컴파일, 훈련
start_time = time.time()
model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['acc'])

es = EarlyStopping(monitor = 'val_acc',
                   patience = 50,
                   verbose = 1,
                   restore_best_weights=True,
                   mode = 'auto'
                   )

mcp = ModelCheckpoint(monitor = 'val_acc',
                      mode='auto',
                      save_best_only=True,
                      verbose=1,
                      filepath = ''.join(filepath+'_37_1_'+date+'_'+filename))

hist = model.fit(x_train, y_train,
                 epochs=2000,
                 validation_split=0.2,
                 verbose=1,
                 batch_size=32,
                 callbacks=[es])

end_time = time.time()
#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('loss :', result[0])
print('acc :', result[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_true = np.argmax(y_test, axis=1)
acc = accuracy_score(y_predict, y_true)

print('acc :', acc)

print('걸린시간 :', round(end_time - start_time, 2))

# loss : 0.11206462234258652
# acc : 0.968500018119812
# acc : 0.9685
# 걸린시간 : 1495.55