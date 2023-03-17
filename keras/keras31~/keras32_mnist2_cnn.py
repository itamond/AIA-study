from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import datetime
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#실습#
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
#reshape = 안에 있는 내용과 순서는 바뀌면 안됨, 구조만 바꾸는 함수.
# 60000,28, 14, 2 이런 식도 가능하다.
# 60000,14, 28, 2 도 가능. 
# 뒤의 rows, columns, channels 의 곲은 항상 같아야한다.
#Transpose = 열과 행을 바꾸는것. reshape와 다르다


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath='./_save/cnn/mnist/'
filename='{epoch:04d}-{val_acc:.4f}.hdf5'



#2. 모델구성

model = Sequential()
model.add(Conv2D(20, 
                 (2,2),
                 padding='same',
                 input_shape = (28, 28, 1),
                 activation='relu'))
model.add(Conv2D(10,
                 (3,3),
                 padding='same',
                 activation='relu'))
model.add(Conv2D(10,
                 (4,4),
                #  padding='same',
                 activation='relu'))
model.add(Conv2D(10,
                 (2,2),
                #  padding='same',
                 activation='relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

model = load_model('./_save/cnn/mnist/_k32_2_0316_2056_0920-0.9886.hdf5') 
#3. 컴파일, 훈련
import time
start_time = time.time()


es = EarlyStopping(monitor='val_acc',
                   mode='auto',
                   restore_best_weights=True,
                   patience=20)

mcp = ModelCheckpoint(monitor='val_acc',
                      mode='auto',
                      save_best_only=True,
                      verbose=1,
                      filepath = ''.join([filepath+'_k32_2_'+date+'_'+filename]))


model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

hist = model.fit(x_train,y_train,
                 epochs = 5000,
                 batch_size = 32,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es,mcp])

end_time = time.time()

#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_argm = np.argmax(y_test, axis=1)
acc = accuracy_score(y_argm, y_pred)

print('acc :', acc)

print('걸린시간 : ', round(end_time - start_time,2))    # round의 2는 소수 둘째까지 반환하라는것

# result : [0.07119229435920715, 0.9807999730110168]
# acc : 0.9808


# result : [0.21724455058574677, 0.988099992275238]
# acc : 0.9881
# 걸린시간 :  461.0
