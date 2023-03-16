from keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import datetime
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (50000, 32, 32, 3) (50000, 1)
# (10000, 32, 32, 3) (10000, 1)

# x_train = x_train.reshape(50000, 32*32*3)
# x_test = x_test.reshape(10000, 32*32*3)


# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
#****************x_train = x_train/255.     - 콤마는 파이썬의 부동 소수점 연산이라는 표시
#****************x_test = x_test/255.


x_train = x_train.reshape(50000, 32*32*3)/255.
x_test = x_test.reshape(10000, 32*32*3)/255.         #reshape와 scaling 동시에 하기.


x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath='./_save/cnn/mnist/'
filename='{epoch:04d}-{val_acc:.4f}.hdf5'



#2. 모델구성

model = Sequential()
model.add(Conv2D(64, 
                 (3,3),
                 padding='same',
                 input_shape = (32, 32, 3),
                 activation='relu'))
model.add(Conv2D(64,
                 (3,3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D())   
model.add(Conv2D(128,
                 (3,3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D())   
model.add(Conv2D(256,
                 (3,3),
                 padding='same',
                 activation='relu'))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

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
                 batch_size =64,
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




# result : [2.391897201538086, 0.7656000256538391]
# acc : 0.7656
# 걸린시간 :  424.89


# result : [1.910552978515625, 0.741100013256073]
# acc : 0.7411
# 걸린시간 :  252.9

# acc : 0.7448
# 걸린시간 :  205.25