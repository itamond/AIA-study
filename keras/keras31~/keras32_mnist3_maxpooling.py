from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
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

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

#2. 모델
model = Sequential()
model.add(Conv2D(64, (2,2), 
                 padding = 'same', 
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (2,2), 
                 padding = 'same', 
))
model.add(MaxPooling2D())              #맥스풀링 사용법.
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(32, 2))    #32 = 필터 갯수, 2 = (2,2) 통상적으로 정사각형이기때문에 한변만 기재하도록 편의화
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train,epochs=30, batch_size = 128,)


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print ('loss :', results[0])
print ('acc :', results[1])



#맥스풀링
#정제되지않은 처음 이미지 데이터는 연산량이 너무 많다.
#최초의 이미지 데이터에서 필터를 거쳐 가장 높은 데이터만을 땡겨온다. 이 과정은 '연산'이 아니다.
#이 과정을 통하면 최초의 쓸모없는 데이터 버리는 연산을 하는 시간을 단축할 수 있다.
#이 과정을 실행해보면 통상적으로 성능과 속도 모두 좋아진다.
#맥스풀링은 디폴트 (2,2)의 필터로 최대값을 추출하지만 '중첩 하지않는다.'