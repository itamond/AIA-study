from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import time
#1. 데이터

datasets= fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x.shape)   #(20640, 8)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=221,
                                                    )


scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(np.min(x_test), np.max(x_test))
print(x_train.shape, x_test.shape)  #(16512, 8) (4128, 8)
x_train = x_train.reshape(16512, 2, 2, 2)
x_test = x_test.reshape(4128, 2, 2, 2)




#2. 모델 구성

input1 = Input(shape=(2, 2, 2))
conv1 = Conv2D(20, (2,2),
               padding = 'same',
               activation='relu')(input1)
conv2 = Conv2D(20, (2,2),
               padding = 'same',
               activation='relu')(conv1)
# mp1 = MaxPooling2D()
# pooling1 = mp1(conv2)
flat1 = Flatten()(conv2)
dense1 = Dense(10, activation='relu')(flat1)
output1 = Dense(1)(dense1)

model=Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
start_time=time.time()
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor = 'val_loss',
                   patience=20,
                   restore_best_weights=True,
                   verbose=1)
hist = model.fit(x_train, y_train,
                 epochs =1000,
                 batch_size=256,
                 validation_split = 0.2,
                 callbacks=[es])

end_time=time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred=model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 :', r2)


# result : 0.24936871230602264
# r2 : 0.816139800241104