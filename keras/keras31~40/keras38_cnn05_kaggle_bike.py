from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import time
import pandas as pd 
#1. 데이터

path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission = pd.read_csv(path+'sampleSubmission.csv', index_col = 0)

print(train_csv.shape) #(10886, 11)
x = train_csv.drop(['casual','registered','count'], axis=1)
y = train_csv['count']
print(type(x))
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=221,
                                                    )
print(type(x_train))

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(np.min(x_test), np.max(x_test))
print(x_train.shape, x_test.shape)  #(8708, 8) (2178, 8)
x_train = x_train.reshape(-1, 8, 1, 1)
x_test = x_test.reshape(-1, 8, 1, 1)




#2. 모델 구성

input1 = Input(shape=(8, 1, 1))
conv1 = Conv2D(20, (2,1),
               padding = 'same',
               activation='relu')(input1)
conv2 = Conv2D(20, (2,1),
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
                 batch_size=128,
                 validation_split = 0.2,
                 callbacks=[es])

end_time=time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred=model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 :', r2)


# result : 5022.6787109375
# r2 : 0.1678296833020645

# result : 25397.431640625
# r2 : 0.26491983517954387