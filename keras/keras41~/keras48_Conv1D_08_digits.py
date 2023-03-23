from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, GRU, LSTM, Conv1D, Flatten
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import time
import pandas as pd 
from tensorflow.keras.utils import to_categorical
#1. 데이터

datasets=load_digits()

# print(train_csv.shape) #(10886, 11)
x = datasets.data
y = datasets['target']

# print(np.unique(y, return_counts=True))
# print(x.shape, y.shape) #(150, 4) (150,)

y = pd.get_dummies(y)
print(y.shape)

y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=221,
                                                    stratify=y
                                                    )

print(x_train.shape, x_test.shape)  #(1437, 64) (360, 64)



scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(np.min(x_test), np.max(x_test))
# print(x_train.shape, x_test.shape)  
x_train = x_train.reshape(-1, 8, 8)
x_test = x_test.reshape(-1, 8, 8)




#2. 모델 구성

input1 = Input(shape=(8, 8))
Conv1 = Conv1D(20, 2)(input1)
Conv2 = Conv1D(20, 2)(Conv1)
Flat1 = Flatten()(Conv2)
dense1 = Dense(10, activation='relu')(Flat1)
dense2 = Dense(10, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
output1 = Dense(10, activation='softmax')(dense3)

model=Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
start_time=time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam')
es = EarlyStopping(monitor = 'val_loss',
                   patience=20,
                   restore_best_weights=True,
                   verbose=1)

hist = model.fit(x_train, y_train,
                 epochs =1000,
                 batch_size=16,
                 validation_split = 0.2,
                 callbacks=[es])

end_time=time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred=model.predict(x_test)
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, y_pred)
print('acc :', acc)



# result : 0.028841407969594002
# acc : 0.9666666666666667


# result : 0.014466214925050735
# acc : 0.9722222222222222

# result : 0.007362670265138149
# acc : 0.9611111111111111

# result : 0.23147152364253998
# acc : 0.925  LSTM적용


# result : 0.2815871834754944
# acc : 0.9166666666666666 Conv1D 적용