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

path = './_data/ddarung/'
path_save = './_save/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission = pd.read_csv(path+'submission.csv', index_col = 0)

train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']



# print(x.shape)  #(1328, 9)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=221,
                                                    )


scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(np.min(x_test), np.max(x_test))
print(x_train.shape, x_test.shape)  #(1062, 9) (266, 9)
x_train = x_train.reshape(-1, 3, 3, 1)
x_test = x_test.reshape(-1, 3, 3, 1)
test_csv=np.array(test_csv)
test_csv = scaler.transform(test_csv)
test_csv=test_csv.reshape(-1, 3, 3, 1)



#2. 모델 구성

input1 = Input(shape=(3, 3, 1))
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
                 epochs =2000,
                 batch_size=32,
                 validation_split = 0.2,
                 callbacks=[es])

end_time=time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result :', result)
y_pred=model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 :', r2)


y_submit=model.predict(test_csv)

submission=pd.read_csv(path+'submission.csv',index_col=0)


submission['count']=y_submit
#카운트에 y서브밋을 넣고

submission.to_csv(path_save+'submission_validation.csv')


# result : 5022.6787109375
# r2 : 0.1678296833020645


# result : 2365.85595703125
# r2 : 0.6608607911204452