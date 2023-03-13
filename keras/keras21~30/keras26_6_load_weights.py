

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model                #load 모델의 경로
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler   


#1. 데이터

datasets = load_boston()
x= datasets.data
y= datasets['target']


x_train, x_test, y_train, y_test = train_test_split (x,y,
                                                     train_size=0.8,
                                                     random_state=333,
                                                     )


scaler = StandardScaler()
# scaler.fit(x_train)   
# x_train = scaler.transform(x_train)

x_train= scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)            
# print(np.min(x_test), np.max(x_test))      






#2. 모델

# model = load_model('./_save/keras26_3_save_model.h5')     
#저장된 모델의 구조만 불러옴


input1 = Input(shape=(13,))        
dense1 = Dense(30)(input1)
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)

# model.load_weights('./_save/keras26_5_save_weights1.h5')
#컴파일 전에 저장한 save_weights1은 초기 랜덤값의 웨이트만 저장되어 있다.


model.load_weights('./_save/keras26_5_save_weights2.h5')
#컴파일은 항상 필요하다.



#3. 컴파일, 훈련


model.compile(loss = 'mae', optimizer='adam',)
# model.fit(x_train,y_train,
#           epochs=10,
#           batch_size=32,
#           verbose=1,
#           )



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)




