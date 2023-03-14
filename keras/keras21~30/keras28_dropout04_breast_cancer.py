#y값이 실수가 아닌 0과 1일때의 모델링

#이진분류인 데이터

#나중에는 컬런에 대한 분석을 해야한다


import numpy as np
from sklearn.datasets import load_breast_cancer #유방암에 걸렸는지 아닌지 하는 데이터
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import datetime

#1. 데이터

datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)    #numpy의 디스크라이브, pd는 .describe()
# print(datasets.feature_names) # 판다스 : columns()


x = datasets['data']     #딕셔너리의 key
y = datasets.target
# print(x.shape, y.shape)  #(569, 30) (569,)

# print(datasets.isnull().sum())
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,
    random_state=333,
    test_size=0.2
)

# scaler=MinMaxScaler()
scaler=MaxAbsScaler()
# scaler=StandardScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

filepath = './_save/MCP/keras28/04/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


#2. 모델구성

# model = Sequential()
# model.add(Dense(100, activation='relu',input_dim=30))
# model.add(Dense(80,activation='relu'))
# model.add(Dense(70,activation='linear'))
# model.add(Dense(60,activation='relu'))
# model.add(Dense(30,activation='linear'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))  

input1=Input(shape=(30,))
dense1=Dense(100,activation='relu')(input1)
drop1=Dropout(0.2)(dense1)
dense2=Dense(80,activation='relu')(drop1)
drop2=Dropout(0.2)(dense2)
dense3=Dense(70,activation='relu')(drop2)
drop3=Dropout(0.2)(dense3)
dense4=Dense(60,activation='relu')(drop3)
drop4=Dropout(0.2)(dense4)
dense5=Dense(30,activation='relu')(drop4)
drop5=Dropout(0.2)(dense5)
dense6=Dense(10,activation='relu')(drop5)
drop6=Dropout(0.2)(dense6)
output1=Dense(1,activation='relu')(drop6)
model=Model(inputs=input1, outputs=output1)



#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            #   metrics=['accuracy','mse'],    # 훈련 과정에 accuracy를 추가.   hist.history에 다 기록됨. 대괄호=리스트
              )                             #대괄호 안에 다른 지표도 추가 가능(list이기 때문에)
                                            #metrics = 훈련에 영향을 주지않음. 지표확인용, 풀네임 줄임말 상관 없음


mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath = ''.join([filepath, 'k28_04_', date, '_', filename])
                      )
         
es =EarlyStopping(monitor='val_loss',
                  mode='min',
                  patience=100,
                  restore_best_weights=True,
                  verbose=1,
                  )

hist = model.fit(x_train,y_train,
                 epochs=1000,
                 batch_size=4,
                 validation_split=0.2,
                 verbose=1,
                 callbacks=[es, mcp],
                 )


#4. 평가, 예측
result = model.evaluate(x_test,y_test)    #엄밀히 얘기하면 loss = result이다. 
                                         #model.evaluate=
                                         #model.compile에 추가한 loss 및 metrix 모두 result로 표시된다.
                                         #metrix의 accuracy는 sklearn의 accuracy_score 값과 동일하다.
print('result :', result)

y_predict= np.round(model.predict(x_test))
r2= r2_score(y_test, y_predict)
print('r2 :', r2)

def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_test, y_predict)
print('rmse :', rmse)

#평소처럼 수치가 아닌 0이냐 1이냐를 맞춰야한다면 accuracy_score 사용 



from sklearn.metrics import accuracy_score, r2_score
acc = accuracy_score(y_test, y_predict)
print('acc :', acc)


# plt.rcParams['font.family']='Malgun Gothic'
# plt.figure(figsize=(9,6))
# plt.title('유방암')
# plt.plot(hist.history['loss'], c='red', marker='.', label='로쓰')
# plt.plot(hist.history['val_loss'], c='blue', marker='.', label='발_로쓰')
# plt.grid()
# plt.legend()
# plt.show()



# acc : 0.8947368421052632


# result : 0.11957277357578278
# r2 : 0.7999297999298
# acc : 0.956140350877193

# result : 0.1412312537431717
# r2 : 0.87995787995788
# rmse : 0.16222142113076254
# acc : 0.9736842105263158    스케일러 적용

# result : 0.42463839054107666
# r2 : -1179.974376974377
# rmse : 16.09020623988721
# acc : 0.3508771929824561 드롭아웃 적용