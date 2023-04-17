#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301,401)])     # 삼성, 아모레
x2_datasets = np.array([range(101,201),range(411, 511),range(150, 250)])
x3_datasets = np.array([range(201,301),range(511, 611),range(1300, 1400)])


#온도, 습도, 강수량

print(x1_datasets.shape)      
print(x2_datasets.shape)

x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
x3 = x3_datasets.T
print(x1.shape)
print(x2.shape)
# (100, 2)
# (100, 3)

y = np.array(range(2001, 2101))  # 환율

from sklearn.model_selection import train_test_split
# x1_train, x1_test, x2_train, x2_test = train_test_split(
#     x1, x2, train_size=0.7, random_state=333
# )


# y_train, y_test = train_test_split(
#     y, train_size=0.7, random_state=333
# )

# #동일한 랜덤 스테이트로 짜르면 동일한 순서로 짤림

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1, x2, x3, y, train_size=0.7, random_state=333
)

print(x1_train.shape, x1_test.shape)
print(x2_train.shape, x2_test.shape)
print(x3_train.shape, x3_test.shape)
print(y_train.shape, y_test.shape)

# (70, 2) (30, 2)
# (70, 3) (30, 3)
# (70,) (30,)



#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(40, activation='swish', name='stock1')(input1)
dense2 = Dense(30, activation='swish', name='stock2')(dense1)
dense3 = Dense(20, activation='swish', name='stock3')(dense2)
output1 = Dense(11, activation='swish', name='output1')(dense3)


#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(40,activation='swish', name='weather1')(input2)
dense12 = Dense(30,activation='swish', name='weather2')(dense11)
dense13 = Dense(20,activation='swish', name='weather3')(dense12)
dense14 = Dense(10,activation='swish', name='weather4')(dense13)
output2 = Dense(5, name='output2')(dense14)

#2-3. 모델3
input3 = Input(shape=(3,))
dense21 = Dense(40,activation='swish', name='weather11')(input3)
dense22 = Dense(30,activation='swish', name='weather21')(dense21)
dense23 = Dense(20,activation='swish', name='weather31')(dense22)
dense24 = Dense(10,activation='swish', name='weather41')(dense23)
output3 = Dense(5, name='output3')(dense24)


from tensorflow.keras.layers import concatenate, Concatenate

#concatenate  사슬처럼 엮다. 소문자는 함수 대문자는 클래스

merge1 = concatenate([output1, output2, output3], name='mg1')   #a모델과 b모델의 아웃풋이 merge의 인풋이 된다.
#리스트 형태로 입력
merge2 = Dense(30, activation='swish', name='mg2')(merge1)
merge3 = Dense(20, activation='swish', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2, input3], outputs=last_output)

model.summary()


from tensorflow.keras.callbacks import EarlyStopping


#모델 1과 모델 2의 아웃풋은 결국 큰 모델의 히든레이어이기 때문에
#노드가 1개일 필요가 없다. 오히려 값이 소멸됨.

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss',
                   patience=50,
                   restore_best_weights=True,
                   verbose=1)
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train,x2_train,x3_train],y_train,
          epochs=3000,
          batch_size=8,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )

#4. 평가, 예측
from sklearn.metrics import r2_score, mean_squared_error
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))
result = model.evaluate([x1_test, x2_test, x3_test], y_test)
print('result :', result)

predict = model.predict([x1_test, x2_test, x3_test])

r2 = r2_score(y_test, predict)
print('r2 :', r2)

rmse = RMSE(y_test, predict)
print('rmse :', rmse)

print(predict)


# result : 0.013598564080893993
# r2 : 0.9999769636257603

# result : 0.012901284731924534
# r2 : 0.9999781367088494
# rmse : 0.11359280679863003


# result : 0.0018022020813077688
# r2 : 0.9999969485717409
# rmse : 0.042436996958864055

# result : 0.0026946489233523607
# r2 : 0.9999954361710967
# rmse : 0.05189885050781414