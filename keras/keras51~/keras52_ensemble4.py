


#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301,401)])     # 삼성, 아모레



#온도, 습도, 강수량

print(x1_datasets.shape)      

x1 = np.transpose(x1_datasets)



y1 = np.array(range(2001, 2101))  # 환율
y2 = np.array(range(1001, 1101))  # 금리


from sklearn.model_selection import train_test_split
# x1_train, x1_test, x2_train, x2_test = train_test_split(
#     x1, x2, train_size=0.7, random_state=333
# )


# y_train, y_test = train_test_split(
#     y, train_size=0.7, random_state=333
# )

# #동일한 랜덤 스테이트로 짜르면 동일한 순서로 짤림

x1_train, x1_test,\
y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1,y1, y2, train_size=0.7, random_state=333
)

# 코드가 너무 길면 엔터 치고 \ 하면 '한 줄'이라고 명시됨



print(x1_train.shape, x1_test.shape)
print(y1_train.shape, y1_test.shape)

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



from tensorflow.keras.layers import concatenate, Concatenate

#concatenate  사슬처럼 엮다. 소문자는 함수 대문자는 클래스



bungi1 = Dense(40,activation='swish')(output1)
bungi2 = Dense(30,activation='swish')(bungi1)
bungi3 = Dense(20,activation='swish')(bungi2)
bungi4 = Dense(10,activation='swish')(bungi3)
output2 =Dense(1)(bungi4)


bungi21 = Dense(30,activation='swish')(output1)
bungi22 = Dense(20,activation='swish')(bungi21)
bungi23 = Dense(10,activation='swish')(bungi22)
output3 = Dense(1)(bungi23)


model = Model(inputs=input1, outputs=[output2, output3])

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

model.fit(x1_train,[y1_train,y2_train],
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
result = model.evaluate(x1_test, [y1_test,y2_test])
print('result :', result)

predict = model.predict(x1_test,)

r2_1 = r2_score(y1_test, predict[0])
r2_2 = r2_score(y2_test, predict[1])
print('r2 :', (r2_1+r2_2)/2)

rmse1 = RMSE(y1_test, predict[0])
rmse2 = RMSE(y2_test, predict[1])
print('rmse :', (rmse1+rmse2)/2)

print(predict)
print(len(predict), len(predict[0]))       # 2, 30     #리스트의 행과 열 보는법
#리스트는 파이썬 기본 자료형으로써 shape 함수를 사용할 수 없다. 따라서 len을 사용하여 데이터를 확인함
#np.array 로 넘파이화 하면 쉐이프 볼수잇음


# result : [110.96007537841797, 22.88892936706543, 88.0711441040039]          #첫번째= 로스의 합, 두번째 = 첫번째 로스, 세번째 = 두번째 로스
# r2 : 0.9059950760159865
# rmse : 7.0844300831876605


# r2 : 0.9999227776980644
# rmse : 0.19104251536220024