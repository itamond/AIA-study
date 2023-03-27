#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301,401)])     # 삼성, 아모레
x2_datasets = np.array([range(101,201),range(411, 511),range(150, 250)])
#온도, 습도, 강수량

print(x1_datasets.shape)      
print(x2_datasets.shape)

x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T

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

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y, train_size=0.7, random_state=333
)

print(x1_train.shape, x1_test.shape)
print(x2_train.shape, x2_test.shape)
print(y_train.shape, y_test.shape)

# (70, 2) (30, 2)
# (70, 3) (30, 3)
# (70,) (30,)



#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='stock1')(input1)
dense2 = Dense(20, activation='relu', name='stock2')(dense1)
dense3 = Dense(30, activation='relu', name='stock3')(dense2)
output1 = Dense(11, activation='relu', name='output1')(dense3)


#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10, name='weather1')(input2)
dense12 = Dense(10, name='weather2')(dense11)
dense13 = Dense(10, name='weather3')(dense12)
dense14 = Dense(10, name='weather4')(dense13)
output2 = Dense(11, name='output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate

#concatenate  사슬처럼 엮다. 소문자는 함수 대문자는 클래스

merge1 = concatenate([output1, output2], name='mg1')   #a모델과 b모델의 아웃풋이 merge의 인풋이 된다.
#리스트 형태로 입력
merge2 = Dense(2, activation='relu', name='mg2')(merge1)
merge3 = Dense(3, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()


#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  input_2 (InputLayer)           [(None, 3)]          0           []

#  input_1 (InputLayer)           [(None, 2)]          0           []

#  weather1 (Dense)               (None, 10)           40          ['input_2[0][0]']

#  stock1 (Dense)                 (None, 10)           30          ['input_1[0][0]']

#  weather2 (Dense)               (None, 10)           110         ['weather1[0][0]']

#  stock2 (Dense)                 (None, 20)           220         ['stock1[0][0]']

#  weather3 (Dense)               (None, 10)           110         ['weather2[0][0]']

#  stock3 (Dense)                 (None, 30)           630         ['stock2[0][0]']

#  weather4 (Dense)               (None, 10)           110         ['weather3[0][0]']

#  output1 (Dense)                (None, 1)            31          ['stock3[0][0]']

#  output2 (Dense)               (None, 1)            11          ['weather4[0][0]']

#  mg1 (Concatenate)              (None, 2)            0           ['output1[0][0]',
#                                                                   'output2[0][0]']

#  mg2 (Dense)                    (None, 2)            6           ['mg1[0][0]']

#  mg3 (Dense)                    (None, 3)            9           ['mg2[0][0]']

#  last (Dense)                   (None, 1)            4           ['mg3[0][0]']

# ==================================================================================================
# Total params: 1,311
# Trainable params: 1,311
# Non-trainable params: 0
# __________________________________________________________________________________________________



#모델 1과 모델 2의 아웃풋은 결국 큰 모델의 히든레이어이기 때문에
#노드가 1개일 필요가 없다. 오히려 값이 소멸됨.