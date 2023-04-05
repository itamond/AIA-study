from tensorflow.keras.datasets import imdb              #영화평 데이터
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
#1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words= 10000 #트레인 데이터의 디폴트는 80?
)

print(x_train)
print(y_train)
print(x_train.shape, y_train.shape)
# (8982,) (8982,)
print(x_test.shape, y_test.shape)
# (2246,) (2246,)

# print(len(x_train[0]),len(x_train[1]))   #87 56        길이가 각각 다름. 넘파이 안의 리스트이기 때문에 가능하다
print(np.unique(y_train, return_counts=True))
#array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    #    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    #    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    #46개의 클래지파이어

# 모델의 아웃풋은 46, softmax로 맞춤
# 임베딩의 인풋 딤은 10000개, num_words다.
# input_length = 최대 길이로 맞춤


print(x_train)

print(y_train)   #[1 0 0 ... 0 1 0] 긍정 부정 문제?
print(x_train.shape, y_train.shape)  #(25000,) (25000,)
# print(np.unique(y_train, return_counts=True))    #(array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
print(pd.value_counts(y_train))
#1    12500
#0    12500


#이진분류에서 o과 1의 갯수가 너무 많이 차이난다면, 적은 쪽의 데이터를 증폭하는 것을 고민해야 한다.
#이진분류에서는 증폭으로 인한 과적합은 크게 걱정하지 않아도 된다. 단순히 0과 1을 찾아내는 문제이기 때문이다.
#판다스에선 value_counts


print("영화평의 최대 길이 :", max(len(i) for i in x_train))  #뉴스기사의 최대길이 : 2494
print("영화평의 평균 길이 :", sum(map(len, x_train)) / len(x_train))  #뉴스기사의 평균 길이 : 238.71




x_train = pad_sequences(x_train, truncating='pre', maxlen=200, padding='pre'        #truncating= 어디 잘라서 버릴거야? 100이상이면 뒤 100개 남기고 버림. 100 이하면 패딩 씌움
                        )
x_test = pad_sequences(x_test, truncating='pre', maxlen=200, padding='pre'        #truncating= 어디 잘라서 버릴거야? 100이상이면 뒤 100개 남기고 버림. 100 이하면 패딩 씌움
                        )


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


model = Sequential()
model.add(Embedding(10000, 40, input_length=200))
model.add(LSTM(32))
# model.add(Dense(64, activation='swish'))
# model.add(Dense(128, activation='swish'))
# model.add(Dense(64, activation='swish'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train,
          epochs=30,
          batch_size=128)

acc = model.evaluate(x_test, y_test)[1]

print('acc :', acc)

