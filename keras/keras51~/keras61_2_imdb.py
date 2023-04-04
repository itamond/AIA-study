from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd

#1. 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words= 10000, test_split=0.2
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