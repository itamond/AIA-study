from tensorflow.keras.preprocessing.text import Tokenizer

text1 = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2= '나는 지구용사 배환희다. 멋있다. 또 또 얘기해부아'

# Tokenizer = 문장을 어절별로 Token화 해서 잘라내는 개념
# 컴퓨터에게 인식 시키기 위해서 이미지를 수치화 하듯 문장도 수치화 해야한다.
# 토크나이저는 문장을 어절 단위로 잘라낸 다음 수치를 부여한다.
# 한국어는 조사(은 는 이 가)가 붙어 있기 때문에 잘 먹히지 않는다.

token = Tokenizer()
token.fit_on_texts([text1, text2])    #리스트 형태로 받아준다.

print(token.word_index)   
# {'마구': 1, '나는': 2, '매우': 3, '또': 4, '진짜': 5, '맛있는': 6, '밥을': 7, '엄청': 8, '먹었다': 9, '지구용사': 10, '배환희다': 11, '멋있다': 12, '얘기해부아': 13}
# 가장 많이 사용된 단어가 앞의 숫자를 배정받는다.
print(token.word_counts)
# OrderedDict([('나는', 2), ('진짜', 1), ('매우', 2), ('맛있는', 1), 
# ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1), ('지구용사', 1), ('배환희다', 1), ('멋있다', 1), ('또', 2), ('얘기해부아', 1)])
# 단어가 사용된 횟수 출력

# text= text1 + text2

#토큰화 한 문장을 숫자로 바꿔주기
x = token.texts_to_sequences([text1, text2])       #바꾼 텍스트를 변수 지정 
# print(x)
# [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]   1행 11열
# 수치마다 우열이 있으면 안되므로 원핫 인코딩 해줘야함


####### 1. to_categorical #######
from tensorflow.keras.utils import to_categorical




##################################################
x = x[0]+x[1]               #리스트는 더하기가 된다~~
# 혹은 append로 붙일 수 있다.
##################################################





# x = to_categorical(x)
# print(x.shape)         #(18, 14)      0이 붙어서 14다.
# # (1, 11, 9) to categorical은 쓸모없는 0데이터에도 위치를 부여한다...
# # 이럴경우 첫 0을 삭제하고 리쉐이프 해주면 됨

####### 2. get_dummies ########
import pandas as pd
import numpy as np

# x= np.array(x)
# x = x.reshape(-1)
# print(x)
# x= pd.get_dummies(x)
# x= np.array(x)
# print(x)
# # x = pd.get_dummies(np.array(x).reshape(-1,))
# x = pd.get_dummies(np.array(x).ravel()) #flatten과 동일. 넘파이의 flatten
# # x = np.array(pd.get_dummies(x[0]))

# print(x)
# print(x.shape)


# # ###### 3. 사이킷런 onehot ######
from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder()
x = np.array(x).reshape(-1,1)    # 1행 11열 형태를 11행 1열 형태로 바꿈
x = ohe.fit_transform(x).toarray() # 2차원 형태로 받아야 하는 원핫인코딩

print(x)

print(x.shape)