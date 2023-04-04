# 자연어처리
# 효율적인 원핫이라고 생각.
# 텍스트 데이터 처리할때 유용
# 각 단어를 고정된 크기의 벡터로 변환.
# 일반적인 원핫은 데이터가 너무 늘어남
# 유사한 데이터끼리 모인다.


from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


#1. 데이터
docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요','너무 재미없다', '참 재밋네요', '환희가 잘 생기긴 했어요',
        '환희가 안해요'
        ]



#긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])


token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '환희가': 4, '재밌어요': 5, '최고에요': 6, '만든': 7, '영화예요': 8, '추천하고': 9, '싶은': 10, '영화입니다': 11,
# '한': 12, '번': 13, '더': 14, '보고': 15, '싶네요': 16, '글쎄요': 17, '별로에요': 18, '생각보다': 19,
# '지루해요': 20, '연기가': 21, '어색해요': 22, '재미없어요': 23, '재미없다': 24, '재밋네요': 25, '생기긴': 26, '했어요': 27, '안해요': 28}
# '단어 사전'의 종류가 28개이다

# print(token.word_counts)
# OrderedDict([('너무', 2), ('재밌어요', 1), ('참', 3), ('최고에요', 1), ('잘', 2), ('만든', 1), ('영화예요', 1), ('추천하고', 1), ('싶은', 1), ('영화입니다', 1), ('한', 1), ('번', 1), ('더', 1), (' 
# 보고', 1), ('싶네요', 1), ('글쎄요', 1), ('별로에요', 1), ('생각보다', 1), ('지루해요', 1), ('연기가', 1), ('어색해요', 1), ('재미없어요', 1), ('재미없다', 1), ('재밋네요', 1), ('환희가', 2), ('생 
# 기긴', 1), ('했어요', 1), ('안해요', 1)])
x = token.texts_to_sequences(docs)
# print(x)
#[[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17], [18], [19, 20], [21, 22], [23], [2, 24], [1, 25], [4, 3, 26, 27], [4, 28]]
#이것이 x값이다.
#문제는 이것이 사이즈가 다 다르다는것.
#사이즈를 하나로 맞춰주기 위해, 사이즈가 작은 데이터를 사이즈가 제일큰 데이터 만큼 빈 자리에 0의 데이터를 넣어 크기를 맞춰준다.
#패딩은 데이터의 앞에 채운다. 중요한 데이터는 뒤로 보내야한다. ex [2,5] = [0,0,0,2,5]
#단어의 연관성을 계산해야 하기 때문에 시계열 데이터라고 볼 수 있다.
#따라서 RNN 쓴다.

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=6)    #padding='pre' =앞에서부터 작업하겠다. 'post'= 뒤에서부터 작업하겠다. maxlen= 가장 긴 데이터의 길이
# print(pad_x)
# print(pad_x.shape)



# [[ 0  0  0  2  5]
#  [ 0  0  0  1  6]
#  [ 0  1  3  7  8]
#  [ 0  0  9 10 11]
#  [12 13 14 15 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0  0 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0 21 22]
#  [ 0  0  0  0 23]
#  [ 0  0  0  2 24]
#  [ 0  0  0  1 25]
#  [ 0  4  3 26 27]
#  [ 0  0  0  4 28]]
# (14, 5)

word_size = len(token.word_index) #단어 사전의 길이 
print('단어사전의 갯수 : ', word_size)    #단어사전의 갯수 :  28

# print(pad_x.shape)
pad_x= pad_x.reshape(pad_x.shape[0],pad_x.shape[1],1)
# print(pad_x.shape)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape, Embedding
import pandas as pd

# pad_x= np.array(pad_x)
# x = x.reshape(-1)
# x= pd.get_dummies(x)

# x = np.array(pd.get_dummies(pad_x[0]))

model = Sequential()
# model.add(Embedding(28, 32, input_length=5))
# model.add(Embedding(28, 32, 5)) error. input_length는 쓰거나 안쓰거나 둘중 하나지만 쓸때는 명시해줘야함
model.add(Embedding(input_dim=28, output_dim=10, input_length=6))
#input_dim = 단어 사전의 갯수
#output_dim = 아웃풋.
#input_length = maxlen 혹은 timesteps 명시하지 않아도 알아서 입력된다.
model.add(LSTM(32)) #임베딩 뒤에 통상 LSTM 써준다.
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# model.summary()
# #컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=1,)

#평가, 예측

acc = model.evaluate(pad_x, labels)[1]
print('result', acc)


x_predict = ['나는 성호가 정말 재미없다 너무 정말']
token.fit_on_texts(x_predict)
x_predict = token.texts_to_sequences(x_predict)
x_predict = np.array(x_predict)
x_pred=x_predict.reshape(x_predict.shape[0],x_predict.shape[1],1)

# print(x_pred)
# print(token.word_index)
pred = model.predict(x_pred)

# print('긍정일까 부정일까 : ',pred)

def xpred(x):
    if x > 0.5:
        return print("긍정")
    else:
        return print("부정")

print(pred)
xpred(pred)


#긍정인지 부정인지 맞춰봐!!!