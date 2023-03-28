# 삼성전자와 현대자동차 주가로 삼성전자 주가 맞추기

# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timesteps와 feature는 알아서 잘라라

# 제공된 데이터 외 추가 데이터 사용금지

# 1. 삼성전자 28일(화) 종가 맞추기 (점수 배점 0.3)
# 2. 삼성전자 29일(수) 아침 시가 맞추기 (점수 배점 0.7)


#마감시간 : 27일 월 23시 59분 59초        /    28일 화 23시 59분 59초
#배환희 [삼성 1차] 60,350,07원   (np.round 소수 둘째자리까지)
#배환희 [삼성]
#첨부파일 : keras53_samsung2_bhh_submit.py       데이터 및 가중치 불러오는 로드가 있어야함
#          keras53_samsung4_bhh_submit.py
#가중치 :  _save/samsung/keras53_samsung2_bhh.h5 / hdf5
#         _save/samsung/keras53_samsung4_bhh.h5 / hdf5



from tensorflow.python.keras.layers import concatenate
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling2D, Input, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split as tts
import numpy as np
import pandas as pd
import datetime
import time

#1. 데이터

path1 = './_data/시험/'
savepath = './_save/samsung/'
mcpname = '{epoch:04d}-{val_loss:.2f}.hdf5'

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

datasets1= pd.read_csv(path1+'삼성전자 주가2.csv', index_col=0, encoding='cp949')
datasets2= pd.read_csv(path1+'현대자동차.csv', index_col=0, encoding='cp949')


# print(datasets1.columns)
# Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],

feature_cols = ['시가', '고가', '저가', 'Unnamed: 6', '등락률', '거래량', '기관', '개인', '외국계', '종가']


x1 = datasets1[feature_cols]
x2 = datasets2[feature_cols]
x1 = x1.rename(columns={'Unnamed: 6':'증감량'})
x2 = x2.rename(columns={'Unnamed: 6':'증감량'})
y = datasets1['종가']
#concat
x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)
# print(x1)

# x1.fillna('None', inplace=True)
# x2.fillna('None', inplace=True)
# y.fillna(y.mean(), inplace=True)
x1 = x1[:200]
x2 = x2[:200]
y = y[:200]

# print(x1)

x1 =np.char.replace(x1.astype(str), ',', '').astype(np.float64)
x2 = np.char.replace(x2.astype(str), ',', '').astype(np.float64)
y = np.char.replace(y.astype(str), ',', '').astype(np.float64)



def RMSE(a,b) :
    return np.sqrt(mean_squared_error(a,b))


timesteps=10


def split_x(datasets, timesteps):
    aaa=[]
    for i in range(len(datasets)-timesteps):
        subset = datasets[i:(i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)



x1_train,x1_test, x2_train, x2_test, y_train, y_test = tts(x1,x2,y,
                                                           train_size=0.8,
                                                           shuffle=False,
                                                           random_state=30,
                                                           )


print('분리',x1_train.shape, x1_test.shape)


scaler = MinMaxScaler()

x1_train = split_x(x1_train, timesteps)
x1_test = split_x(x1_test, timesteps)
x2_train = split_x(x2_train, timesteps)
x2_test = split_x(x2_test, timesteps)
print('스플릿',x1_train.shape, x1_test.shape)



x1_train = x1_train.reshape(-1,timesteps*10)
x1_test = x1_test.reshape(-1,timesteps*10)
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)





x2_train = x2_train.reshape(-1,timesteps*10)
x2_test = x2_test.reshape(-1,timesteps*10)
x2_train = scaler.fit_transform(x2_train)
x2_test = scaler.transform(x2_test)

y_train = y_train[timesteps:]
y_test = y_test[timesteps:]

# x1_train = x1_train.reshape(-1,20, 10)
# x1_test = x1_test.reshape(-1,20, 10)
# x2_train = x2_train.reshape(-1,20, 10)
# x2_test = x2_test.reshape(-1,20, 10)



x1_train = np.array(x1_train)
x1_test = np.array(x1_test)
x1_train= x1_train.reshape(-1,timesteps,10)
x1_test= x1_test.reshape(-1,timesteps,10)



x2_train = np.array(x2_train)
x2_test = np.array(x2_test)
x2_train= x2_train.reshape(-1,timesteps,10)
x2_test= x2_test.reshape(-1,timesteps,10)

#print(x1_train.shape, x1_test.shape)

#2. 모델구성

# # 2-1. 모델1
# input1 = Input(shape=(timesteps,10))
# lstm1 = LSTM(256, activation='swish', name='lstm1')(input1)
# dense1 = Dense(128, activation='swish', name='dense1')(lstm1)
# dense2 = Dense(64, activation='swish', name='dense2')(dense1)
# dense3 = Dense(32, activation='swish', name='dense3')(dense2)
# dense4 = Dense(64, activation='swish', name='dense4')(dense3)
# dense5 = Dense(32, activation='swish', name='dense5')(dense4)
# output1 = Dense(16, name='output1')(dense5)



# # 2-2. 모델2
# input2 = Input(shape=(timesteps, 10))
# lstm2 = LSTM(256, activation='swish', name='lstm2')(input2)
# dense11 = Dense(128, activation='swish', name='dense11')(lstm2)
# dense12 = Dense(64, activation='swish', name='dense12')(dense11)
# dense13 = Dense(32, activation='swish', name='dense13')(dense12)
# dense14 = Dense(64, activation='swish', name='dense14')(dense13)
# dense15 = Dense(32, activation='swish', name='dense15')(dense14)
# output2 = Dense(16, name='output2')(dense13)

# merge1 = concatenate([output1, output2], name='merge1')
# merge2 = Dense(50, activation='swish', name='merge2')(merge1)
# merge3 = Dense(30, activation='swish', name='merge3')(merge2)
# merge4 = Dense(20, activation='swish', name='merge4')(merge3)
# merge5 = Dense(10, activation='swish', name='merge5')(merge4)
# last_output = Dense(1, name='last')(merge5)

# model = Model(inputs=[input1, input2], outputs=[last_output])
model = load_model('./_save/samsung/keras53_samsung2_bhh.h5')

# #3. 컴파일, 훈련
# es = EarlyStopping(monitor='val_loss',
#                    patience=50,
#                    restore_best_weights=True,
#                    verbose=1,
#                    mode='auto',
#                    )


# mcp = ModelCheckpoint(monitor='val_loss',
#                       save_best_only=True,
#                       mode='auto',
#                       filepath=''.join('_save/samsung/keras53_samsung4_bhh.hdf5'))


model.compile(loss = 'mse', optimizer = 'adam',
              )


# hist = model.fit([x1_train, x2_train], y_train,
#                  epochs=200,
#                  batch_size=64,
#                  verbose=1,
#                  validation_split=0.2,
#                  callbacks=[es, mcp])

# model.save_weights("./_save/samsung/keras_samsung2_bhh.h5")
# #4. 평가, 예측



result= model.evaluate([x1_test,x2_test], y_test)
print('mse_loss :', result)


x1_pred = x1_test[-timesteps:]
x2_pred = x2_test[-timesteps:]

pred = model.predict([x1_pred,x2_pred])
# r2 = r2_score(y_test, pred)
# print('r2_score :', r2)

print(x1_train.shape, x2_train.shape, y_train.shape)
print(x1_test.shape,x2_test.shape,y_test.shape)

# def RMSE(a,b) :
#     return np.sqrt(mean_squared_error(a,b))

# rmse = RMSE(y_test, pred)
# print('rmse :', rmse)
print('y_pred :', pred[-1:])

# mse_loss : 12736994.0
# y_pred : [[62259.945]]


