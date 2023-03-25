import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv2D, MaxPooling2D, LSTM, SimpleRNN, GRU, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import datetime


#1. 데이터

path = './_data/kaggle_jena/'
savepath = './_save/MCP/'
mcpname = '{epoch:04d}-{val_loss:.2f}.hdf5'

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# print(datasets.shape)   #(420551, 14)

# print(datasets.columns)

# Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
#        'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
#        'wd (deg)'],
#       dtype='object')

x = datasets.drop(['T (degC)'], axis=1)
y = datasets['T (degC)']

# print(x.shape, y.shape)    #(420551, 13) (420551,)




#트레인/ 테스트/ 프레드 분리 및 스케일러 적용

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=False,
                                                    )


x_test, x_pred, y_test, y_pred = train_test_split(x_test, y_test,
                                                  train_size=0.67,
                                                  shuffle=False,
                                                  )

print(x_train.shape)
print(x_test.shape)
print(x_pred.shape)
# (294385, 13)
# (84531, 13)
# (41635, 13)

scaler = Normalizer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_pred =scaler.transform(x_pred)


#스플릿 x 함수 정의
ts = 10

def split_x(a, ts):
    aaa = []
    for i in range(len(a)-ts-1) :
        subset = a[i : (i + ts)]
        aaa.append(subset)
    return np.array(aaa)


x_train = split_x(x_train, ts)
x_test = split_x(x_test, ts)
x_pred = split_x(x_pred, ts)

y_train = y_train[(ts+1): ]
y_test = y_test[(ts+1): ]
y_pred = y_pred[(ts+1): ]

# print(x_train.shape, x_test.shape, x_pred.shape)    #(294374, 10, 13) (84520, 10, 13) (41624, 10, 13)
# print(y_train.shape, y_test.shape, y_pred.shape)



#2. 모델 구성


input1 = Input(shape =(10,13))
lstm1 = LSTM(256)(input1)
drop1 = Dropout(0.5)(lstm1)
dense1 = Dense(128)(drop1)
drop2 = Dropout(0.5)(dense1)
dense2 = Dense(64)(drop2)
dense3 = Dense(32)(dense2)
dense4 = Dense(8)(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss',
                   patience=30,
                   restore_best_weights=True,
                   verbose=1,
                   mode='auto',
                   )


mcp = ModelCheckpoint(monitor='val_loss',
                      save_best_only=True,
                      mode='auto',
                      filepath=''.join(savepath+'_jena_1_'+date+mcpname))


model.compile(loss = 'mse', optimizer = 'adam',
              metrics='mae')


hist = model.fit(x_train, y_train,
                 epochs=300,
                 batch_size=256,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp])



#4. 평가, 예측

result= model.evaluate(x_test, y_test)
print('mse_loss :', result[0])
print('mae_loss :', result[1])

pred = model.predict(x_pred)
r2 = r2_score(y_pred, pred)
print('r2_score :', r2)

def RMSE(a,b) :
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_pred, pred)
print('rmse :', rmse)


print('y_pred :', pred[-3:])


plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.figure(figsize=(9,6))
plt.title('예나')
plt.plot(hist.history['loss'], marker='.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발_로스')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.show()



# mse_loss : 17.41248893737793
# mae_loss : 0.3072885274887085
# r2_score : 0.9974301776799478
# rmse : 0.39604515171333937
# y_pred : [[-3.951447 ]
#  [-3.2507105]
#  [-3.0111127]]   드롭아웃 적용, 노멀라이저




# mse_loss : 16.184242248535156
# mae_loss : 0.2839994430541992
# r2_score : 0.9979587766126897
# rmse : 0.35297050843829364
# y_pred : [[-4.26638  ]
#  [-3.1232798]
#  [-3.2360218]] 드롭아웃 제거, 노멀라이저


# mse_loss : 0.16830530762672424
# mae_loss : 0.21902316808700562
# r2_score : 0.9980274589795217
# rmse : 0.3469813835902129
# y_pred : [[-4.359234 ]
#  [-3.3047986]
#  [-3.2054977]] 드롭아웃 제거, 민맥스 스케일러



# mse_loss : 0.41001108288764954
# mae_loss : 0.24271726608276367
# r2_score : 0.9977067159362174
# rmse : 0.37412960831307307
# y_pred : [[-4.229699 ]
#  [-3.2323225]
#  [-3.2646744]]  드롭아웃 민맥스