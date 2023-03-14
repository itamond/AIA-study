import numpy as np
import pandas as pd
import datetime
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
#1. 데이터

path = './_data/ddarung/'
path_save = './_save/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission = pd.read_csv(path + 'submission.csv', index_col = 0)
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/keras28/09/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


# print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
# print(train_csv.isnull().sum())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.98,
    random_state=321,
    shuffle=True
)


# scaler=MinMaxScaler()
# scaler=MaxAbsScaler()
scaler=StandardScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)




# print(x_train.shape)

#2. 모델구성

# model=Sequential()
# model.add(Dense(50, input_dim=9))
# model.add(Dense(30))
# model.add(Dense(20))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(1))

input1 = Input(shape=(9,))
dense1 = Dense(50)(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(30)(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(20)(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(10, activation='relu')(drop3)
drop4 = Dropout(0.3)(dense4)
output1 = Dense(1)(drop4)
model=Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   restore_best_weights=True,
                   patience=50,
                   verbose=1)

mcp=ModelCheckpoint(monitor='val_loss',
                    mode='auto',
                    save_best_only=True,
                    verbose=1,
                    filepath=''.join([filepath,'k28_09_',date,'_',filename]),
                    )


model.compile(loss = 'mse', optimizer='adam')
hist = model.fit(x_train, y_train,
                 epochs=1000,
                 batch_size=10,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es,mcp])


#4. 평가, 예측
def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a,b))





loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

rmse = RMSE(y_test, y_predict)
print('rmse :', rmse)

y_submit = model.predict(test_csv)
submission['count']=y_submit

submission.to_csv(path_save + 'submission_drop_1.csv')


# # 시각화

# plt.rcParams['font.family']= 'Malgun Gothic'
# plt.figure(figsize=(9,6))
# plt.title('개쩌는따릉이그래프')
# plt.plot(hist.history['loss'],c='red', marker='.', label='로스')
# plt.plot(hist.history['val_loss'],c='blue', marker='.', label='발_로스')
# plt.grid()
# plt.legend()
# plt.show()



# loss : 3024.846923828125
# r2 : 0.536186690865533
# rmse : 54.998610288874765


# loss : 2885.412353515625
# r2 : 0.5472856240253858
# rmse : 53.71603522314404



# loss : 1714.6185302734375
# r2 : 0.7144258740372252
# rmse : 41.407950818712166


# loss : 2095.519287109375
# r2 : 0.6509857831352365
# rmse : 45.77684631618882  맥스앱스 스케일러 적용


# loss : 2147.805419921875
# r2 : 0.6422774457259992
# rmse : 46.34442223297733 민맥스 스케일러 


# loss : 1425.8115234375
# r2 : 0.7625274132074109
# rmse : 37.75991884517939  스탠다드 스케일러

# loss : 2025.1976318359375
# r2 : 0.6626981312547544
# rmse : 45.00219367142776  드롭아웃 적용