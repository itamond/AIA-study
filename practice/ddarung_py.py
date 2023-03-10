import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
#.1 데이터
path='./_data/ddarung/'
path_save='./_save/ddarung/'
train_csv=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)
test_csv=pd.read_csv(path+'test.csv',index_col=0) 

print(train_csv)
print(train_csv.shape) #(1459, 10)


print(test_csv)
print(test_csv.shape) #(715, 9)

# print(train_csv.columns)
# print(train_csv.info())
# print(train_csv.describe())

###결측치 처리하기 1. 제거하기 ###
#print(train_csv.isnull().sum()) #널의 갯수를 더해라 /컬럼 당 결측치의 갯수를 확인 할 수 있다.
train_csv=train_csv.dropna()
#print(train_csv.isnull().sum())
#print(train_csv.shape) #(1328, 10) 130개 정도 사라졌음
#####
# test_csv=test_csv.fillna(0)

x=train_csv.drop(['count'],axis=1)
# print(x)
# print(x.columns)
# print(x.shape) #(1459, 9)


y=train_csv['count']
# print(y)
# print(y.shape) #(1459,)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.98,shuffle=True, random_state=5330)


# scaler = MinMaxScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# scaler.fit(x_train)     #scaler의 fit의 범위가 x_train이라는 뜻. (x_train을 0~1로 변환)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)  



#2. 모델구성
model=Sequential()
model.add(Dense(20,input_dim=9))
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer="adam")

es = EarlyStopping(monitor='val_loss',
                   patience=100,
                   mode='min',
                   restore_best_weights=True,
                   verbose=1,
                   )

model.fit(x_train,y_train,
          epochs=3000,
          batch_size=32,
          validation_split=0.2,
          callbacks=[es],
          )

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse=RMSE(y_test,y_predict)
print("rmse :",rmse)



y_submit=model.predict(test_csv)

submission=pd.read_csv(path+'submission.csv',index_col=0)


submission['count']=y_submit
#카운트에 y서브밋을 넣고

submission.to_csv(path_save+'submission_v12.csv')
