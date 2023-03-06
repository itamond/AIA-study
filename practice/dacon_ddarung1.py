import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader


#.1 데이터
path='./_data/ddarung/'
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
        train_size=0.99,shuffle=True, random_state=750)

#2. 모델구성
model=Sequential()
model.add(Dense(20,input_dim=9))
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer="adam")
model.fit(x_train,y_train,epochs=301,batch_size=10)

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

submission.to_csv(path+'submission0850.csv')

# loss: 722.6207885742188
# RMSE 26.881613397126884

# loss: 838.26513671875
# rmse : 28.952807795104775


#loss: 867.4390869140625
# rmse : 29.452317914345098