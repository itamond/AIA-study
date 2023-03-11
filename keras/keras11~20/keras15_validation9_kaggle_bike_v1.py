# # 데이콘 따릉이 문제풀이
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error , mean_absolute_error
import pandas as pd 
from tensorflow.python.keras.callbacks import EarlyStopping


#1. 데이터
path='./_data/kaggle_bike/'
path_save='./_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                 
                                                     
test_csv = pd.read_csv(path+'test.csv',
                       index_col=0)


print(train_csv.columns)
print(test_csv.columns)

# print(train_csv.shape)


# print(train_csv)  

# test_csv = pd.read_csv(path + 'test.csv',
#                        index_col=0)

# print(test_csv)
# print(test_csv.shape)

#=================================================================

print(train_csv.columns)



print(train_csv.info())    





print(type(train_csv))   


######################################결측치 처리###############################################

#print(train_csv.isnull())   # isnull   -> 데이터가 null값인가요? 하고 물어보는 함수
print(train_csv.isnull().sum())  
train_csv = train_csv.dropna()   
print(train_csv.isnull().sum())
print(train_csv.info())          
print(train_csv.shape)          




############################train_csv 데이터에서 x와 y를 분리(매우 중요)#########################
x = train_csv.drop(['count', 'registered', 'casual'], axis=1)   


y = train_csv['count']

###############################################################################################




x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    shuffle=True,
    train_size=0.95,
    random_state=138
)

print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape) 





#2. 모델구성
model = Sequential()
model.add(Dense(6, input_dim=8))
model.add(Dense(8, activation='relu')) #↓ 값을 전달할때 값을 조절하는 함수 activation (활성화 함수) , 다음에 전달하는 내용을 *한정*시킨다.   
model.add(Dense(6, activation='relu')) # Relu -> 0 이상의 값은 양수, 0이하의 값은 0이 된다. 항상 양수로 만드는 활성화 함수
model.add(Dense(8, activation='relu'))   # 회귀모델->선형회귀. linear는 디폴트 활성화 함수
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss',
                   patience=100,
                   mode='min',
                   restore_best_weights=True,
                   verbose=1,
                   )

model.fit(x_train, y_train,
          epochs= 2400,
          batch_size=32,
          verbose=1,
          validation_split=0.2,
          callbacks=[es])

#4. 평가, 예측

loss= model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2=r2_score(y_test, y_predict)
print('r2 :', r2)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)

print('rmse :', rmse)



y_submit = model.predict(test_csv)

submission = pd.read_csv(path+'sampleSubmission.csv', index_col=0)

submission['count'] = y_submit



submission.to_csv(path_save + 'kagglebike_validationv1.csv')




# y_predict= model.predict(x_test)


#rmse : 152.73700287967176  random 221
#rmse : 140.02564272362673

# r2 : 0.419029901125878
# rmse : 134.51105899209864

#r2 : 0.5024578827750664
# rmse : 125.08617857203974



## Validation = 검증, 확인   << 이 행동은 fit에서 수행

#Validation->모의고사  predict(test_csv)->수능

#r2 : 0.31177054592678666
#rmse : 145.78497245205958