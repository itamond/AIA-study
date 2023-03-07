
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터

path = './_data/ddarung/'
path_save = './_save/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path+'test.csv', index_col=0)
submission = pd.read_csv(path+'submission.csv', index_col=0)



# print(train_set.shape)   #(1459, 10)
train_set = train_set.dropna()


x = train_set.drop(['count'], axis=1)
y = train_set['count']
# print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.8, random_state=332)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size =0.5, random_state=223)




#2. 모델구성
model=Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100, batch_size=100, validation_data=(x_val, y_val))


#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2= r2_score(y_test, y_predict)
print('r2 :', r2)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse=RMSE(y_test, y_predict)
print('rmse :', rmse)

y_submit = model.predict(test_set)
submission['count']= y_submit
submission.to_csv(path_save + 'submission_validation_v3.csv')



#loss: 990.2715454101562
#rmse : 31.46858069677725   v1 validation 적용



#loss : 3081.24365234375
# r2 : 0.5225717070839723
# rmse : 55.50895164987975   v2



# loss : 5573.13720703125
# r2 : 0.32590150084208125
# rmse : 74.65344516542235    v3