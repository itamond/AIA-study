#y값이 실수가 아닌 0과 1일때의 모델링

#이진분류인 데이터

#나중에는 컬런에 대한 분석을 해야한다

import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping


#1. 데이터

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_set = pd.read_csv(path + 'train.csv',index_col=0)
test_set = pd.read_csv(path + 'test.csv',index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)



# print(datasets)
print(train_set.describe)     # [652 rows x 9 columns]
#numpy의 디스크라이브, pd는 .describe()
print(train_set.columns) # 판다스 : columns()



x=train_set.drop(['Outcome'], axis=1)
y=train_set['Outcome']
print(x.shape, y.shape)    #(652, 8) (652,)





# x = datasets['data']     #딕셔너리의 key
# y = datasets.target
# # print(x.shape, y.shape)  #(569, 30) (569,)

# # print(datasets.isnull().sum())
# # print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,
    random_state=7909,
    test_size=0.2,
    stratify=y,
)

#2. 모델구성

model = Sequential()
model.add(Dense(10, activation='linear',input_dim=8))
model.add(Dense(30,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(1,activation='sigmoid'))  


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'],    # 훈련 과정에 accuracy를 추가.   hist.history에 다 기록됨. 대괄호=리스트
              )                             #대괄호 안에 다른 지표도 추가 가능(list이기 때문에)
                                            #metrics = 훈련에 영향을 주지않음. 지표확인용, 풀네임 줄임말 상관 없음
                                            
                                            
es =EarlyStopping(monitor='val_accuracy',
                  mode='max',
                  patience=100,
                  restore_best_weights=True,
                  verbose=1,
                  )

hist = model.fit(x_train,y_train,
                 epochs=1000,
                 batch_size=20,
                 validation_split=0.2,
                 verbose=1,
                 callbacks=[es],
                 )


#4. 평가, 예측
result = model.evaluate(x_test,y_test)    #엄밀히 얘기하면 loss = result이다. 
                                         #model.evaluate=
                                         #model.compile에 추가한 loss 및 metrix 모두 result로 표시된다.
                                         #metrix의 accuracy는 sklearn의 accuracy_score 값과 동일하다.
print('result :', result)

y_predict= np.round(model.predict(x_test))


# r2= r2_score(y_test, y_predict)
# print('r2 :', r2)

def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_test, y_predict)
print('rmse :', rmse)

#평소처럼 수치가 아닌 0이냐 1이냐를 맞춰야한다면 accuracy_score 사용 



from sklearn.metrics import accuracy_score, r2_score
acc = accuracy_score(y_test, y_predict)
print('acc :', acc)

y_submit = np.round(model.predict(test_set))
submission['Outcome']=y_submit
submission.to_csv(path_save+'submission_sigmoid9.csv')





# result : 0.742574155330658
# r2 : -0.38624338624338606
# rmse : 0.5795497067512154
# acc : 0.6641221374045801


# result : 0.6706214547157288
# r2 : -0.2917267917267916
# rmse : 0.5594435621549695
# acc : 0.6870229007633588

# result : 0.6403940320014954
# r2 : -0.3135501355013548
# rmse : 0.5314534298427315
# acc : 0.7175572519083969

# result : 0.6238195300102234
# r2 : -0.24254742547425456
# rmse : 0.5168902906024488
# acc : 0.732824427480916
    
    
# r2 : -0.008823529411764675
# rmse : 0.4629100498862757
# acc : 0.7857142857142857


# result : 0.5473717451095581
# rmse : 0.5248906591678238
# acc : 0.7244897959183674

# result : 0.5648024082183838
# rmse : 0.5075192189225523
# acc : 0.7424242424242424


# result : 0.5427650213241577
# rmse : 0.4605661864718383
# acc : 0.7878787878787878

# result : 1.4083452224731445
# rmse : 0.5075192189225523
# acc : 0.7424242424242424   sub 3


# result : 1.872929334640503
# rmse : 0.5168902906024488
# acc : 0.732824427480916 sbu4


# result : 0.7009929418563843
# rmse : 0.5904735420248883
# acc : 0.6513409961685823   sub5 



# result : 0.5392670035362244
# rmse : 0.5
# acc : 0.75 sub5

# result : 0.5519040822982788
# rmse : 0.47380354147934284
# acc : 0.7755102040816326 sub6


# result : 0.5715904235839844
# rmse : 0.4573660169594892
# acc : 0.7908163265306123   sub7


## sig 8