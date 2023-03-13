#iris 데이터셋 = 꽃받침과 이파리로 꽃을 맞추는 데이터셋
#케라스 원 핫 인코딩



import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler


from tensorflow.python.keras.callbacks import EarlyStopping
#1. 데이터

datasets=load_iris()
print(datasets.DESCR)      #판다스 describe
# print(datasets.feature_names)   # 판다스 columns

x = datasets.data
y = datasets['target']
# .reshape(-1,1)    #사이킷런 원핫 인코딩 할때 씀
print(x.shape, y.shape)   #(150, 4) (150,)
print('y의 라벨값 :', np.unique(y))    #unique =  각 라벨값을 표시

########################요 지점에서 원핫을 해야한다###########################
#1. 판다스 
#import pandas as pd
# y = pd.get_dummies(y)
# print(y.shape)

#2. 사이킷런
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y= y.reshape(-1,1)
y=encoder.fit_transform(y).toarray()


#3. 케라스
# y = to_categorical(y)
# print(y.shape)

###################################################

x_train,x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,
    random_state=321,
    train_size=0.8,
    stratify=y              #y를 통계적으로 해라. (y 데이터의 비율따라 나누기) 
    )


# scaler=MinMaxScaler()
scaler=MaxAbsScaler()
# scaler=StandardScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)





print(y_train)
print(np.unique(y_train, return_counts=True))     #unique< 라벨값 표시, return_counts = 갯수까지 표시
#          (array([0, 1, 2]), array([40, 40, 40], dtype=int64))




#2. 모델구성
# model=Sequential()
# model.add(Dense(50,activation='relu', input_dim=4))
# model.add(Dense(40,activation='relu'))
# model.add(Dense(30,activation='relu'))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(3,activation='softmax'))     # softmax 는 각각의 라벨에 대한 확률을 부여하여 출력한다.
                                            #y의 라벨의 갯수(클래스의 갯수)만큼 노드를 잡는다.
                                            #ex 1번이 0.5확률이면 1번이 답.
                                            #다중분류의 마지막 activation은 softmax

input1 = Input(shape=(4,))
dense1 = Dense(50,activation='relu')(input1)
dense2 = Dense(40,activation='relu')(dense1)
dense3 = Dense(30,activation='relu')(dense2)
dense4 = Dense(20,activation='relu')(dense3)
dense5 = Dense(10,activation='relu')(dense4)
outputs1 = Dense(3,activation='softmax')(dense5)

model = Model(inputs=input1, outputs=outputs1)





#3. 컴파일, 훈련                                     #softmax, 다중분류의 loss는 categorical_crossentropy 하나뿐
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])


es = EarlyStopping(monitor='val_acc',
                   mode='max',
                   verbose=1,
                   patience=50)



model.fit(x_train, y_train,
          epochs=1000, batch_size=5,
          validation_split=0.4,
          verbose=1,
          callbacks=[es],
          )

#4. 평가, 예측
# result = model.evaluate(x_test,y_test)    #엄밀히 얘기하면 loss = result이다. 
#                                          #model.evaluate=
#                                          #model.compile에 추가한 loss 및 metrix 모두 result로 표시된다.
#                                          #metrix의 accuracy는 sklearn의 accuracy_score 값과 동일하다.
# print('result :', result)

# y_predict= np.argmax(model.predict(x_test), axis=1)
# print(y_predict)
# y_test = np.argmax(y_test, axis=1)
# acc = accuracy_score(y_test, y_predict)
# print('acc :', acc)


# # accuracy_score를 사용하여 스코어를 빼세요.

results = model.evaluate(x_test, y_test)
print(results)
print('loss :', results[0])    # 리절트의 0번째
print('acc :', results[1])    #리절트의 1번째
y_pred = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)  # axis = 1 각 행에 있는 '열' 끼리 비교
y_pred = np.argmax(y_pred, axis=1)

# print(y_test)
acc = accuracy_score(y_test_acc, y_pred)

print('acc :', acc)




# loss : 0.09008350968360901
# acc : 0.9666666388511658
# acc : 0.9666666666666667    maxabs스케일러 적용