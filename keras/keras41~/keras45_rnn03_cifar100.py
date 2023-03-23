from keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import datetime
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)




x_train = x_train.reshape(50000, 32*32*3)/255.
x_test = x_test.reshape(10000, 32*32*3)/255.         #reshape와 scaling 동시에 하기.


x_train = x_train.reshape(50000, 32, 32*3)
x_test = x_test.reshape(10000, 32, 32*3)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath='./_save/cnn/cifar10/'
filename='{epoch:04d}-{val_acc:.4f}.hdf5'



#2. 모델구성


input1 = Input(shape=(32, 32*3))
LSTM1 = LSTM(5)(input1)
dense1 = Dense(256,activation='relu')(LSTM1)
dense2 = Dense(256,activation='relu')(dense1)
dense3 = Dense(128,activation='relu')(dense2)
output1 = Dense(100,activation='softmax')(dense3)
model = Model(inputs=input1, outputs=output1)
model.summary()




#3. 컴파일, 훈련
import time
start_time = time.time()


es = EarlyStopping(monitor='val_acc',
                   mode='auto',
                   restore_best_weights=True,
                   patience=100)

mcp = ModelCheckpoint(monitor='val_acc',
                      mode='auto',
                      save_best_only=True,
                      verbose=1,
                      filepath = ''.join([filepath+'_k33_2_'+date+'_'+filename]))


model.compile(loss='categorical_crossentropy', optimizer='adam',    
              metrics=['acc'])

hist = model.fit(x_train,y_train,
                 epochs = 50,
                 batch_size =3200,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es])

end_time = time.time()

#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_argm = np.argmax(y_test, axis=1)
acc = accuracy_score(y_argm, y_pred)

print('acc :', acc)

print('걸린시간 : ', round(end_time - start_time,2),'초')   


# result : [2.5717358589172363, 0.39660000801086426]
# acc : 0.3966  
# 걸린시간 :  235.59

# result : [2.4068357944488525, 0.3970000147819519]
# acc : 0.397
# 걸린시간 :  291.8 초