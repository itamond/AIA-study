from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import datetime
#1. 데이터

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
     

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# (60000, 28, 28) (60000,)
# (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))



x_train = x_train.reshape(60000, 28*28)/255.
x_test = x_test.reshape(10000, 28*28)/255.         #reshape와 scaling 동시에 하기.


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath='./_save/cnn/cifar10/'
filename='{epoch:04d}-{val_acc:.4f}.hdf5'



#2. 모델구성

input1 = Input(shape=(28, 28, 1))
conv1 = Conv2D(64, (3,3),
               padding='same', 
               activation='relu')(input1)
conv2 = Conv2D(64, (3,3),
               padding='same', 
               activation='relu')(conv1)
mp1 = MaxPooling2D()
pooling1 = mp1(conv2)
conv3 = Conv2D(64, (3,3),
               padding='same', 
               activation='relu')(conv2)
pooling2 = mp1(conv3)
flat1 = Flatten()(pooling2)
dense1 = Dense(256,activation='relu')(flat1)
dense2 = Dense(256,activation='relu')(dense1)
dense3 = Dense(128,activation='relu')(dense2)
output1 = Dense(10,activation='softmax')(dense3)
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
                 epochs = 5000,
                 batch_size =32,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es,mcp])

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