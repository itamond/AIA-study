from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.datasets import cifar100
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet101, InceptionV3, InceptionResNetV2, DenseNet121, EfficientNetB0,Xception
from tensorflow.keras.applications.vgg19 import preprocess_input
import time

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

res = Xception(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))
# res.trainable = False  

model = Sequential()
model.add(res)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128, callbacks=[es, reduce_lr], validation_split=0.2)
end = time.time()

loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)


print('걸린시간 : ', end - start)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))

# 걸린시간 :  1180.4663059711456
# loss :  2.3004
# accuracy :  0.4114
