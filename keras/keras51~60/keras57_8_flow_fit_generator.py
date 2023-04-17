#57에 5 카피해서 복붙 x와 y를 xy형태의 이터레이터로 만들어서 fit generator로 훈련시키기

#수치로 제공된 데이터의 증폭

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

np.random.seed(3123)   #시드값 부여 하는법



(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

train_datagen2 = ImageDataGenerator(
    rescale=1./1,
)


augment_size = 40000   #증강, 증폭

#10만개의 데이터로 쓰고싶다. 때문에 6만+x = 10만, 4만개 증가시켜줌
#랜덤하게 인트값을 뽑을거다, 6만개에서 4만개를 뽑을거다.그것을 랜드인덱스라고 부르겠다
# randidx=np.random.randint(60000, size = 40000)
randidx=np.random.randint(x_train.shape[0], size=augment_size)

print(randidx)   #[33731  1990  5122 ... 40892  9013 29985]
print(randidx.shape)   #(40000,)
print(np.min(randidx), np.max(randidx))   #0 59999   4 59996


x_augmented = x_train[randidx].copy()          #x_train에 randidx라는 랜덤 추출 함수를 적용하여 x_augmented로 명명
y_augmented = y_train[randidx].copy()          # copy를 쓰면 x_train와 y_train의 값을 건들이지 않고 새로 만드는것

print(x_augmented.shape, y_augmented.shape)   #(40000, 28, 28) (40000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 
                        x_test.shape[1], 
                        x_test.shape[2],
                        1)

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],
                                  1)



x_augmented = train_datagen.flow(           #y는 넣어줄 필요 없지만 xy를 넣어야 해서 넣음?
    x_augmented, y_augmented, 
    batch_size = augment_size,
    shuffle=False,
).next()[0]            #.next()까지만 하면 x_augmented[0] 이므로  뒤에 [0]을 붙혀 x_augmented[0][0]로 만들어줌.


print(x_augmented)
print(x_augmented.shape)   #(40000, 28, 28, 1)


x_train = np.concatenate((x_train/255., x_augmented))   
y_train = np.concatenate((y_train, y_augmented))   
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_test = x_test/255.
###################x,y 합치기#######################


batch_size = 64

xy_train = train_datagen2.flow(x_train, y_train,
                               batch_size = batch_size, shuffle=True)





# print(x_train.shape, y_train.shape)


# print(np.max(x_train), np.min(x_train))
# print(np.max(x_augmented), np.min(x_augmented))
#255.0 0.0
#1.0 0.0         x_augmented는 datagen에서 스케일 되어있다. 때문에 스케일링 해줘야함.


# 모델 맹그러
# 증폭과 안증폭 성능비교





model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['acc'])

# model.fit(xy_train[:][0], xy_train[:][1],
#           epochs=10,
#           )   #에러

es = EarlyStopping(monitor='val_acc',
                   mode = 'max',
                   patience=30,
                   verbose=1,
                   restore_best_weights=True,
                   )

hist = model.fit_generator(xy_train, epochs=5000,   #x데이터 y데이터 배치사이즈가 한 데이터에 있을때 fit 하는 방법
                    steps_per_epoch=len(xy_train)/batch_size,    #전체데이터크기/batch = 160/5 = 32
                    # validation_split=0.1,
                    shuffle=True,
                    # batch_size = 16,
                    # validation_steps=24,    #발리데이터/batch = 120/5 = 24
                    callbacks=[es],
                    )


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])
# print('acc : ', acc[-1])
# print('val_acc : ', val_acc[-1])



ett = time.time()



from sklearn.metrics import accuracy_score

result = model.evaluate(x_test,y_test)
print('result :', result)

pred = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test, pred)
print('acc:',acc)



# acc : 0.8565  Conv1D적용
# acc : 0.8232  LSTM 적용

# acc: 0.9276 idg 적용