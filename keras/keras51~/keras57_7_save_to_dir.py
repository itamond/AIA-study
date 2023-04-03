#수치로 제공된 데이터의 증폭

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time

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

augment_size = 1000   #증강, 증폭

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

###################################################
# x_augmented = train_datagen.flow(           #y는 넣어줄 필요 없지만 xy를 넣어야 해서 넣음?
#     x_augmented, y_augmented, 
#     batch_size = augment_size,
#     shuffle=False,
# )
#플로우 통과하면 이터레이터가 됨.
#<keras.preprocessing.image.NumpyArrayIterator object at 0x000001C1711BB0D0>
# print(x_augmented)
# print(x_augmented[0][0].shape)
#################이렇게 써도 됨######################


stt = time.time()
print("시작~ ")
x_augmented = train_datagen.flow(           #y는 넣어줄 필요 없지만 xy를 넣어야 해서 넣음?
    x_augmented, y_augmented, 
    batch_size = augment_size,
    shuffle=False,
    save_to_dir='d:/temp/'
).next()[0]            #.next()까지만 하면 x_augmented[0] 이므로  뒤에 [0]을 붙혀 x_augmented[0][0]로 만들어줌.

ett = time.time()

print("끝 ")

ett1 = ett-stt
print(augment_size, "개 증폭에 걸린 시간 :", round(ett1,2), '초')

# print(x_augmented)
# print(x_augmented.shape)   #(40000, 28, 28, 1)


# x_train = np.concatenate((x_train/255., x_augmented))   
# y_train = np.concatenate((y_train, y_augmented))   
# x_test = x_test/255.

# print(x_train.shape, y_train.shape)


# print(np.max(x_train), np.min(x_train))
# print(np.max(x_augmented), np.min(x_augmented))
#255.0 0.0
#1.0 0.0         x_augmented는 datagen에서 스케일 되어있다. 때문에 스케일링 해줘야함.