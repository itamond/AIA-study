#수치로 제공된 데이터의 증폭

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

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

augment_size = 40000   #증강, 증폭

#10만개의 데이터로 쓰고싶다. 때문에 6만+x = 10만, 4만개 증가시켜줌
#랜덤하게 인트값을 뽑을거다, 6만개에서 4만개를 뽑을거다.그것을 랜드인덱스라고 부르겠다
# randidx=np.random.randint(60000, size = 40000)
randidx=np.random.randint(x_train.shape[0], size=augment_size)

print(randidx)   #[33731  1990  5122 ... 40892  9013 29985]
print(randidx.shape)   #(40000,)
print(np.min(randidx), np.max(randidx))   #0 59999   4 59996