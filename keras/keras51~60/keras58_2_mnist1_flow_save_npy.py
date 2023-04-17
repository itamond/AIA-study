from tensorflow.keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)


augument_size = 40000 #증폭

randidx = np.random.randint(x_train.shape[0], size=augument_size) #(60000)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_train = x_train.reshape(x_train.shape[0],
                          x_train.shape[1],
                          x_train.shape[2],
                          1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2],
                        1)
x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],
                                  1)

x_augmented  = train_datagen.flow(x_augmented,
                                  y_augmented,
                                  batch_size=augument_size,
                                  shuffle=False
                                  ).next()[0]
y_augmented  = train_datagen.flow(x_augmented,
                                  y_augmented,
                                  batch_size=augument_size,
                                  shuffle=False
                                  ).next()[1]



x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# xy_train  = test_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=False)
# xy_test = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

np.save('d:/study_data/_save/_npy/58_mnist_train_x.npy', arr=x_train) # train x값
np.save('d:/study_data/_save/_npy/58_mnist_train_y.npy', arr=y_train) # train y값
np.save('d:/study_data/_save/_npy/58_mnist_test_x.npy', arr=x_test) # test x값
np.save('d:/study_data/_save/_npy/58_mnist_test_y.npy', arr=y_test) # test y값

