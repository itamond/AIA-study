# upsampling -> 이미지에 대한 증폭.
# maxpooling과 반대로 이미지를 증폭시키는 방법. 각 커널을 통하는 값과 인접한 커널 값의 bogan(선형보간 interpolate)을 통해 이미지를 증폭시킨다.

import numpy as np
from tensorflow.keras.datasets import mnist

# 1. 데이터

(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.5, size = x_train.shape) #정규분포 형식의 임의의 값
x_test_noised = x_test + np.random.normal(0, 0.5, size = x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)

print(np.max(x_train_noised), np.min(x_train_noised))
print(np.max(x_test_noised), np.min(x_test_noised))


x_train_noised = np.clip(x_train_noised, a_min = 0, a_max = 1)
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1) #최저값 0 최고값 1로 고정시키는 함수

print(np.max(x_train_noised), np.min(x_train_noised))
print(np.max(x_test_noised), np.min(x_test_noised))


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten



def autoencoder():
    # 인코더
    model = Sequential()
    model.add(Conv2D(32, 3, activation = 'relu', padding = 'same', input_shape = (28, 28, 1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, 3, activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D()) # (n, 7, 7, 8)

    # 디코더
    model.add(Conv2D(16, 3, activation= 'relu', padding= 'same'))
    model.add(UpSampling2D())
    model.add(Conv2D(32, 3, activation= 'relu', padding= 'same'))
    model.add(UpSampling2D())
    model.add(Conv2D(1, 3, activation= 'sigmoid', padding= 'same'))
    # model.summary()


    return model

model = autoencoder()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', )

model.fit(x_train_noised, x_train, epochs = 30, batch_size = 128, validation_split = 0.2)

# 4. 평가, 예측

# decoded_imgs = np.round(autoencoder.predict(x_test))
decoded_imgs = model.predict(x_test_noised)

###########################################################################


from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20, 7))


# 이미지 다섯개 무작위 선정
random_images = random.sample(range(decoded_imgs.shape[0]), 5)

# 원본 이미지를 상단에 그린다

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
        

# 노이즈 이미지

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


# 오토인코더 출력 이미지

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]) :
    ax.imshow(decoded_imgs[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()