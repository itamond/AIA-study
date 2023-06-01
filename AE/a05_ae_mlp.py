# ml의 m33참조. pca를 사용하여 feature를 줄여도 성능은 크게 차이 없지만 속도는 빨라진다. 
# PCA 성능에 따른 노드 갯수 변화를 유도해보고, 최종 결과물을 확인해보자
import numpy as np
from tensorflow.keras.datasets import mnist

# 1. 데이터

(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

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
from tensorflow.keras.layers import Dense, Input

hidden_layer_size = [512, 256, 512, 256, 512]

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units = hidden_layer_size[0], input_shape = (784,)))
    model.add(Dense(units = hidden_layer_size[1]))
    model.add(Dense(units = hidden_layer_size[2]))
    model.add(Dense(units = hidden_layer_size[3]))
    model.add(Dense(units = hidden_layer_size[4]))
    model.add(Dense(784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=hidden_layer_size) #PCA 95% 성능


# 3. 컴파일, 훈련
model.compile(optimizer = 'adam', loss = 'mse', )

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