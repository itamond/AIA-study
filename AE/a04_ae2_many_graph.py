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

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units = hidden_layer_size, input_shape = (784,)))
    model.add(Dense(784, activation='sigmoid'))
    return model

model_01 = autoencoder(hidden_layer_size=1)
model_08 = autoencoder(hidden_layer_size=8)
model_32 = autoencoder(hidden_layer_size=32)
model_64 = autoencoder(hidden_layer_size=64)
model_154 = autoencoder(hidden_layer_size=154) # PCA 95% 성능 
model_331 = autoencoder(hidden_layer_size=331) # PCA 99% 성능
model_486 = autoencoder(hidden_layer_size=486) # PCA 99.9% 성능
model_713 = autoencoder(hidden_layer_size=713) # PCA 100% 성능


print('================== node 1개 시작 =====================')
model_01.compile(optimizer = 'adam', loss = 'mse', )
model_01.fit(x_train_noised, x_train, epochs = 3, batch_size = 128, validation_split = 0.2)
decoded_imgs_01 = model_01.predict(x_test_noised)


print('================== node 8개 시작 =====================')
model_08.compile(optimizer = 'adam', loss = 'mse', )
model_08.fit(x_train_noised, x_train, epochs = 3, batch_size = 128, validation_split = 0.2)
decoded_imgs_08 = model_08.predict(x_test_noised)


print('================== node 32개 시작 =====================')
model_32.compile(optimizer = 'adam', loss = 'mse', )
model_32.fit(x_train_noised, x_train, epochs = 3, batch_size = 128, validation_split = 0.2)
decoded_imgs_32 = model_32.predict(x_test_noised)


print('================== node 64개 시작 =====================')
model_64.compile(optimizer = 'adam', loss = 'mse', )
model_64.fit(x_train_noised, x_train, epochs = 3, batch_size = 128, validation_split = 0.2)
decoded_imgs_64 = model_64.predict(x_test_noised)


print('================== node 154개 시작 =====================')
model_154.compile(optimizer = 'adam', loss = 'mse', )
model_154.fit(x_train_noised, x_train, epochs = 3, batch_size = 128, validation_split = 0.2)
decoded_imgs_154 = model_154.predict(x_test_noised)


print('================== node 331개 시작 =====================')
model_331.compile(optimizer = 'adam', loss = 'mse', )
model_331.fit(x_train_noised, x_train, epochs = 3, batch_size = 128, validation_split = 0.2)
decoded_imgs_331 = model_331.predict(x_test_noised)


print('================== node 486개 시작 =====================')
model_486.compile(optimizer = 'adam', loss = 'mse', )
model_486.fit(x_train_noised, x_train, epochs = 3, batch_size = 128, validation_split = 0.2)
decoded_imgs_486 = model_486.predict(x_test_noised)


print('================== node 713개 시작 =====================')
model_713.compile(optimizer = 'adam', loss = 'mse', )
model_713.fit(x_train_noised, x_train, epochs = 3, batch_size = 128, validation_split = 0.2)
decoded_imgs_713 = model_713.predict(x_test_noised)

# 4. 평가, 예측

# decoded_imgs = np.round(autoencoder.predict(x_test))

###########################################################################


from matplotlib import pyplot as plt
import random

fig, axes = plt.subplots(9, 5, figsize=(15, 15))

random_imgs = random.sample(range(decoded_imgs_01.shape[0]), 5)
outputs = [x_test, decoded_imgs_01, decoded_imgs_08, decoded_imgs_32, decoded_imgs_64,
           decoded_imgs_154, decoded_imgs_331, decoded_imgs_486, decoded_imgs_713]

outputs_name = ['x', '01', '08', '32', '64',
           '154', '331', '486', '713']

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        if col_num ==0:
            ax.set_ylabel(outputs_name[row_num], size=20, rotation=0)
            ax.yaxis.set_label_coords(-0.2, 0.5)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()