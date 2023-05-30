import numpy as np
from tensorflow.keras.datasets import mnist

# 1. 데이터

(x_train, _), (x_test, _) = mnist.load_data()
#오토인코더 연습용이기 때문에 y값 필요없음

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.3, size = x_train.shape) #정규분포 형식의 임의의 값
x_test_noised = x_test + np.random.normal(0, 0.3, size = x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)

print(np.max(x_train_noised), np.min(x_train_noised))
print(np.max(x_test_noised), np.min(x_test_noised))
# 1.5163782081088089 -0.5382002821326444
# 1.4551918233222643 -0.5110962399532574

x_train_noised = np.clip(x_train_noised, a_min = 0, a_max = 1)
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1) #최저값 0 최고값 1로 고정시키는 함수

print(np.max(x_train_noised), np.min(x_train_noised))
print(np.max(x_test_noised), np.min(x_test_noised))
# 1.0 0.0
# 1.0 0.0



from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
# encoded = Dense(64, activation='relu')(input_img) # 꼭 64개로 할 필요는 없다 노드가 적어지면 그만큼 사진이 뿌얘진다.
# encoded = Dense(1024, activation='relu')(input_img)
# encoded = Dense(32, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(input_img)
# encoded = Dense(1, activation='relu')(input_img)

# decoded = Dense(784, activation='linear')(encoded)
decoded = Dense(784, activation='sigmoid')(encoded)  #어차피 mnist의 784개 특성들의 값은 스케일링 되어 0~1 사이의 값이다. 따라서 relu linear sigmoid 전부 사용 가능하고 성능만 좋으면 된다.
# decoded = Dense(784, activation='relu')(encoded)

autoencoder = Model(input_img, decoded)

autoencoder.summary()


# 3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss = 'mse', metrics=['acc'])
autoencoder.compile(optimizer='adam', loss = 'mse')

autoencoder.fit(x_train, x_train, epochs = 30, batch_size=128, validation_split=0.2)

# 오토인코더에서는 acc는 상관 없음. loss가 중요. 다만 결과물을 실제로 눈으로 보고 판단해야함. loss를 전적으로 믿을 수는 없다.

# 4. 평가, 예측

# decoded_imgs = np.round(autoencoder.predict(x_test))
decoded_imgs = autoencoder.predict(x_test_noised)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noised[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
