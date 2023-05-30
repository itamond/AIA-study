#노이즈가 있는 사진(x)로 노이즈가 없는 사진(x)으로 훈련하여 노이즈 제거
import numpy as np
from tensorflow.keras.datasets import mnist

# 1. 데이터

(x_train, _), (x_test, _) = mnist.load_data()
#오토인코더 연습용이기 때문에 y값 필요없음

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.


# 2. 모델

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img) # 꼭 64개로 할 필요는 없다 노드가 적어지면 그만큼 사진이 뿌얘진다.
# encoded = Dense(1024, activation='relu')(input_img)
# encoded = Dense(32, activation='relu')(input_img)
# encoded = Dense(1, activation='relu')(input_img)

# decoded = Dense(784, activation='linear')(encoded)
decoded = Dense(784, activation='sigmoid')(encoded)  #어차피 mnist의 784개 특성들의 값은 스케일링 되어 0~1 사이의 값이다. 따라서 relu linear sigmoid 전부 사용 가능하고 성능만 좋으면 된다.
# decoded = Dense(784, activation='relu')(encoded)

autoencoder = Model(input_img, decoded)

autoencoder.summary()


# 784 -> 64 -> 784 레이어를 거치면서 큰 특성은 크게 남고 작은 특성은 소실됨
# 오토인코더의 고질적인 문제는 데이터가 소실되기 때문에 이미지가 뿌옇게 된다는 것이다.

# 3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss = 'mse', metrics=['acc'])
autoencoder.compile(optimizer='adam', loss = 'mse', metrics=['acc'])

autoencoder.fit(x_train,x_train, epochs = 30, batch_size=128, validation_split=0.2)

# 오토인코더에서는 acc는 상관 없음. loss가 중요. 다만 결과물을 실제로 눈으로 보고 판단해야함. loss를 전적으로 믿을 수는 없다.

# 4. 평가, 예측

# decoded_imgs = np.round(autoencoder.predict(x_test))
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 20
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


