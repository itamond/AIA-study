from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA

(x_train,_),(x_test,_) = mnist.load_data()   #파이썬 기초 문법. _ 언더바 입력시 메모리에 할당하지 않음

#28x28의 이미지의 경우 784개의 컬런이라고 볼 수도 있다. (60000,784)
#이미지의 데이터는 쓸모없는 0의 데이터가 많다. 이를 PCA를 통해 압축시킬 수 있다.
#EVR을 통해 몇개의 컬런을 지워야 성능이 좋을지 확인한다.

# x = np.concatenate((x_train,x_test),axis=0) #train과 test를 붙히는 방법
x = np.append(x_train,x_test, axis=0)  #둘 다 가능하다.
# print(x.shape)
#(70000, 28, 28)



###########실습############
# pca를 통해 0.95 이상인 n_components는 몇개?
# 0.95 몇개?
# 0.99 몇개?
# 0.999 몇개?
# 1.0 몇개?
x = x.reshape(-1,28*28)

pca = PCA(n_components=784)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
cumsum = np.cumsum(pca_EVR)
print(cumsum)


print(np.argmax(cumsum >= 0.95) + 1)    #154 
print(np.argmax(cumsum >= 0.99) + 1)    #331 
print(np.argmax(cumsum >= 0.999) + 1)    #486 
print(np.argmax(cumsum >= 1.0) + 1)    #713 숫자는 0부터 시작하므로 1 더해준다.


# d95 = np.argmax(cumsum >= 0.95) + 1
# d99 = np.argmax(cumsum >= 0.99) + 1
# d999 = np.argmax(cumsum >= 0.999) + 1
# d100 = np.argmax(cumsum == 1.0) + 1


# print(f"0.95 이상의 n_components 갯수 : {d95}")
# print(f"0.99 이상의 n_components 갯수 : {d99}")
# print(f"0.999 이상의 n_components 갯수 : {d999}")
# print(f"1.0 이상의 n_components 갯수 : {d100}")

