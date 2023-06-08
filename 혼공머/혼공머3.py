from sklearn.model_selection import train_test_split
import numpy as np


fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = np.column_stack((fish_length, fish_weight))


# print(fish_data[:5])
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

# print(fish_target)
x_train, x_test, y_train, y_test = train_test_split(fish_data, fish_target, random_state=42, stratify=fish_target)

# print(x_train.shape, x_test.shape)  #(36, 2) (13, 2)

# print(y_test)

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(x_train, y_train)
# print(kn.score(x_test, y_test))

# print(kn.predict([[25,150]]))

distances, indexes = kn.kneighbors([[25, 150]])

import matplotlib.pyplot as plt
# plt.scatter(x_train[:,0], x_train[:,1])
# plt.scatter(25, 150, marker='^')
# plt.scatter(x_train[indexes, 0], x_train[indexes,1], marker='D')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()


# https://matplotlib.org/stable/api/markers_api.html 
# 자연어처리, 갠, 디퓨전  

mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

print(mean, std)

train_scaled = (x_train - mean) / std

new = ([25, 150] - mean) / std

kn.fit(x_train, y_train)

test_scaled = (x_test - mean) / std

kn.score(x_test, y_test)

print(kn.predict([new]))

distances, indexes = kn.kneighbors([new])

plt.scatter(x_train[:,0], x_train[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(x_train[indexes, 0], x_train[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()  