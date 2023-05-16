import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd


#1. 데이터


path='./_data/ddarung/'     
path_save='./_save/ddarung/'
train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                
test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0)

train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1)    
y = train_csv['count']
y = y.values.reshape(-1, 1)

# x(442, 10) * w(?, ?) + b(?) = y(442, 1)   w의 답 = (10, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337,
)

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# (353, 10) (353, 1)
# (89, 10) (89, 1)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 9])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([9,1]), name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')


#2. 모델
hypothesis = tf.compat.v1.matmul(xp, w) + b  # matmul = 행렬 곱하기

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - yp))   #mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 5000
for step in range(epochs):
    _, loss_val, w_val, b_val = sess.run([train,hypothesis,loss,b], feed_dict={xp:x_train, yp:y_train})
    if step%20 == 0:
        print(step, loss_val, w_val)    




#4. 평가 예측
# R2, mse로 결과 도출

y_pred = sess.run(hypothesis, feed_dict={xp:x_test, yp:y_test})

r2 = r2_score(y_test, y_pred)
print('r2: ', r2)

mse = mean_squared_error(y_test, y_pred)
print('mse: ', mse)

# r2:  0.6201685032039299
# mse:  2763.734928328592