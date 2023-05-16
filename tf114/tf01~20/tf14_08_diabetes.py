import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

#1. 데이터


x, y = load_diabetes(return_X_y=True)
# print(x.shape, y.shape) #(442, 10) (442,)
# print(y[:10]) # [151.  75. 141. 206. 135.  97. 138.  63. 110. 310.]

y = y.reshape(-1, 1) # (442, 1)

# x(442, 10) * w(?, ?) + b(?) = y(442, 1)   w의 답 = (10, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337,
)

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# (353, 10) (353, 1)
# (89, 10) (89, 1)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]), name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')


#2. 모델
hypothesis = tf.compat.v1.matmul(xp, w) + b  # matmul = 행렬 곱하기

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - yp))   #mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101
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