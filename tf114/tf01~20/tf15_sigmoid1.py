from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
tf.compat.v1.set_random_seed(337)
import numpy as np
x_data = np.array([[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]])
y_data = np.array([[0], [0], [0], [1], [1], [1]])

# [실습] 시그모이드 빼고 만들기

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([x_data.shape[1], 1]))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]))

hypothesis = tf.compat.v1.matmul(xp, w) + b

loss = tf.reduce_mean(tf.square(hypothesis - yp))

optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)


# 3-2. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


epochs = 3001

for step in range(epochs):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={xp:x_data, yp:y_data})
    if step<20:
        print(loss_val)


# 4. 평가, 예측

xp2 = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y_pred = tf.compat.v1.matmul(xp2, w_val) + b_val
y_predict = sess.run([y_pred], feed_dict={xp2:x_data})


print('r2 : ', r2_score(y_data, y_predict[0]))
print('mse : ', mean_squared_error(y_data, y_predict[0]))