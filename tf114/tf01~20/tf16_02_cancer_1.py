from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
tf.compat.v1.set_random_seed(337)
import numpy as np

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
y = y.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, shuffle=True, random_state=337)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, x.shape[1]])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([x.shape[1], 1]))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]))

hypothesis = tf.nn.sigmoid(tf.compat.v1.matmul(xp, w) + b)
# loss = -tf.reduce_mean(yptf.log_sigmoid(hypothesis) + (1-yp)tf.log_sigmoid(1-hypothesis))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yp, logits=hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

# 3-2. 훈련
epochs = 1000
with tf.compat.v1.Session() as sess:
    for step in range(epochs):
        sess.run(tf.global_variables_initializer())
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={xp:x_train, yp:y_train})
        # print(w_val[0][0])
    # 4. 평가, 예측
    xp2 = tf.compat.v1.placeholder(tf.float32, shape=[None, x.shape[1]])
    y_pred = tf.nn.sigmoid(tf.compat.v1.matmul(xp2, w_val) + b_val)
    y_predict = sess.run([y_pred], feed_dict={xp2:x_test})
    # print(np.round(y_predict))
    print('acc : ', accuracy_score(y_test, np.round(y_predict[0])))