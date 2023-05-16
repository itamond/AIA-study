import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
tf.set_random_seed(337)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random.normal([2, 1]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias')

# 2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(xp, w) + b)


# 3-1 컴파일
cost = -tf.reduce_mean(yp*tf.log(hypothesis) + (1-yp) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, yp), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict = {xp:x_data, yp:y_data})

        if step & 200 == 0:
            print(step, cost_val)
            
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {xp:x_data, yp:y_data})
    print(f'hypothesis : \n{h}\n predicted : \n{p}\n accuracy : {a}')
















# xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
# yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([x_data.shape[1], 1]))
# b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]))


# hypothesis = tf.nn.sigmoid(tf.compat.v1.matmul(xp, w) + b)
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yp, logits=hypothesis))

# optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
# train = optimizer.minimize(loss)

# 3-2. 훈련
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# epochs = 10000
# for step in range(epochs):
#     _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={xp:x_data, yp:y_data})
    
# xp2 = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
# y_pred = tf.nn.sigmoid(tf.compat.v1.matmul(xp2, w_val) + b_val)
# y_predict = sess.run([y_pred], feed_dict={xp2:x_data})
# print(y_predict)
# print('acc : ', accuracy_score(y_data, np.round(y_predict[0])))