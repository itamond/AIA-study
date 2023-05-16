import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
tf.set_random_seed(337)

tv = tf.compat.v1

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

x = tv.placeholder(tf.float32, shape=[None, 2])
y = tv.placeholder(tf.float32, shape=[None, 1])

#2. 모델

# model.add(Dense(10, input_shape=2))
w1 = tv.Variable(tf.random.normal([2, 100]), name='weight1')
b1 = tv.Variable(tf.zeros([100]), name='bias1')
layer1 =tv.matmul(x, w1) + b1

# model.add(Dense(7))
w2 = tv.Variable(tf.random.normal([100, 700]), name='weight2')
b2 = tv.Variable(tf.zeros([700]), name='bias2')
layer2 =tv.sigmoid(tv.matmul(layer1, w2) + b2)

# model.add(Dense(1, activation = 'sigmoid'))
w3 = tv.Variable(tf.random.normal([700, 1]), name='weight3')
b3 = tv.Variable(tf.zeros([1]), name='bias3')
hypothesis =tv.sigmoid(tv.matmul(layer2, w3) + b3)

# hypothesis = tv.sigmoid(tv.matmul(x, w) + b)


# 3-1 컴파일
cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_data, y:y_data})

        if step & 20 == 0:
            print(step, cost_val)
            
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x:x_data, y:y_data})
    print(f'hypothesis : \n{h}\n predicted : \n{p}\n accuracy : {a}')




