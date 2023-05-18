# pip install keras==1.2.2
# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
import keras
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
import time


tv = tf.compat.v1

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)/255.
x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
output_node = y_train.shape[1]

xp = tv.placeholder('float', shape = [None, 28, 28, 1])
yp = tv.placeholder('float', shape = [None, output_node])

hidden_node1 = 64
hidden_node2 = 32

w1 = tv.Variable(tv.random.normal([3, 3, 1, hidden_node1]), name='weight1')     # kernel_size = (3, 3), channels = 1, filters = 64
b1 = tv.Variable(tv.zeros(hidden_node1), name='bias1')
layer1 = tf.nn.conv2d(xp, w1, strides=[1,1,1,1], padding='SAME') + b1
L1_maxpool = tf.nn.max_pool2d(layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# (N, 14, 14, 64)

w2 = tv.Variable(tv.random.normal([3, 3, hidden_node1, hidden_node2]), name='weight2')
b2 = tv.Variable(tv.zeros([hidden_node2]), name='bias2')
layer2 = tf.nn.conv2d(L1_maxpool, w2, padding='SAME') + b2
# (N, 14, 14, 10)

flat_layer = tf.reshape(layer2, [-1, layer2.shape[1]*layer2.shape[2]*layer2.shape[3]])

w3 = tv.Variable(tv.random.normal([int(flat_layer.shape[1]), output_node]), name='weight3')
b3 = tv.Variable(tv.zeros([output_node]), name='bias3')
hypothesis = tf.nn.softmax(tv.matmul(flat_layer, w3) + b3)

loss = tf.reduce_mean(-tf.reduce_sum(yp*tf.nn.log_softmax(hypothesis), axis=1))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=yp, logits=hypothesis))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

batch_size = 1000
total_batch = int(x_train.shape[0]/batch_size)      # 60000 / 100 = 600
epochs = 300

sess = tv.Session()
sess.run(tv.global_variables_initializer())

strat_time = time.time()
for step in range(epochs):
    sum_of_batch_loss = 0
    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        loss_val, _, w_val, b_val = sess.run([loss, train, w2, b2], feed_dict={xp:x_train[start:end], yp:y_train[start:end]})
        sum_of_batch_loss += loss_val / total_batch
    print(f'epoch : {step + 1}, loss : {sum_of_batch_loss}')
print('train complete')
end_time = time.time()
y_pred = sess.run(hypothesis, feed_dict={xp:x_test})
print('acc : ', accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)), 'interval time : ', end_time - strat_time)