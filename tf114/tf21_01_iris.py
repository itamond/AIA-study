import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes

tf.set_random_seed(337)
tv = tf.compat.v1

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes]
node_list = [32, 64, 64, 32, 8]

def hidden_layer():
    for i in range(4):
        globals()[f'w{i+2}'] = tv.Variable(tv.random.normal([node_list[i], node_list[i+1]]))
        globals()[f'b{i+2}'] = tv.Variable(tv.zeros(node_list[i+1]))
        globals()[f'layer{i+2}'] = tv.matmul(globals()[f'layer{i+1}'], globals()[f'w{i+2}']) + globals()[f'b{i+2}']

for i in range(len(data_list)):

    x, y = data_list[i](return_X_y=True)
    xp = tv.placeholder(tf.float32, shape = [None, x.shape[1]])
    
    if i < 4:
        n = len(np.unique(y))
        y = tf.keras.utils.to_categorical(y)
    else:
        n = 1
        y = y.reshape(-1, 1)
    yp = tv.placeholder(tf.float32, shape = [None, n])

    w1 = tv.Variable(tv.random.normal([x.shape[1], node_list[0]]), name='weight1')
    b1 = tv.Variable(tv.zeros(node_list[0]), name='bias1')
    layer1 = tv.matmul(xp, w1) + b1

    hidden_layer()

    if i < 4:
        w6 = tv.Variable(tv.random.normal([node_list[4], n]), name='weight6')
        b6 = tv.Variable(tv.zeros([n]), name='bias6')
        hypothesis = tf.nn.softmax(tv.matmul(layer5, w6) + b6)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=yp, logits=hypothesis))
    else:
        w6 = tv.Variable(tv.random.normal([node_list[4], 1]), name='weight6')
        b6 = tv.Variable(tv.zeros([1]), name='bias6')
        hypothesis = tv.matmul(layer5, w6) + b6
        
        loss = tf.reduce_mean(tf.square(hypothesis - yp))
        
    train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1).minimize(loss)
    
    epochs = 10

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for step in range(epochs):
            _, loss_val = sess.run([train, loss], feed_dict={xp:x, yp:y})
        
        # 4. 평가, 예측
        
        xp2 = tf.compat.v1.placeholder(tf.float32, shape=[None, x.shape[1]])
        
        if i < 4:
            y_pred = tf.nn.softmax(tv.matmul((tv.matmul((tv.matmul((tv.matmul((tv.matmul((tv.matmul(xp2, w1) + b1), w2) + b2), w3) + b3), w4) + b4), w5) + b5), w6) + b6)
            y_predict = sess.run([y_pred], feed_dict={xp2:x})
            print(data_list[i].__name__, 'acc : ', accuracy_score(np.argmax(y, axis=1), np.argmax(y_predict[0], axis=1)))
        else:
            # y_pred = tv.matmul((tv.matmul((tv.matmul((tv.matmul((tv.matmul((tv.matmul(xp2, w1) + b1), w2) + b2), w3) + b3), w4) + b4), w5) + b5), w6) + b6
            y_predict = tv.matmul(layer5, w6) + b6

            print(y_predict[0])
            print(y)
            print(data_list[i].__name__, 'r2 : ', r2_score(y, y_predict[0]))