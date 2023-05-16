
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_iris, load_digits, load_wine, fetch_covtype
tf.compat.v1.set_random_seed(337)
#1. 데이터

data_list = [load_iris, load_wine, load_digits]
learning_rate_list = [0.01, 0.1, 100]
for i in range(len(data_list)):
    tf.compat.v1.global_variables_initializer()
    x_data, y_data = data_list[i](return_X_y=True)
    # y_data = tf.one_hot(y_data, depth=len(np.unique(y_data)))
    y_col_num = len(np.unique(y_data))
    y_data = tf.keras.utils.to_categorical(y_data)
    
    
    #2. 모델구성
    
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
    w = tf.Variable(tf.random.normal([x_data.shape[1], y_col_num]), name='weight')
    b = tf.Variable(tf.zeros([1, y_col_num]), name='bias')
    y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_col_num])

    hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

    #3-1 컴파일

    # loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))
    # loss = tf.reduce_mean(-tf.reduce_sum(y * tf.nn.log_softmax(hypothesis), axis=1))
    # loss =y * tf.nn.log_softmax(hypothesis)
    loss = tf.compat.v1.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=hypothesis))    
    # loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
    # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
    # train = optimizer.minimize(loss)
    
    train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate_list[i]).minimize(loss)


    epochs = 5000
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for step in range(epochs):
            _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
            # if step % 100 == 0:
            #     print("epoch: %d, loss_val: %f" % (step, loss_val))
            # print(w_val[0][0])
        # 4. 평가, 예측
        xp2 = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
        y_pred = tf.nn.softmax(tf.compat.v1.matmul(xp2, w_val) + b_val)
        y_predict = sess.run([y_pred], feed_dict={xp2:x_data})
        print(data_list[i].__name__,'acc : ', accuracy_score(np.argmax(y_data, axis=1), np.argmax(y_predict[0], axis=1)))


    # r2 :  -244.58336231944563
    # acc :  0.375


    #iris wine fetchcov digits daconwine