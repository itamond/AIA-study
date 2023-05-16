
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_iris, load_digits, load_wine, fetch_covtype
tf.compat.v1.set_random_seed(337)

data_list = [load_iris, load_digits, load_wine]

for i in range(len(data_list)):
    tf.compat.v1.global_variables_initializer()
    x_data, y_data = data_list[i](return_X_y=True)
    y_col_num = len(np.unique(y_data))
    y_data = tf.keras.utils.to_categorical(y_data)

    x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
    w = tf.Variable(tf.random.normal([x_data.shape[1], y_col_num]), name='weight')
    b = tf.Variable(tf.zeros([1, y_col_num]), name='bias')
    y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_col_num])

    hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

    # 수정된 부분1: Loss 함수를 cross_entropy loss로 변경
    loss = tf.compat.v1.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=hypothesis))

    # 수정된 부분2: Learning rate 값을 조정
    train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    # Epoch 값을 조정 (e.g., 10000)
    epochs = 30000
    # with 문 밖에서 선언
    with tf.compat.v1.Session() as sess:
        # 수정된 부분4: 세션을 시작하기 전에 변수를 초기화
        sess.run(tf.compat.v1.global_variables_initializer())

        for step in range(epochs):
            _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
            
            # if step % 100 == 0:
            #     print("epoch: %d, loss_val: %f" % (step, loss_val))

        xp2 = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
        y_pred = tf.nn.softmax(tf.compat.v1.matmul(xp2, w) + b)
        y_predict = sess.run([y_pred], feed_dict={xp2:x_data})

        print(data_list[i].__name__,'acc : ', accuracy_score(np.argmax(y_data, axis=1), np.argmax(y_predict[0], axis=1)))
    sess.close()