import tensorflow as tf
tf.compat.v1.set_random_seed(337)

#실습
# 08_2 카피


###############1.Session() // sess.run(변수)

###############2.Session() // 변수.eval(session=sess)

###############3.InteractiveSession() // 변수.eval

#1. 데이터
x= tf.placeholder(tf.float32, shape=[None])
y= tf.placeholder(tf.float32, shape=[None])
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #random_uniform 균등 분포(N빵), random_normal 정규 분포 [1]은 1개짜리
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #random_uniform 균등 분포(N빵), random_normal 정규 분포 [1]은 1개짜리
#2. 모델 구성
hypothesis = x * w + b
# #3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001355)
#w = w - 러닝레이트*(로스를 w로 편미분한 값)
#
train = optimizer.minimize(loss)





with tf.compat.v1.Session() as sess :
# sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())
    w1 = sess.run(w)
    b1 = sess.run(b)
    #model.fit()
    epochs = 1001
    for step in range(epochs) :
        # sess.run(train)
        _, loss_val= sess.run([train, loss], feed_dict = {x:[1,2,3,4,5], y:[2,4,6,8,10]})   
        #플레이스홀드로 공간만 만들어뒀기 때문에 feed_dic으로 값 지정
        #
        if step %20 == 0 :
            # print(step, 'loss :', sess.run(loss), 'w :', sess.run(w), 'b :', sess.run(b))            
            print(step, 'loss :', loss_val, 'w :', w1, 'b :', b1)

    
    x_data= tf.placeholder(tf.float32, shape=[None])
    # x_data = [6,7,8]
    y_pred = x_data*w1+b1
    y_predict = sess.run([y_pred], feed_dict={x_data:[6,7,8]})
    
    print('y_predict :', y_predict[0][0], y_predict[0][1], y_predict[0][2])
    
    
    # 쌤 코드
    # x_data = [6,7,8]
    # x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
    # y_predict = x_test * w_val + b_val
    # print(sess.run(y_predict,feed_dict={x_test:x_data}))
    
    
    
    
with tf.compat.v1.Session() as sess :
# sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    w2 = w.eval(session=sess)
    b2 = b.eval(session=sess)
    #model.fit()
    epochs = 1001
    for step in range(epochs) :
        # sess.run(train)
        _, loss_val = sess.run([train, loss], feed_dict = {x:[1,2,3,4,5], y:[2,4,6,8,10]})   
        #플레이스홀드로 공간만 만들어뒀기 때문에 feed_dic으로 값 지정
        #
        if step %20 == 0 :
            # print(step, 'loss :', sess.run(loss), 'w :', sess.run(w), 'b :', sess.run(b))            
            print(step, 'loss :', loss_val, 'w :', w2, 'b :', b2)

    
    x_data= tf.placeholder(tf.float32, shape=[None])
    # x_data = [6,7,8]
    y_pred = x_data*w2+b2
    y_predict = sess.run([y_pred], feed_dict={x_data:[6,7,8]})
    
    print('y_predict :', y_predict[0][0], y_predict[0][1], y_predict[0][2])
    
    




sess = tf.compat.v1.InteractiveSession()


# with tf.compat.v1.InteractiveSession() as sess :
# sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
w3 = w.eval()
b3 = b.eval()
#model.fit()
epochs = 1001
for step in range(epochs) :
    # sess.run(train)
    _, loss_val= sess.run([train, loss], feed_dict = {x:[1,2,3,4,5], y:[2,4,6,8,10]})   
    #플레이스홀드로 공간만 만들어뒀기 때문에 feed_dic으로 값 지정
    #
    if step %20 == 0 :
        # print(step, 'loss :', sess.run(loss), 'w :', sess.run(w), 'b :', sess.run(b))            
        print(step, 'loss :', loss_val, 'w :', w3, 'b :', b3)

x_data= tf.placeholder(tf.float32, shape=[None])
# x_data = [6,7,8]
y_pred = x_data*w3+b3
y_predict = sess.run([y_pred], feed_dict={x_data:[6,7,8]})

print('y_predict :', y_predict[0][0], y_predict[0][1], y_predict[0][2])

sess.close()


#실습



