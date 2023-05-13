import tensorflow as tf

x_train = [1,2,3]
y_train = [1,2,3]
# x_train = [1]
# y_train = [2]
x_test = [4,5,6]
y_test = [4,5,6]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))   #mse



####################optimizer#####################
lr = 0.1

gradient = tf.reduce_mean((x * w - y) * x)     # -> 로스의 미분값이다.

# y = x*w + b
# w_f = W_i - lr * (delta e / delta w)

descent = w - lr * gradient   # descent 갱신된 웨이트
# w = descent
update = w.assign(descent)   # w = w - lr * gradient


##################################################

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    
    _, loss_v, w_v =sess.run([update, loss, w], feed_dict={x : x_train, y : y_train})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)

sess.close()   #리스트는 초기화되지 않음



# print("==============================w history=================================")
# print(w_history)
# print("==============================loss history=================================")
# print(loss_history)


#체인룰 = 미분에 미분은 미분미분이다


from sklearn.metrics import r2_score, mean_absolute_error

y_pred = x_test * w_v

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('R2 score :', r2)
print('mae :', mae)

# R2 score : 0.999999989276489
# mae : 8.344650268554688e-05