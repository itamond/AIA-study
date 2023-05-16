import tensorflow as tf
import matplotlib.pyplot as plt

# x = [1,2,3]
# y = [1,2,3]
x = [1,2]
y = [1,2]

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess :
    for i in range(-30, 51):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict = {w : curr_w})
        
        w_history.append(curr_w)
        loss_history.append(curr_loss)


print("==============================w history=================================")
print(w_history)
print("==============================loss history=================================")
print(loss_history)


plt.plot(w_history, loss_history)
plt.xlabel('weight')
plt.ylabel('loss')
plt.show()

#w = w - lr * (e(로스)/w(웨이트) 로스를 웨이트로 미분한 양)
#기울기가 음수라면, -러닝레이트와 곱해져서 양의 방향으로 이동. 
#기울기가 양수라면, -러닝레이트와 곱해져서 음의 방향으로 이동.


#(e(로스)/w(웨이트) 로스를 웨이트로 미분한 양) 웨이트와 로스의 변화량에 대한 내용. 기울기에 대한 것
