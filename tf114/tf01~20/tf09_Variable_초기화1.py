import tensorflow as tf
tf.compat.v1.set_random_seed(123)

변수 = tf.compat.v1.Variable(tf.random_normal([2]), name='weight')
#random_normal 안의 숫자는 쉐이프
print(변수)
# <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

# 초기화 첫번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)
print('aaa :', aaa) #aaa : [-1.5080816   0.26086742]
sess.close()



# 초기화 두번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess) #두번째 초기화법. 텐서플로 데이터형을 파이썬 데이터형으로 변환
print('bbb :', bbb)
sess.close()



# 초기화 세번째
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval() #session=sess라고 명시 안해도 됨
print('ccc :', ccc)
