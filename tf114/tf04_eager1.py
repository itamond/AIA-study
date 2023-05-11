import tensorflow as tf
print(tf.__version__)

# 즉시실행모드
print(tf.executing_eagerly()) #텐서1 = False(그래프 연산 방식)   텐서2 = True
# 즉시실행모드 비활성
tf.compat.v1.disable_eager_execution() # 텐서 2.0을 텐서1.0 방식으로 사용
# 즉시실행모드 활성
tf.compat.v1.enable_eager_execution() # 

aaa = tf.constant('hello world')

sess = tf.compat.v1.Session()

print(sess.run(aaa))
