import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)
y = tf.Variable([3], dtype=tf.float32)#디폴트가 행렬 연산이기때문에 리스트 형태로 들어감.

#전역 변수 초기화
init = tf.compat.v1.global_variables_initializer()#기본문법 변수 초기화
sess.run(init)

print(sess.run(x + y))  # [5.]


#텐서1은 변수를 사용하기 전에 항상 초기화 해야한다.
#컨스턴트, 플레이스홀더, 배리어블 전부 그래프 연산 방식
#플레이스홀더는 딕셔너리 형태로 feed dic에 값을 꼭 넣어얗ㅁ
#배리어블은 변수를 꼭 초기화시켜줘야함
#모든것은 sess.run()에 넣어야 작동함