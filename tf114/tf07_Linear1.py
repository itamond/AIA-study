import tensorflow as tf
tf.set_random_seed(337) #시드 고정

#1. 데이터
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111, dtype=tf.float32) #111은 초기값 지정, 일반적으로 랜덤값 넣어 준다
b = tf.Variable(0, dtype=tf.float32) # 일반적으로 0을 많이 쓴다

#2. 모델구성

# y = w*x + b
# y = x*w + b  둘은 차이가 있다. 행렬 연산이면 달라짐.
# 실직적으로는 y = x*w + b 가 맞다.



hypothesis = x * w + b  #hypothesis = 가설, 예측한 값
#hypothesis와 y의 차가 loss.

#3-1 컴파일

loss = tf.reduce_mean(tf.square(hypothesis- y)) #mse
# w = w - 러닝레이트*(로스를 w로 편미분한 값)      <- epoch마다 이 값 만큼 갱신 시킨다.           미분 = 그 지점의 변화량을 구하는 것 로스의 기울기


optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
#lr이 너무 적다면 로컬미니마에 빠질 수 있다.


train = optimizer.minimize(loss)  #경사 하강법 방식으로 옵티마이져를 최적화하여 loss의 최소값을 뽑는다
# model.compile(loss='mse',optimizer='sgd') 이것과 같다. Stochastic Gradient Descent


#3-2 훈련

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) #세션을 열면 항상 초기화부터 실행


#model.fit 부분
epochs = 2001
for step in range(epochs) :
    sess.run(train)
    if step %100 == 0:  #100으로 나눈 나머지가 0일때
        print(step, 'loss :', sess.run(loss), 'w :', sess.run(w), 'b :',sess.run(b)) #loss 또한 세션 런 해야함 (verbose 부분)
        
sess.close()  #세션 종료. 생성한 메모리 terminate

