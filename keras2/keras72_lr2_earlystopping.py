# 1. 최소값을 넣을 변수를 하나 준비, 카운트할 변수 하나 준비
# 2. 다음 에포의 값과 최소값을 비교. 최소값이 갱신되면 그 변수에 최소값을 넣어준다, 카운트 변수 초기화
# 3. 갱신이 되지 않으면 카운트 변수 +1


x = 10
y = 10
w = 10
lr = 0.1
epochs = 1000
es_constant = 0.001


for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y) ** 2    # se
    
    print('Loss :', round(loss, 4), '\tPredict : ', round(hypothesis, 4))

    up_predict = x * (w + lr)
    up_loss = (y- up_predict) **2
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2
    
    if(up_loss >= down_loss):
        w = w - lr
    elif (up_loss < down_loss):
        w = w + lr
    if round(loss, 4) < es_constant:
        
        break
