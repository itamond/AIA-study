
x = 10
y = 10
w = 10
lr = 0.1
epochs = 1000

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
    if round(loss, 4) == 0:
        break
