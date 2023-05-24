# [실습] earlystopping

x = 10
y = 10
w = 10
lr = 0.004
epochs = 10000
count = 0
min_loss = 0
earlystopping_count = 3

loss_list = []
hypothesis_list = []
for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y) ** 2
    
    print('epoch : ', i+1, '\tloss : ', loss, '\tPredict : ', hypothesis)
    
    up_predict = x * (w+lr)
    up_loss = (y - up_predict) ** 2 
    
    down_predict = x * (w-lr)
    down_loss = (y - down_predict) ** 2
    
    if (up_loss > down_loss):
        w = w - lr
    elif (up_loss < down_loss):
        w = w + lr
        
    loss_list.append(loss)
    hypothesis_list.append(hypothesis)
    
    if count != 0:
        if min_loss < loss_list[i]:
            count +=1
        elif min_loss > loss_list[i]:
            count = 0
    if i > 0 and loss_list[i-1] < loss_list[i] and count==0:
        min_loss = loss_list[i-1]
        count+=1
    if count == earlystopping_count-1:
        print(f'Earlystopping activated - best weight\nepoch : {i-earlystopping_count+1} \tloss : {loss_list[i-earlystopping_count]}\t Predict : {hypothesis_list[i-earlystopping_count]}')
        break