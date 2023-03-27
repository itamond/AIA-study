import numpy as np

dataset = np.array(range(1, 11))    # 1부터 10까지
timesteps = 5                       # 5개씩 잘라라


def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)
# aaa라는 리스트라는 공간을 만들고
#for,  반복할거야 range len(dataset) - timesteps + 1   (난 6번을 반복 하겠어요) i는 반복될때마다 카운트가 하나씩 올라감. 
#  아래 종속된것                  i는 : (0부터) 0 + 5
# aaa리스트에 subset이 append된다 그 다음 i 는 1부터 1+ 5, 반복이다.

#def 밑으로 함수 생성구간. 실행되진 않는다.


bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape)    #(6, 5)

# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]


x = bbb[:, :4]   #콤마 다음은 열에 대한 이야기 이다
x = bbb[:, :-1]  #모든행, 마지막 열 전'까지'
y = bbb[:, -1]   #모든 행, 마지막 열
print(x)