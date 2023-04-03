import numpy as np

aaa = np.array([1,2,3])

bbb = aaa       #aaa의 주소값이 bbb로 바뀌었다.

bbb[0] = 4
print(aaa)      #[4 2 3]
print(bbb)      #[4 2 3]


print("===========================================")
ccc = aaa.copy()             #새로운 메모리 구조가 생성. 진 짜 복 사
ccc[1] = 7

print(ccc)      #[4 7 3]
print(aaa)      #[4 2 3]
