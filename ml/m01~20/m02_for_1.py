# for i in 리스트 데이터 : 

list = ['a', 'b', 'c', 4]
#list 형태는 중간에 문자와 숫자 섞을수 있다. numpy 형식은 문자나 숫자 하나의 형태만 가능

for i in list :
    print(i)
#abc가 순서대로 출력됨

for index, value in enumerate(list) : #인덱스는 순서, 밸류는 값이다. enumerate = 순서도 뽑는 함수
    print(index, value)
    


