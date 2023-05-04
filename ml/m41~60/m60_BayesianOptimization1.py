#최대값 찾는 알고리즘
#파라미터와 '최대값을 리턴할' 함수를 입력해야한다**********************

param_bounds = {'x1' : (-1, 5), #변수명은 텍스트, 값은 튜플 형태로 넣어준다.
                'x2' : (0, 4)} #딕셔너리에 튜플 형태

def y_function(x1, x2):
    return -x1 **2 - (x2 -2) **2 +10
# 이 함수의 최대값을 찾을것이다.

from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(
    f = y_function,
    pbounds = param_bounds,
    random_state = 337
)


optimizer.maximize(init_points=5,
                   n_iter=16)

print(optimizer.max)


# | 21        | 10.0      | 0.0007767 | 2.0       |        x1이 0일때,  x2가 2일때 10이라는 최대값을 갖는다.


#베이시안옵티마이져는 '최대값'을 찾는 옵티마이져
#분류 모델에서는 ACC에 베이시안옵티마이져를 적용
#회귀 에서는 R2를 최대값으로 보면 좋지만, R2는 신빙성이 낮다.