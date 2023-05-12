#최소값을 찾는 알고리즘
#베이지안옵티마이제이션은 최대값 찾는것
import pandas as pd
import numpy as np
import hyperopt
print(hyperopt.__version__)  #0.2.7

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK #fmin은 최소값을 찾는 함수

search_space = {
    'x1' : hp.quniform('x1', -10, 10, 1),   #-10부터 10까지 1단위 라는 뜻
    'x2' : hp.quniform('x2', -15, 15, 1)   
    #      hp.quniform(label, low, high, q) #Q유니폼의 베이스라인
}
print(search_space)
# {'x1': <hyperopt.pyll.base.Apply object at 0x00000197D40C9340>, 'x2': <hyperopt.pyll.base.Apply object at 0x0000019795A4BD60>}

def objective_func(search_space):
    x1 = search_space['x1']
    x2 = search_space['x2'] 
    return_value = x1**2 - 20*x2
    
    return return_value
    #권장리턴방식 return {'loss' : return_value, 'status' : STATUS_OK}

trials_val = Trials()  #훈련 내용이 기록되는곳 hist와 같다

best = fmin(
    fn = objective_func, #목적함수
    space = search_space, 
    algo=tpe.suggest, #디폴트 알고리즘 선택 파라미터
    max_evals=20,  #n_iter와 같은 파라미터
    trials = trials_val,
    rstate = np.random.default_rng(seed=10), #랜덤스테이트    
)
# print('best : ',best)
# print(trials_val.results) #각 훈련의 verbose
'''
# results_val = pd.DataFrame(trials_val.vals)
# results_df = pd.DataFrame(trials_val.results)

# results = pd.concat([results_df, results_val], axis=1)
# results = results.drop(['status'],axis=1)


# min_loss_idx = results['loss'].idxmin()
# min_loss_row = results.loc[min_loss_idx]
# print(results)
# print(min_loss_row)
'''
results = [aaa['loss'] for aaa in trials_val.results]
#위 수식은 아래 포문과 같은 내용이다.
# for aaa in trials_val.results :
#     losses = aaa['loss']

df = pd.DataFrame({'x1' : trials_val.vals['x1'],
                   'x2' : trials_val.vals['x2'],
                   'results' : results})

print(df)