import os
import numpy as np
import pandas as pd
from haversine import haversine
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from typing import Tuple

def load_aws_and_pm()->Tuple[pd.DataFrame,pd.DataFrame]:
    path='./_data/finedust'
    path_list=os.listdir(path)
    print(f'datafolder_list:{path_list}')

    meta='/'.join([path,path_list[0]])
    meta_list=os.listdir(meta)

    print(f'META_list:{meta_list}')
    awsmap=pd.read_csv('/'.join([meta,meta_list[0]]))
    awsmap=awsmap.drop(awsmap.columns[-1],axis=1)
    pmmap=pd.read_csv('/'.join([meta,meta_list[1]]))
    pmmap=pmmap.drop(pmmap.columns[-1],axis=1)
    return awsmap,pmmap

def load_distance_of_pm_to_aws(awsmap:pd.DataFrame,pmmap:pd.DataFrame)->pd.DataFrame:
    '''pm과 ams관측소 사이의 거리들을 프린트해준다'''
    a = []
    for i in range(pmmap.shape[0]):
        b=[]
        for j in range(awsmap.shape[0]):
            b.append(haversine((np.array(pmmap)[i, 1], np.array(pmmap)[i, 2]), (np.array(awsmap)[j, 1], np.array(awsmap)[j, 2])))
        a.append(b)    
    distance_of_pm_to_aws = pd.DataFrame(np.array(a),index=pmmap['Location'],columns=awsmap['Location'])
    return distance_of_pm_to_aws

def scaled_score(distance_of_pm_to_aws:pd.DataFrame,pmmap:pd.DataFrame,near:int=3)->Tuple[pd.DataFrame,np.ndarray]:
    '''pm으로부터 가까운 상위 near개의 환산점수'''
    min_i=[]
    min_v=[]
    for i in range(distance_of_pm_to_aws.shape[0]):
        min_i.append(np.argsort(distance_of_pm_to_aws.values[i,:])[:near])
        min_v.append(distance_of_pm_to_aws.values[i, min_i[i]])

    min_i = np.array(min_i)
    min_v = pd.DataFrame(np.array(min_v),index=distance_of_pm_to_aws.index)
    
    for i in range(pmmap.shape[0]):
        for j in range(near):
            min_v.values[i, j]=min_v.values[i, j]**2
            
    sum_min_v = np.sum(min_v, axis=1)

    recip=[]
    for i in range(pmmap.shape[0]):
        recip.append(sum_min_v[i]/min_v.values[i, :])
    recip = np.array(recip)
    recip_sum = np.sum(recip, axis=1)
    coef = 1/recip_sum

    result = []
    for i in range(pmmap.shape[0]):
        result.append(recip[i, :]*coef[i])
    result = pd.DataFrame(np.array(result),index=distance_of_pm_to_aws.index)
    return result,min_i
