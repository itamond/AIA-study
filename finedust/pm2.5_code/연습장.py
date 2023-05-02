import os
import numpy as np
import pandas as pd
from haversine import haversine
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from typing import Tuple,List


imputer = IterativeImputer(XGBRegressor())
le = OrdinalEncoder()

from preprocess import load_aws_and_pm
awsmap,pmmap=load_aws_and_pm()

from preprocess import load_distance_of_pm_to_aws
distance_of_pm_to_aws=load_distance_of_pm_to_aws(awsmap,pmmap)
# distance_of_pm_to_aws=distance_of_pm_to_aws.values

# def sort_data_by_first_element(*args: List[List]) -> Tuple[List]:
#     return tuple(map(list, zip(*sorted(zip(*args), key=lambda x: x[0]))))

def min_dist_from_pm(distance_of_pm_to_aws:pd.DataFrame,pmmap:pd.DataFrame,near:int=3)->np.ndarray:
    min_index_of_dis_to_pm=[]
    min_value_of_dis_to_pm=[]
    for i in range(distance_of_pm_to_aws.shape[0]):
        min_index_of_dis_to_pm.append(np.argsort(distance_of_pm_to_aws.values[i,:])[:near])
        min_value_of_dis_to_pm.append(distance_of_pm_to_aws.values[i, min_index_of_dis_to_pm[i]])

    min_index_of_dis_to_pm = np.array(min_index_of_dis_to_pm)
    min_value_of_dis_to_pm = pd.DataFrame(np.array(min_value_of_dis_to_pm),index=distance_of_pm_to_aws.index)
    
    for i in range(pmmap.shape[0]):
        for j in range(near):
            min_value_of_dis_to_pm.values[i, j]=min_value_of_dis_to_pm.values[i, j]**2
            
    sum_min_v = np.sum(min_value_of_dis_to_pm, axis=1)

    recip=[]
    for i in range(pmmap.shape[0]):
        recip.append(sum_min_v[i]/min_value_of_dis_to_pm.values[i, :])
    recip = np.array(recip)
    recip_sum = np.sum(recip, axis=1)
    coef = 1/recip_sum

    result = []
    for i in range(pmmap.shape[0]):
        result.append(recip[i, :]*coef[i])
    result = pd.DataFrame(np.array(result),index=distance_of_pm_to_aws.index)
    return result
result=min_dist_from_pm(distance_of_pm_to_aws,pmmap,near=3)
print(result)