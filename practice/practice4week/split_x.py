import numpy as np
import pandas as pd


timesteps = 10

def split_x(datasets, timesteps) :
    aaa = []
    for i in range(len(datasets)-timesteps) :
        subset = datasets[i : (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)


