import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Sampler
import unicodedata
class AirDataset(Dataset):
    def __init__(self, i, set_type, is_train=True):
        types = {'train': 0, 'vali': 1, 'test': 2}
        self.seq_len = 48
        self.pred_len = 72
        self.set_type = types[set_type]
        self.i = i
        self.is_train = is_train
        self.__read_data__()
        
    def __read_data__(self):
        self.x = pd.read_csv(f'./data/{self.i}.csv', index_col=False)
        train_len = int(len(self.x)*0.7)
        val_len = int(len(self.x)*0.85)

        border1s = [0, train_len, val_len]
        border2s = [train_len, val_len, len(self.x)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data = self.x
        self.data['PM'].interpolate(inplace=True, limit_direction='backward')
        self.data['temperature'].interpolate(inplace=True, limit_direction='backward')
        self.data['direction'].interpolate(inplace=True, limit_direction='backward')
        self.data['velocity'].interpolate(inplace=True, limit_direction='backward')
        self.data['rain'].interpolate(inplace=True, limit_direction='backward')
        self.data['humidity'].interpolate(inplace=True, limit_direction='backward') 
        
        train_data = self.data[border1s[0]:border2s[0]]
        train_max = train_data.max(axis=0)
        train_min = train_data.min(axis=0)
            
        # 연도가 안 나와 있어서 사실상 시간 정보는 쓸 수가 없음 -> 근데 시간 정보에 따라서 확실히 좋아지는 게 있을듯
        df_data_col = self.data.columns[1:]
        df_data = self.data[df_data_col][border1:border2]
        df_data['PM'] = (df_data['PM'] - train_min['PM']) / (train_max['PM'] - train_min['PM'])
        df_data['temperature'] = (df_data['temperature'] - train_min['temperature']) / (train_max['temperature'] - train_min['temperature'])
        df_data['direction'] = (df_data['direction'] - train_min['direction']) / (train_max['direction'] - train_min['direction'])
        df_data['velocity'] = (df_data['velocity'] - train_min['velocity']) / (train_max['velocity'] - train_min['velocity'])
        df_data['rain'] = (df_data['rain'] - train_min['rain']) / (train_max['rain'] - train_min['rain'])
        df_data['humidity'] = (df_data['humidity'] - train_min['humidity']) / (train_max['humidity'] - train_min['humidity'])
        
        self.data_x = df_data.values
        self.data_y = df_data.values

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y


class PredDataset(Dataset):
    def __init__(self, i):
        types = {'train': 0, 'vali': 1, 'test': 2}
        self.seq_len = 48
        self.pred_len = 72
        self.i = i
        self.__read_data__()
        
    def __read_data__(self):
        region =  {'아름동': '세종고운', '신흥동': '세종연서', '노은동': '계룡', '문창동': '오월드', '읍내동': '장동', 
          '정림동': '오월드', '공주': '공주', '논산': '논산', '대천2동': '대천항', '독곶리': '대산', '동문동': '태안', 
          '모종동': '아산', '신방동': '성거', '예산군': '예산', '이원면': '태안', '홍성읍': '홍북', '성성동': '성거'}

        self.train = pd.read_csv(f'./data/{self.i}.csv', index_col=False)
        self.x = pd.read_csv(f'./dataset/TEST_INPUT/{self.i}.csv', index_col=False)
        self.x_aws = pd.read_csv(f'./dataset/TEST_AWS/{region[self.i]}.csv', index_col=False)
        
        train_len = int(len(self.x)*0.7)
        val_len = int(len(self.x)*0.85)

        border1s = [0, train_len, val_len]
        border2s = [train_len, val_len, len(self.x)]

        self.data = self.x
        self.data_aws = self.x_aws

        train_data = self.train[border1s[0]:border2s[0]]
        train_max = train_data.max(axis=0)
        train_min = train_data.min(axis=0)
        
        # 연도가 안 나와 있어서 사실상 시간 정보는 쓸 수가 없음
        self.data['PM2.5'] = (self.data['PM2.5'] - train_min['PM']) / (train_max['PM'] - train_min['PM'])
        self.data_aws['기온(°C)'] = (self.data_aws['기온(°C)'] - train_min['temperature']) / (train_max['temperature'] - train_min['temperature'])
        self.data_aws['풍향(deg)'] = (self.data_aws['풍향(deg)'] - train_min['direction']) / (train_max['direction'] - train_min['direction'])
        self.data_aws['풍속(m/s)'] = (self.data_aws['풍속(m/s)'] - train_min['velocity']) / (train_max['velocity'] - train_min['velocity'])
        self.data_aws['강수량(mm)'] = (self.data_aws['강수량(mm)'] - train_min['rain']) / (train_max['rain'] - train_min['rain'])
        self.data_aws['습도(%)'] = (self.data_aws['습도(%)'] - train_min['humidity']) / (train_max['humidity'] - train_min['humidity'])
        
        data = self.data['PM2.5'].values
        data_aws_col = self.data_aws.columns[3:]
        data_aws = self.data_aws[data_aws_col].values
        
        data = (torch.Tensor(data).unsqueeze(1)).numpy()
        df_data =np.concatenate((data, data_aws), axis=-1)        
        
        self.data_x = df_data
        self.data_y = df_data

    def __len__(self):
        return len(self.data_x) - (self.seq_len + self.pred_len) + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]  
        return seq_x

class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        for i in range(len(self.indices)):
            yield self.indices[i]

    def __len__(self):
        return len(self.indices)
