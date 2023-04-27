import numpy as np
import pandas as pd
import os
import unicodedata

# train: x
x = {}
train = os.scandir('./_data/aif/TEST_INPUT/')
train_aws = os.scandir('./_data/aif/TEST_AWS/')
region = {'아름동': '세종고운', '신흥동': '세종연서', '노은동': '계룡', '문창동': '오월드', '읍내동': '장동', 
          '정림동': '오월드', '공주': '공주', '논산': '논산', '대천2동': '대천항', '독곶리': '대산', '동문동': '태안', 
          '모종동': '아산', '신방동': '성거', '예산군': '예산', '이원면': '태안', '홍성읍': '홍북', '성성동': '성거'}

for file in region.keys():
    x = {}
    pm_datapath = './_data/aif/TEST_INPUT/' + unicodedata.normalize('NFC', file) + '.csv'
    pm_data = pd.read_csv(pm_datapath, index_col=False)
    aws_datapath = './_data/aif/TEST_AWS/' + unicodedata.normalize('NFC', region[file]) + '.csv'
    aws_data = pd.read_csv(aws_datapath, index_col=False)
    
    df_x = pd.DataFrame({
        'date': pm_data['일시'],
        'PM': pm_data['PM2.5'],
        'temperature': aws_data['기온(°C)'],
        'direction': aws_data['풍향(deg)'],
        'velocity': aws_data['풍속(m/s)'],
        'rain': aws_data['강수량(mm)'],
        'humidity': aws_data['습도(%)'],
    }, )
    
    df_x.to_csv(f'./_data/aif/NTEST/{file}.csv', index=False)
    
    
    
    