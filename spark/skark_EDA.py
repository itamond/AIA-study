import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
from haversine import haversine
nonsan = pd.read_csv('./_data/aif/TRAIN/논산.csv', index_col=False)
nonsan_aws = pd.read_csv('./_data/aif/TRAIN_AWS/논산.csv', index_col=False)

def visual():
    fig, axis = plt.subplots(figsize=(50, 20))

    # 기온
    plt.subplot(1, 5, 1)
    nonsan.interpolate(inplace=True)
    nonsan_aws['기온(°C)'].interpolate(inplace=True)
    cov = np.cov(nonsan['PM2.5'], nonsan_aws['기온(°C)'])[0,1]
    plt.title(f'Temperature \n (Cov = {round(cov, 8)})', fontsize=50)
    plt.scatter(nonsan_aws['기온(°C)'], nonsan['PM2.5'])
    ax = plt.gca()
    ax.xaxis.set_major_locator(dates.HourLocator(interval=10))
    plt.grid()

    # 풍향
    plt.subplot(1, 5, 2)
    nonsan.interpolate(inplace=True)
    nonsan_aws['풍향(deg)'].interpolate(inplace=True)
    cov = np.cov(nonsan['PM2.5'], nonsan_aws['풍향(deg)'])[0,1]
    plt.title(f'Wind - direction \n (Cov = {round(cov, 8)})', fontsize=50)
    plt.scatter(nonsan_aws['풍향(deg)'], nonsan['PM2.5'])
    ax = plt.gca()
    ax.xaxis.set_major_locator(dates.HourLocator(interval=10))
    plt.grid()

    # 풍속
    plt.subplot(1, 5, 3)
    nonsan.interpolate(inplace=True)
    nonsan_aws['풍속(m/s)'].interpolate(inplace=True)
    cov = np.cov(nonsan['PM2.5'], nonsan_aws['풍속(m/s)'])[0,1]
    plt.title(f'Wind - Velocity \n (Cov = {round(cov, 8)})', fontsize=50)
    plt.scatter(nonsan_aws['풍속(m/s)'], nonsan['PM2.5'])
    ax = plt.gca()
    ax.xaxis.set_major_locator(dates.HourLocator(interval=10))
    plt.grid()

    # 강수량
    plt.subplot(1, 5, 4)
    nonsan.interpolate(inplace=True)
    nonsan_aws['강수량(mm)'].interpolate(inplace=True)
    cov = np.cov(nonsan['PM2.5'], nonsan_aws['강수량(mm)'])[0,1]
    plt.title(f'Rain \n (Cov = {round(cov, 8)})', fontsize=50)
    plt.scatter(nonsan_aws['강수량(mm)'], nonsan['PM2.5'])
    ax = plt.gca()
    ax.xaxis.set_major_locator(dates.HourLocator(interval=10))
    plt.grid()

    # 습도
    plt.subplot(1, 5, 5)
    nonsan.interpolate(inplace=True)
    nonsan_aws['습도(%)'].interpolate(inplace=True)
    cov = np.cov(nonsan['PM2.5'], nonsan_aws['습도(%)'])[0,1]
    plt.title(f'Humidity \n (Cov = {round(cov, 8)})', fontsize=50)
    plt.scatter(nonsan_aws['습도(%)'], nonsan['PM2.5'])
    ax = plt.gca()
    ax.xaxis.set_major_locator(dates.HourLocator(interval=10))
    plt.grid()

    plt.savefig('nonsan.png')
    plt.show()

visual()

df = pd.DataFrame({'PM': nonsan['PM2.5'], 'temp': nonsan_aws['기온(°C)'], 'velo': nonsan_aws['풍속(m/s)'], 'deg': nonsan_aws['풍향(deg)'], 'rain': nonsan_aws['강수량(mm)'], 'humid': nonsan_aws['습도(%)']})
print(df.corr(method='pearson'))

aws = pd.read_csv('./_data/aif/META/awsmap.csv', index_col=False)
pm = pd.read_csv('./_data/aif/META/pmmap.csv', index_col=False)

aws_info = {}
for i in range(len(aws)):
    aws_info[aws.iloc[i]['Location']] = (aws.iloc[i]['Latitude'], aws.iloc[i]['Longitude'])

pm_info = {}
for i in range(len(pm)):
    pm_info[pm.iloc[i]['Location']] = (pm.iloc[i]['Latitude'], pm.iloc[i]['Longitude'])

result = {}
for i in pm_info.keys():
    if i not in result.keys():
        maxDist=10000000000
        for j in aws_info.keys():
            dist = haversine(pm_info[i], aws_info[j])
            if maxDist > dist:
                maxDist = dist
                current = j
    result[i] = current

print(result)