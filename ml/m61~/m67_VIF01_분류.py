from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

data_list = [load_iris(), load_breast_cancer(), load_digits(), load_wine(), fetch_covtype(), fetch_california_housing(), load_diabetes()]
for i in range(len(data_list)):
    datasets = data_list[i]
    df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
    df['target'] = datasets.target
    print(df)

    y = df['target']
    x = df.drop(['target'], axis=1)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # 다중공선성
    vif = pd.DataFrame()
    vif['variables'] = np.arange(x.shape[1])

    vif['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
    print(vif)

    x = pd.DataFrame(x.values)
    x = x.drop(vif['VIF'].idxmax(), axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=337, test_size=0.2)

    scaler2 = StandardScaler()
    x_train = scaler2.fit_transform(x_train)
    x_test = scaler2.transform(x_test)

    # 2. 모델
    model = RandomForestRegressor(random_state=337)
    model.fit(x_train, y_train)

    # 4. 평가, 예측
    results = model.score(x_test, y_test)
    print('results : ', results)