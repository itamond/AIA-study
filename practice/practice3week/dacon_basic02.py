

import numpy as np
import pandas as pd
import datetime
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
# from keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


path = './_data/basic/'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv = pd.read_csv(path+'test.csv',index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv',index_col=0)

# enc = LabelEncoder()
# enc.fit(train_csv['type'])
# train_csv['type'] = enc.transform(train_csv['type'])
# test_csv['type'] = enc.transform(test_csv['type'])



numerical_columns = train_csv.select_dtypes(exclude='object').columns.tolist()
numerical_columns.remove('전화해지여부')
numerical_columns.remove('상담전화건수')
numerical_columns.remove('음성사서함이용')
def show_dist_plot(df, columns):
    for column in columns:
        f, ax = plt.subplots(1,2,figsize=(16,4))
        sns.stripplot(x=df['전화해지여부'],y=df[column], ax=ax[0],hue=df['전화해지여부'])
        sns.violinplot(data=df, x='전화해지여부', y=column, ax=ax[1])
        
show_dist_plot(train_csv, numerical_columns)


plt.figure(figsize=(18,8))
corr= train_csv.corr()
sns.heatmap(corr, annot=True, square=False, vmin=-.6, vmax=1.0)
plt.show()


#Library
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import plot_roc_curve,accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

#Standardscaler
ss= StandardScaler()
train_csv[numerical_columns] = ss.fit_transform(train_csv[numerical_columns])

#factorize


X = train_csv.drop(['전화해지여부','음성사서함이용', '상담전화건수'],axis=1)
y = train_csv.전화해지여부


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=42)

def Model(model):
    model.fit(X_train,y_train)
    score = model.score(X_test,y_test)
    model_train_score= model.score(X_train,y_train)
    model_test_score=model.score(X_test,y_test)
    prediction = model.predict(X_test)
    cm = confusion_matrix(y_test, prediction)
    print("Testing Score\n", score)
    plot_confusion_matrix(model,X_test,y_test,cmap='OrRd')
    
#RandomForest
rf= RandomForestClassifier()
params = {
    'max_depth': [2, 3],
    'min_samples_split': [2, 3]
}

grid_tree = GridSearchCV(rf, param_grid=params, cv=3, refit=True)
rf.fit(X_train,y_train)
Model(rf)

grid_tree.fit(X_train, y_train)


ss= StandardScaler()
test_csv[numerical_columns] = ss.fit_transform(test_csv[numerical_columns])

test_csv = test_csv.drop(['음성사서함이용', '상담전화건수'],axis=1)

#factorize
final_pred = rf.predict(test_csv)

submission['전화해지여부'] = final_pred
submission.to_csv("./_save/basic/submission_3_.csv")
rf.fit(X_train, y_train)
print(rf.feature_importances_)
y_predict = rf.predict(X_test)
macro_f1 = f1_score(y_test, y_predict, average='macro')
print('f1 :',macro_f1)

print(train_csv.columns)


# print('best parameters : ', grid_tree.best_params_)
# print('best score : ', grid_tree.best_score_)

