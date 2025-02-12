import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('parkinsons.data')
df.head()

df.columns

df.describe()

df.info()

sns.countplot(df['status'])

df.dtypes

from math import nan
features=df.loc[:,df.columns!='status'].values[:,1:]

labels=df.loc[:,'status'].values

print(labels[labels==1].shape[0], labels[labels==0].shape[0])

scaler=MinMaxScaler((-1,1))

x=scaler.fit_transform(features)

y=labels

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)


model=XGBClassifier(eval_metric='mlogloss')

model.fit(x_train,y_train)

XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                       colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
                      gamma=0, gpu_id=-1, importance_type='gain',
                      interaction_constraints='', learning_rate=0.300000012,
                      max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
                      monotone_constraints='()', n_estimators=100, n_jobs=4,
                      num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
                      scale_pos_weight=1, subsample=1, tree_method='exact',
                      use_label_encoder=False, validate_parameters=1, verbosity=None)

y_pred=model.predict(x_test)

print(accuracy_score(y_test, y_pred)*100)

from sklearn.metrics import confusion_matrix

pd.DataFrame(

    confusion_matrix(y_test, y_pred),

    columns=['Predicted Healthy', 'Predicted Parkinsons'],

    index=['True Healthy', 'True Parkinsons']

)

sc = MinMaxScaler(feature_range = (0,1))

sc = sc.fit(x)

model.save_model("modelForParkinson.json")

with open('standardScalar.sav', 'wb') as f:
  pickle.dump(sc, f)