#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:38:21 2023

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from basicUtils import *
from os import path
from tqdm import tqdm

from numpy import random
from sklearn import metrics
from sklearn.inspection import permutation_importance, partial_dependence

from numpy import random
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale 

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_g1s.csv',index_col=0)
df_g1s = df_g1s.drop(columns=['age','G1S_logistic'])

df_g1s = df_g1s[df_g1s['time_g1s'] >= 0]

X = df_g1s.drop(columns='sgr'); y = df_g1s['sgr']
feature_names = X.columns
    
#%% # Parital correlation prediction on SGR

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay

forest = RandomForestRegressor(n_estimators=100)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
forest.fit(X_train,y_train)

top_imp_features = feature_names[forest.feature_importances_.argsort()[::-1]][0:5]
print(f'Top important features: {top_imp_features}')

# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
#     max_depth=1, random_state=0).fit(X_train, y_train)
features = [0, 22, (0, 22)]
PartialDependenceDisplay.from_estimator(forest, X, features)



