#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:53:13 2023

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_g1s.csv',index_col=0)
df_g1s = df_g1s.drop(columns=['age','G1S_logistic'])

# # Categorical variable
regionnames = np.array(['R1','R2'])
df_g1s['region'] = regionnames[df_g1s['region'].values-1]

#Trim out G2 cells
df_g1s = df_g1s[df_g1s['time_g1s'] >= 0]

cellIDs = df_g1s['cellID']
df_g1s = df_g1s.drop(columns='cellID')


#%% Random effect grouped by cell (only intercept)

from sklearn import linear_model
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

reg = linear_model.RidgeCV()
reg = reg.fit(X_train,y_train)


results = pd.Series(reg.coef_, reg.feature_names_in_)
results.sort_values().plot.bar()
plt.tight_layout()
plt.ylabel('Effect size')
plt.title('Linear regression for G1S timing')

#%%

from merf import MERF