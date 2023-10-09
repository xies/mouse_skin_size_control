#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:13:36 2023

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from basicUtils import z_standardize

from numpy import random
from sklearn import metrics
from sklearn.inspection import permutation_importance, partial_dependence

from numpy import random
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale 

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_g1s.csv',index_col=0)
df_g1s = df_g1s.drop(columns=['age','G1S_logistic'])

#Trim out G2 cells
df_g1s = df_g1s[df_g1s['time_g1s'] >= 0]


X = df_g1s.drop(columns=['time_g1s'])
y = df_g1s['time_g1s']

#Add interaction effects ?
X['vol*sgr'] = z_standardize(X['sgr'] * X['vol_sm'])

#%% Establish the theoretical maximum R2 based on time resolution alone

Nsample = 10000

x = np.random.lognormal(mean=np.log(48),sigma=np.sqrt(0.03),size=(Nsample))

bins = np.arange(0,10*24,12)

x_hat = bins[np.digitize(x,bins)]

plt.scatter(x,x_hat)
max_exp_Rsq = np.corrcoef(x,x_hat)[0,1]**2
plt.title(f'Maximum expected Rsq = {max_exp_Rsq}')

#%%

smfdata = sm.datasets.get_rdataset("dietox", "geepack").data

md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])

mdf = md.fit()
print(mdf.summary())