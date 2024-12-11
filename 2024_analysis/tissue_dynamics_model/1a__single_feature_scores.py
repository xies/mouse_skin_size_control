#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:42:30 2024

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score, log_loss
from os import path

from numpy.linalg import eig

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
df1 = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)
df1['Region'] = 1
df1_ = df1[df1['Phase'] != '?']

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'
df2 = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)
df2['Region'] = 2
df2_ = df2[df2['Phase'] != '?']

df_ = pd.concat((df1_,df2_),ignore_index=True)
df_['UniqueID'] = df_['basalID'].astype(str) + '_' + df_['Region'].astype(str)
N,P = df_.shape

# Make ratios some of the correlated components
df_['Height to BM relative to cell height'] = df_['Height to BM'] / df_['Height']
df_['NC ratio (sm)'] = df_['Volume (sm)'] / df_['Nuclear volume (sm)']

df_['CV neighbor cell volume'] = df_['Std neighbor cell volume'] / df_['Mean neighbor cell volume']
df_['CV neighbor apical area'] = df_['Std neighbor apical area'] / df_['Mean neighbor apical area']
df_['CV neighbor basal area'] = df_['Std neighbor basal area'] / df_['Mean neighbor basal area']

df_['Neighbor CV cell volume frame-1'] = df_['Neighbor std cell volume frame-1'] / df_['Neighbor mean cell volume frame-1']

df_['G1S_logistic'] = (df_['Phase'] == 'SG2').astype(int)

features_list = { 
                # Cell geometry
                'Volume':'vol'
                ,'Volume (sm)':'vol_sm'
                ,'Nuclear volume':'nuc_vol'
                ,'Nuclear volume (sm)':'nuc_vol_sm'
                ,'Surface area':'sa'
                ,'Basal area':'basal'
                ,'Apical area':'apical'
                }

df_g1s = df_.loc[:,list(features_list.keys())]
df_g1s = df_g1s.rename(columns=features_list)
df_g1s['G1S_logistic'] = df_['G1S_logistic']

#%% 

y = df_['G1S_logistic']
names = df_g1s.columns.drop('G1S_logistic')

scores = []
for feature in names:
    
    X = scale(df_g1s[feature].values.reshape(-1,1))
    model = LogisticRegression().fit(X,y)
    
    scores.append(roc_auc_score(y,model.predict(X)))

plt.bar(names,scores)
plt.ylabel('AUC')

#%% Single feature comparisons

def rebalance_g1(df,Ng1):
    #% Rebalance class
    g1_sampled = df_g1s[df_g1s['G1S_logistic'] == 0].sample(Ng1,replace=False)
    # df_g1s[df_g1s['Phase' == 'G1']].sample
    sg2 = df_g1s[df_g1s['G1S_logistic'] == 1]

    df_g1s_balanced = pd.concat((g1_sampled,sg2),ignore_index=True)
    return df_g1s_balanced

Ng1 = 150
Niter = 100
y = df_['G1S_logistic']
# df_g1s = df_g1s.drop(columns='G1S_logistic')
names = df_g1s.columns.drop('G1S_logistic')

AUCs = pd.DataFrame(index=range(Niter),columns=names)
log_losses = pd.DataFrame(index=range(Niter),columns=names)

for feature in names:
    
    for i in range(Niter):
        
        # Balance classes
        df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
        
        y = df_g1s_balanced['G1S_logistic'].values
        X = scale(df_g1s_balanced[feature].values.reshape(-1,1))
        model = LogisticRegression().fit(X,y)
        
        AUCs.loc[i,feature] = roc_auc_score(y,model.predict(X))
        log_losses.loc[i,feature] = log_loss(y,model.predict(X))
    
        
sb.catplot(AUCs,kind='box')
plt.ylabel('Mean AUC')


