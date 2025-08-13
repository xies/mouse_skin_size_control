#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:13:02 2024

@author: xies
"""

import numpy as np
import pandas as pd
from os import path
import seaborn as sb
import matplotlib.pyplot as plt

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/HMECs/HMEC DFB tracked data/'

time_to_g1s = pd.read_excel(path.join(dirname,'full_trace/individuals_full_trace.xlsx'),
                    sheet_name='Ages_wrt_g1s', header=None)
size = pd.read_excel(path.join(dirname,'full_trace/individuals_full_trace.xlsx'),
                     sheet_name='Volume', header=None)
rb = pd.read_excel(path.join(dirname,'full_trace/individuals_full_trace.xlsx'),
                     sheet_name='RB', header=None)

cellIDs = time_to_g1s.columns

df = []
for ID in cellIDs:
    _df = pd.DataFrame()
    I = ~np.isnan(time_to_g1s[ID])
    
    _df['Size'] = size.loc[I,ID]
    _df['Time to G1S'] = time_to_g1s.loc[I,ID]
    _df['RB'] = rb.loc[I,ID]
    _df['Frame'] = size.loc[I,ID].index
    _df['CellID'] = ID+1
    df.append(_df)
df = pd.concat(df,ignore_index=True)
df['G1S_logistic'] = df['Time to G1S'] > 0
df['RB conc'] = df['RB'] / df['Size']
df = df.dropna()

# sb.regplot(df,x='Size',y='G1S_logistic',logistic=True,y_jitter=0.1)
# sb.regplot(df,x='RB conc',y='G1S_logistic',logistic=True,y_jitter=0.1)

#%%

def balance_phase(df):
    (_,g1),(_,sg2) = df.groupby('G1S_logistic')
    L = min(len(g1),len(sg2))
    sg2 = sg2.sample(L)
    g1 = g1.sample(L)
    return pd.concat((g1,sg2),ignore_index=True)

def subsample_from_each_cell(df):
    cells = [c for _,c in df.groupby('CellID')]
    sampled = []
    for c in cells:
        pre_g1 = c[c['G1S_logistic'] == False]
        post_g1 = c[c['G1S_logistic'] == True]
        pre_g1 = pre_g1[::30]
        post_g1 = post_g1[::30]
        sampled.append(pre_g1)
        sampled.append(post_g1)
    return pd.concat(sampled,ignore_index=True)

from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score,average_precision_score

#%%

Niter = 100

M = np.zeros((Niter,2,2))
AUC = pd.DataFrame()
AP = pd.DataFrame()

for i in range(Niter):
    
    sampled = subsample_from_each_cell(df)
    
    X = scale(sampled[['Size']]).reshape(-1,1)
    y = sampled['G1S_logistic']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.4)
    
    ref = linear_model.LogisticRegression()
    ref.fit(X_train,y_train)
    
    y_pred = ref.predict(X_test)
    
    AUC.at[i,'Real model'] = roc_auc_score(y_test,y_pred)
    AP.at[i,'Real model'] = average_precision_score(y_test,y_pred)
    M[i,...] = confusion_matrix(y_test,y_pred) / len(y_test)

avgM = M.mean(axis=0)
sb.heatmap(avgM/avgM.sum(), annot=True)

for i in range(Niter):
    Xrand = np.random.randn(len(X)).reshape(-1,1)
    X_train,X_test,y_train,y_test = train_test_split(Xrand,y,test_size=.2)
    
    ref_rand = linear_model.LogisticRegression()
    ref_rand.fit(X_train,y_train)
    
    y_pred = ref.predict(X_test)
    
    AUC.at[i,'Random model'] = roc_auc_score(y_test,y_pred)
    AP.at[i,'Random model'] = average_precision_score(y_test,y_pred)

#%

plt.figure()
plt.hist(AUC['Real model'])
plt.hist(AUC['Random model'])
plt.legend(['Real','Random'])
plt.xlabel('AUC')


plt.figure()
plt.hist(AP['Real model'])
plt.hist(AP['Random model'])
plt.legend(['Real','Random'])
plt.xlabel('AP')

