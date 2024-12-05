#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:23:15 2024

@author: xies
"""

import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

dt = 1./6
deci_factor = 30

filename = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/HMECs/HMEC DFB tracked data/full_trace/individuals_full_trace.xlsx'

df = pd.read_excel(filename,sheet_name='RB', header=None).melt(var_name='cellID',value_name='RB')
size = pd.read_excel(filename,sheet_name='Size', header=None).melt(var_name='cellID',value_name='size')
time_to_g1s = pd.read_excel(filename,sheet_name='Ages_wrt_g1s', header=None).melt(var_name='cellID',value_name='time_to_g1s')

df['Size'] = size['size']
df['Time to G1S'] = time_to_g1s['time_to_g1s']
df['RB conc'] = df['RB'] / df['Size']
df['G1S_logistic'] = df['Time to G1S'] > 0

df = df[df['Size'] < 250000] # 3 outlier points

df = df.dropna()

#%% Subsample by decimation factor

cells = [c.dropna() for _,c in df.groupby('cellID')]

df_decimated = []
for c in cells:
    df_decimated.append(c[::deci_factor])

df_decimated = pd.concat(df_decimated,ignore_index=True)

def balance_phase(df):
    (_,g1),(_,sg2) = df.groupby('G1S_logistic')
    L = min(len(g1),len(sg2))
    sg2 = sg2.sample(L)
    g1 = g1.sample(L)
    return pd.concat((g1,sg2),ignore_index=True)
    
#%%

Niter = 100

from statsmodels.api import Logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score

AP = pd.DataFrame()
AUC = pd.DataFrame()
Cmlr = np.zeros((Niter,2,2))

for i in range(Niter):
    
    # df_bal = df_decimated
    df_bal = balance_phase(df_decimated)
    
    X = df_bal['Size'].values.reshape(-1,1)
    y = df_bal['G1S_logistic'].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.1)
    
    
    mlr = LogisticRegression().fit(X_train,y_train)
    # mlr = Logit(y_train,X_train).fit()
    
    y_pred = mlr.predict(X_test)
    
    AUC.at[i,'data'] = roc_auc_score(y_test,y_pred)
    AP.at[i,'data'] = average_precision_score(y_test,y_pred)
    Cmlr[i,...] = confusion_matrix(y_pred,y_test)/len(y_test)
    
    Xrand = np.random.randn(len(X_train)).reshape(-1,1)
    mlr_rand = LogisticRegression().fit(Xrand,y_train)
    y_pred = mlr_rand.predict(X_test)
    
    AUC.at[i,'random'] = roc_auc_score(y_test,y_pred)
    AP.at[i,'random'] = average_precision_score(y_test,y_pred)

plt.figure()
sb.histplot(AUC,bins=25); plt.xlabel('AUC')
plt.figure()
sb.histplot(AP,bins=25); plt.xlabel('AP')
plt.figure()
sb.heatmap(Cmlr.mean(axis=0),annot=True)
    
    
    
    
    