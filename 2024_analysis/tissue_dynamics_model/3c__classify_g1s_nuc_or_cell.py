#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:55:35 2024

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from basicUtils import *
from os import path
from tqdm import tqdm

from numpy import random
from sklearn.metrics import roc_curve, auc, confusion_matrix, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from sklearn.linear_model import LogisticRegression

def keep_only_first_sg2(df):
    collated_first_s = []
    # Groupby CellID and then drop non-first S phase
    for (cellID,frame),cell in df.groupby(['cellID','region']):
        if cell.G1S_logistic.sum() > 1:
            first_sphase_frame = np.where(cell.G1S_logistic)[0][0]
            collated_first_s.append(cell.iloc[0:first_sphase_frame+1])
        else:
            collated_first_s.append(cell)
    return pd.concat(collated_first_s,ignore_index=True)

def rebalance_g1(df,Ng1):
    #% Rebalance class
    g1_sampled = df_g1s[df_g1s['G1S_logistic'] == 0].sample(Ng1,replace=False)
    # df_g1s[df_g1s['Phase' == 'G1']].sample
    sg2 = df_g1s[df_g1s['G1S_logistic'] == 1]

    df_g1s_balanced = pd.concat((g1_sampled,sg2),ignore_index=True)
    return df_g1s_balanced


def run_cross_validation(X,y,split_ratio,model,random_state=42):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=split_ratio,random_state=random_state)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    C = confusion_matrix(y_test,y_pred,normalize='all')
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    AUC = auc(fpr,tpr)
    AP = average_precision_score(y_test,y_pred)
    return C, AUC, AP

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_g1s.csv',index_col=0)

df_g1s = keep_only_first_sg2(df_g1s)

df_g1s_cell = df_g1s.drop(columns=['time_g1s','fucci_int_12h','cellID','diff','region','nuc_vol_sm'])
df_g1s_nuc = df_g1s.drop(columns=['time_g1s','fucci_int_12h','cellID','diff','region','vol_sm'])

#%% Logistic for G1/S transition using CELL VOL: skip all non-first SG2 timepoints
# Random rebalance with 1:1 ratio
# No cross-validation, in-model estimates only

Ng1 = 150
Niter = 100

coefficients = np.ones((Niter,df_g1s_cell.shape[1]-1)) * np.nan
li = np.ones((Niter,df_g1s_cell.shape[1]-1)) * np.nan
ui = np.ones((Niter,df_g1s_cell.shape[1]-1)) * np.nan
pvalues = np.ones((Niter,df_g1s_cell.shape[1]-1)) * np.nan

for i in range(Niter):
    
    #% Rebalance class
    df_g1s_balanced = rebalance_g1(df_g1s_cell,Ng1)
    
    ############### G1S logistic as function of age ###############
    try:
        model_g1s = smf.logit('G1S_logistic ~ ' + str.join(' + ',df_g1s_cell.columns[df_g1s_cell.columns != 'G1S_logistic']),
                      data=df_g1s_balanced).fit(maxiter=1000)
    except:
        print('--')
        continue
    
    params = model_g1s.params.drop('Intercept')
    pvals = model_g1s.pvalues.drop('Intercept')

    coefficients[i,:] = params.values
    li[i,:] = model_g1s.conf_int()[0].drop('Intercept')
    ui[i,:] = model_g1s.conf_int()[1].drop('Intercept')
    pvalues[i,:] = pvals.values
    

coefficients = pd.DataFrame(coefficients,columns = df_g1s_cell.columns.drop('G1S_logistic')).dropna()
pvalues = pd.DataFrame(pvalues,columns = df_g1s_cell.columns.drop('G1S_logistic')).dropna()
li = pd.DataFrame(li,columns = df_g1s_cell.columns.drop('G1S_logistic')).dropna()
ui = pd.DataFrame(ui,columns = df_g1s_cell.columns.drop('G1S_logistic')).dropna()

# Effect size v. pvalue
plt.errorbar(x=coefficients.mean(axis=0), y=-np.log10(pvalues).mean(axis=0),
             xerr = coefficients.std(axis=0),
             yerr = np.log10(pvalues).std(axis=0),
             fmt='bo')

# Label sig variables
sig_params = pvalues.columns[-np.log10(pvalues).mean(axis=0) > -np.log10(0.01)]
for p in sig_params:
    plt.text(coefficients[p].mean() + 0.01, -np.log10(pvalues[p]).mean() + 0.01, p)
plt.hlines(-np.log10(0.01),xmin=-1.5,xmax=2.0,color='r')
plt.xlabel('Regression coefficient')
plt.ylabel('-Log(P)')

params = pd.DataFrame()
params['var'] = coefficients.columns.values
params['coef'] = coefficients.mean(axis=0).values
params['li'] = li.mean(axis=0).values
params['ui'] = ui.mean(axis=0).values

params['err'] = params['ui'] - params['coef']
params['effect size'] = params['coef']

order = np.argsort( np.abs(params['coef']) )[::-1][0:5]
params = params.iloc[order]

plt.figure()
plt.bar(range(len(params)),params['coef'],yerr=params['err'])
plt.xticks(range(5),params['var'],rotation=30)

#%% Logistic for G1/S transition using NUCLEAR VOL: skip all non-first SG2 timepoints
# Random rebalance with 1:1 ratio
# No cross-validation, in-model estimates only

Ng1 = 150
Niter = 100

coefficients = np.ones((Niter,df_g1s_nuc.shape[1]-1)) * np.nan
li = np.ones((Niter,df_g1s_nuc.shape[1]-1)) * np.nan
ui = np.ones((Niter,df_g1s_nuc.shape[1]-1)) * np.nan
pvalues = np.ones((Niter,df_g1s_nuc.shape[1]-1)) * np.nan

for i in range(Niter):
    
    #% Rebalance class
    df_g1s_balanced = rebalance_g1(df_g1s_nuc,Ng1)
    
    ############### G1S logistic as function of age ###############
    try:
        model_g1s = smf.logit('G1S_logistic ~ ' + str.join(' + ',df_g1s_nuc.columns[df_g1s_nuc.columns != 'G1S_logistic']),
                      data=df_g1s_balanced).fit(maxiter=1000)
    except:
        print('--')
        continue
    
    # plt.figure()
    params = model_g1s.params.drop('Intercept')
    pvals = model_g1s.pvalues.drop('Intercept')

    coefficients[i,:] = params.values
    li[i,:] = model_g1s.conf_int()[0].drop('Intercept')
    ui[i,:] = model_g1s.conf_int()[1].drop('Intercept')
    pvalues[i,:] = pvals.values

coefficients = pd.DataFrame(coefficients,columns = df_g1s_nuc.columns.drop('G1S_logistic')).dropna()
pvalues = pd.DataFrame(pvalues,columns = df_g1s_nuc.columns.drop('G1S_logistic')).dropna()
li = pd.DataFrame(li,columns = df_g1s_nuc.columns.drop('G1S_logistic')).dropna()
ui = pd.DataFrame(ui,columns = df_g1s_nuc.columns.drop('G1S_logistic')).dropna()

# Effect size v. pvalue
plt.errorbar(x=coefficients.mean(axis=0), y=-np.log10(pvalues).mean(axis=0),
             xerr = coefficients.std(axis=0),
             yerr = np.log10(pvalues).std(axis=0),
             fmt='bo')

# Label sig variables
sig_params = pvalues.columns[-np.log10(pvalues).mean(axis=0) > -np.log10(0.01)]
for p in sig_params:
    plt.text(coefficients[p].mean() + 0.01, -np.log10(pvalues[p]).mean() + 0.01, p)
plt.hlines(-np.log10(0.01),xmin=-1.5,xmax=2.0,color='r')
plt.xlabel('Regression coefficient')
plt.ylabel('-Log(P)')

params = pd.DataFrame()
params['var'] = coefficients.columns.values
params['coef'] = coefficients.mean(axis=0).values
params['li'] = li.mean(axis=0).values
params['ui'] = ui.mean(axis=0).values

params['err'] = params['ui'] - params['coef']
params['effect size'] = params['coef']

order = np.argsort( np.abs(params['coef']) )[::-1][0:5]
params = params.iloc[order]

plt.figure()
plt.bar(range(len(params)),params['coef'],yerr=params['err'])
plt.xticks(range(5),params['var'],rotation=30)


#%% Swich to sklearn and 

from sklearn import metrics

Niter = 100
auc_scores = pd.DataFrame(index=range(Niter))
log_loss_scores = pd.DataFrame(index=range(Niter))

for i in tqdm(range(Niter)):
    
    # Cell volume
    df_g1s_balanced = rebalance_g1(df_g1s_cell,Ng1)
    
    y = df_g1s_balanced['G1S_logistic'].values
    X = df_g1s_balanced.drop(columns = 'G1S_logistic')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
    
    model_cell = LogisticRegression(max_iter=2000).fit(X_train,y_train)
    y_pred = model_cell.predict(X_test)
    auc_scores.loc[i,'Cell'] = metrics.roc_auc_score(y_test,y_pred)
    log_loss_scores.loc[i,'Cell'] = metrics.log_loss(y_test,y_pred)
    
    # Cell volume
    df_g1s_balanced = rebalance_g1(df_g1s_nuc,Ng1)
    y = df_g1s_balanced['G1S_logistic']
    X = df_g1s_balanced.drop(columns = 'G1S_logistic')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
    
    model_nuc = LogisticRegression(max_iter=2000).fit(X_train,y_train)
    y_pred = model_cell.predict(X_test)
    auc_scores.loc[i,'Nucleus'] = metrics.roc_auc_score(y_test,y_pred)
    log_loss_scores.loc[i,'Nucleus'] = metrics.log_loss(y_test,y_pred)
    

sb.catplot(auc_scores,kind='box')
plt.ylabel('AUC (highest is 1)')
sb.catplot(log_loss_scores,kind='box')
plt.ylabel('Log-loss (lower is better)')
    



