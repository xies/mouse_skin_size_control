#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:18:35 2022

@author: xies
"""


import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from basicUtils import *
from os import path
from tqdm import tqdm

from numpy import random
from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale 
from scipy.stats import stats
from sklearn.ensemble import RandomForestClassifier

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

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

df_g1s = df_g1s.drop(columns='time_g1s')
df_g1s = df_g1s.drop(columns=['fucci_int_12h'])

#%% Robust LM for smoothed specific growth rate

model_rlm = smf.logit('G1S_logistic ~ ' + str.join(' + ',
                                      df_g1s.columns.drop(['G1S_logistic'])),data=df_g1s).fit()

params = pd.DataFrame()

params['var'] = model_rlm.params.index
params['coef'] = model_rlm.params.values

order = np.argsort(params['coef'].abs())[::-1]
param_names_in_order = params.iloc[order]['var'].values

#%% Drop feature in order of importance

from sklearn.linear_model import LogisticRegression

# y = random.randn((1000))
# X = random.randn(10,1000)
# X = np.vstack((y + 0.1*randn(1000),X)).T

X = df_g1s.drop(columns='G1S_logistic')
X['Intercept'] = 1
X_sorted = X[param_names_in_order]
y = df_g1s['G1S_logistic']

AUCs = np.zeros(10)
APs = np.zeros(10)

features2drop = []

for i in range(10):

    X_dropped = X_sorted.drop(columns=features2drop)    
    X_train,X_test,y_train,y_test = train_test_split(X_dropped,y,test_size=0.7,random_state = 42)

    # Compute current model
    model = LogisticRegression(random_state=42)
    model.fit(X_train,y_train)
    largest_effect_param = X_train.columns[np.abs(model.coef_).argmax()]
    print(f'{i}: {largest_effect_param}')
    features2drop.append(largest_effect_param)
    
    # Compute some scoring metrics
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    AUCs[i] = auc(fpr,tpr)
    
    #APs
    APs[i] = average_precision_score(y_test,y_pred)

# Collate the scores
scores = pd.DataFrame(AUCs, index=features2drop,columns = ['AUC'])
scores['AP'] = APs


model_scores = pd.DataFrame(-np.diff(AUCs),index=features2drop[:-1],columns=['Importance'])
model_scores['Name'] = model_scores.index
model_scores.sort_values(by='Importance').plot.bar()
plt.tight_layout()

#%% Recursive feature drop

from sklearn.feature_selection import RFE

X = df_g1s.drop(columns='G1S_logistic')
X['Intercept'] = 1
y = df_g1s['G1S_logistic']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

feature2drop = []
selector = RFE(RandomForestClassifier(n_estimators=100), n_features_to_select=1)
selector = selector.fit(X_train,y_train)
features_ranked_by_RFE = X_train.columns.values[selector.ranking_.argsort()]

#%% AUC for volume feature drop v. every other feature

Ng1 = 199
Niter = 100
frac_withheld = 0.1

features2drop = features_ranked_by_RFE[1:10]

AUC = pd.DataFrame(np.zeros((Niter,3+len(features2drop))),columns=np.hstack(['Full','No vol','Random',features2drop]))
AP = pd.DataFrame(np.zeros((Niter,3+len(features2drop))),columns=np.hstack(['Full','No vol','Random',features2drop]))

for i in tqdm(range(Niter)):
    
    df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
    y_balanced = df_g1s_balanced['G1S_logistic']
    df_g1s_balanced = df_g1s_balanced.drop(columns='G1S_logistic')
    
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    _, _AUC,_AP = run_cross_validation(df_g1s_balanced,y_balanced,frac_withheld,forest)
    AUC.at[i,'Full'] = _AUC; AP.at[i,'Full'] = _AP
    
    forest_no_vol = RandomForestClassifier(n_estimators=100, random_state=42)
    _, _AUC,_AP = run_cross_validation(df_g1s_balanced.drop(columns='vol_sm'),y_balanced,frac_withheld,forest_no_vol)
    AUC.at[i,'No vol'] = _AUC; AP.at[i,'No vol'] = _AP
    
    df_random = pd.DataFrame(random.randn(*df_g1s_balanced.shape))
    random_model = RandomForestClassifier(n_estimators=100, random_state=42)
    _,_AUC,_AP = run_cross_validation(df_random,y_balanced,frac_withheld,random_model)
    AUC.at[i,'Random'] = _AUC; AP.at[i,'Random'] = _AP
    
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    for j in range(len(features2drop)):
        _, _AUC,_AP = run_cross_validation(df_g1s_balanced.drop(columns=features2drop[j]),y_balanced,frac_withheld,forest)
        AUC.at[i,features2drop[j]] = _AUC; AP.at[i,features2drop[j]] = _AP

AUC.plot.hist();plt.xlabel('AUC')
AP.plot.hist();plt.xlabel('Average precision')

#%% Single features -> predict

Niter = 100

X = df_g1s.drop(columns='G1S_logistic')
X['Intercept'] = 1
X_sorted = X[param_names_in_order]
other_features2test = param_names_in_order[2:9]
y = df_g1s['G1S_logistic']

AUC_full = np.zeros(Niter)
AUC_random = np.zeros(Niter)
AUC_vol = np.zeros(Niter)
AUC_single = np.zeros((9,Niter))

for i in tqdm(range(Niter)):
    
    X_train,X_test,y_train,y_test = train_test_split(X_sorted,y,test_size=0.2,random_state = i)
    
    # Full model
    model_full = RandomForestClassifier(n_estimators=100, random_state=i)
    model_full.fit(X_train,y_train)
    y_pred_full = model_full.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_full)
    AUC_full[i] = auc(fpr,tpr)
    
    # Volume model
    model_vol = RandomForestClassifier(n_estimators=100, random_state=i)
    model_vol.fit(X_train[['vol_sm','Intercept']],y_train)
    y_pred_vol = model_vol.predict(X_test[['vol_sm','Intercept']])
    fpr,tpr,_ = roc_curve(y_test,y_pred_vol)
    AUC_vol[i] = auc(fpr,tpr)
    
    # Other single features
    for j in range(9):
        model_single = RandomForestClassifier(n_estimators=100, random_state=i)
        model_single.fit(X_train[[other_features2test[j],'Intercept']],y_train)
        y_pred_single = model_single.predict(X_test[[other_features2test[j],'Intercept']])
        
        fpr,tpr,_ = roc_curve(y_test,y_pred_single)
        AUC_single[j,i] = auc(fpr,tpr)
        
    # Random model
    X_random = pd.DataFrame(random.randn(*X_sorted.shape),index=X_sorted.index)
    X_train,X_test,y_train,y_test = train_test_split(X_random,y,test_size=0.7,random_state = 42)
    model_random = RandomForestClassifier(n_estimators=100, random_state=i)
    model_random.fit(X_train,y_train)
    y_pred_random = model_random.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_random)
    AUC_random[i] = auc(fpr,tpr)

#%%

plt.figure()
hist_weights = np.ones(Niter)/Niter
plt.hist(AUC_full,histtype='step',weights=hist_weights)
plt.hist(AUC_vol,histtype='step',weights=hist_weights)
for i in range(6):
    plt.hist(AUC_single[i,:],histtype='step',weights=hist_weights)
plt.hist(AUC_random,histtype='step',weights=hist_weights)
plt.legend(np.hstack([ ['Full','Only volume'],other_features2test[:6],['Random'] ]))
plt.xlabel('AUC')

