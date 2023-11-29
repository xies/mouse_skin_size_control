#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:18:35 2022

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
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.preprocessing import scale 
from scipy.stats import stats

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_g1s.csv',index_col=0)

df_g1s = df_g1s.drop(columns='time_g1s')
df_g1s = df_g1s.drop(columns='fucci_int_24h')

#%% Robust LM for smoothed specific growth rate

from numpy.linalg import eig

############### OLS for specific growth rate ###############
# model_rlm = smf.rlm(f'sgr ~ ' + str.join(' + ',
#                                       df_g1s.columns.drop(['sgr'])),data=df_g1s).fit()
# print(model_rlm.summary())

model_rlm = smf.logit(f'G1S_logistic ~ ' + str.join(' + ',
                                      df_g1s.columns.drop(['G1S_logistic'])),data=df_g1s).fit()

params = pd.DataFrame()

# Total corrcoef
# X,Y = nonan_pairs(model_rlm.predict(df_g1s), df_g1s['sgr'])
# R,P = stats.pearsonr(X,Y)
# Rsqfull = R**2

params['var'] = model_rlm.params.index
params['coef'] = model_rlm.params.values
# params['li'] = model_rlm.conf_int()[0].values
# params['ui'] = model_rlm.conf_int()[1].values
# params['pvals'] = model_rlm.pvalues.values

order = np.argsort(params['coef'].abs())[::-1]
param_names_in_order = params.iloc[order]['var'].values

#%% Recursive feature drop

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn import linear_model

X = df_g1s.drop(columns='G1S_logistic')
X['Intercept'] = 1
y = df_g1s['G1S_logistic']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)


for num_feat in np.arange(1,10):
    selector = RFE(linear_model.LogisticRegression(), n_features_to_select=num_feat)
    selector = selector.fit(X_train,y_train)
    print(f'Num feature = {num_feat}: {X.columns.values[selector.support_]}')

#%% Drop feature in order of importance and look at R2 (cross replicate?)
#NB: Something is weird... need to figure out

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
    # model = linear_model.LogisticRegression()
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train,y_train)
    largest_effect_param = X_train.columns[np.argmax(np.abs(model.feature_importances_))]
    print(f'{i}: {largest_effect_param}')
    features2drop.append(largest_effect_param)
    
    # res[i] = model.score(X_test,y_test)
        
    # Compute some scoring metrics
    # AUC
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    AUCs[i] = auc(fpr,tpr)
    
    #APs
    APs[i] = average_precision_score(y_test,y_pred)


# Random model prediction
X_random = pd.DataFrame(random.randn(*X_sorted.shape),index=X_sorted.index)
X_train,X_test,y_train,y_test = train_test_split(X_random,y,test_size=0.7,random_state = 42)
model_random = RandomForestClassifier(n_estimators=100, random_state=42)
model_random.fit(X_train,y_train)
y_pred_random = model_random.predict(X_test)

# Collate the scores
scores = pd.DataFrame(AUCs, index=features2drop,columns = ['AUC'])
scores['AP'] = APs

fpr, tpr, _ = roc_curve(y_test, y_pred_random)
scores.at['Random','AUC'] = auc(fpr,tpr)
scores.at['Random','AP'] = average_precision_score(y_test,y_pred_random)

scores.index = (['Full',*features2drop[:-1],'Random'])
scores['Dropped features'] = scores.index

scores.plot.bar(x='Dropped features',y=['AUC','AP'])
plt.ylim([0.3,1])
# plt.tight_layout()

# model_scores = pd.DataFrame(-np.diff(AUCs),index=features2drop[:-1],columns=['Importance'])
# model_scores['Name'] = model_scores.index
# sb.barplot(data=model_scores,x='Name',y='Importance');

#%% AUC for volume feature drop

Niter = 100

AUC_full = np.zeros(Niter)
AUC_random = np.zeros(Niter)
AUC_no_vol_sm = np.zeros(Niter)
AUC_no_sgr = np.zeros(Niter)

for i in tqdm(range(Niter)):
    
    X_train,X_test,y_train,y_test = train_test_split(X_sorted,y,test_size=0.7,random_state = i)
    
    # Full model
    model_full = RandomForestClassifier(n_estimators=100, random_state=i)
    model_full.fit(X_train,y_train)
    y_pred_full = model_full.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_full)
    AUC_full[i] = auc(fpr,tpr)
    
    # Vol-drop model
    X_train,X_test,y_train,y_test = train_test_split(X_sorted.drop(columns='vol_sm'),y,test_size=0.7,random_state = i)
    model_no_vol_sm = RandomForestClassifier(n_estimators=100, random_state=42)
    model_no_vol_sm.fit(X_train,y_train)
    y_pred_no_vol_sm = model_no_vol_sm.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_no_vol_sm)
    AUC_no_vol_sm[i] = auc(fpr,tpr)
    
    # Vol-drop model
    X_train,X_test,y_train,y_test = train_test_split(X_sorted.drop(columns='sgr'),y,test_size=0.7,random_state = i)
    model_no_sgr = RandomForestClassifier(n_estimators=100, random_state=42)
    model_no_sgr.fit(X_train,y_train)
    y_pred_no_sgr = model_no_sgr.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_no_sgr)
    AUC_no_sgr[i] = auc(fpr,tpr)
    
    # Random model
    X_random = pd.DataFrame(random.randn(*X_sorted.shape),index=X_sorted.index)
    X_train,X_test,y_train,y_test = train_test_split(X_random,y,test_size=0.7,random_state = 42)
    model_random = RandomForestClassifier(n_estimators=100, random_state=i)
    model_random.fit(X_train,y_train)
    y_pred_random = model_random.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_random)
    AUC_random[i] = auc(fpr,tpr)


plt.hist(AUC_full)
plt.hist(AUC_no_vol_sm)
plt.hist(AUC_no_sgr)
plt.hist(AUC_random)

plt.legend(['Full','No cell size','No SGR','Random'])
plt.xlabel('AUC')
