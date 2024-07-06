#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:07:54 2022

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
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale 
from scipy.stats import stats

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_g1s.csv',index_col=0)

#%% Robust LM for smoothed specific growth rate

from numpy.linalg import eig

############### OLS for specific growth rate ###############
# model_rlm = smf.rlm(f'sgr ~ ' + str.join(' + ',
#                                       df_g1s.columns.drop(['sgr'])),data=df_g1s).fit()
# print(model_rlm.summary())

model_rlm = smf.ols(f'sgr ~ ' + str.join(' + ',
                                      df_g1s.columns.drop(['sgr'])),data=df_g1s).fit_regularized('sqrt_lasso')

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

X = df_g1s.drop(columns='sgr')
X['Intercept'] = 1
y = df_g1s['sgr']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)


for num_feat in np.arange(1,10):
    selector = RFE(linear_model.Ridge(), n_features_to_select=num_feat)
    selector = selector.fit(X_train,y_train)
    print(f'Num feature = {num_feat}: {X.columns.values[selector.support_]}')

#%% Drop feature in order of importance and look at R2 (cross replicate?)
#NB: Something is weird... need to figure out

# y = random.randn((1000))
# X = random.randn(10,1000)
# X = np.vstack((y + 0.1*randn(1000),X)).T

# X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state = 42)
X = df_g1s.drop(columns='sgr')
X['Intercept'] = 1
X_sorted = X[param_names_in_order]
y = df_g1s['sgr']

Rsq = np.zeros(23)
res = np.zeros(23)

features2drop = []

for i in range(23):

    X_dropped = X_sorted.drop(columns=features2drop)    
    X_train,X_test,y_train,y_test = train_test_split(X_dropped,y,test_size=0.5,random_state = 42)

    # Compute current model
    model = linear_model.Ridge()
    model.fit(X_train,y_train)
    largest_effect_param = X_train.columns[np.argmax(np.abs(model.coef_))]
    print(f'{i}: {largest_effect_param}')
    features2drop.append(largest_effect_param)
    
    # Find the largest effect size predcitor
    
    # ypred_mlr = model.predict(X_test)
    res[i] = model.score(X_test,y_test)
    
    
    
