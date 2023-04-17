#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:27:52 2022

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

#%%

X = df_g1s.drop(columns=['time_g1s'])
y = df_g1s['time_g1s']

from sklearn import linear_model
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)

reg = linear_model.RidgeCV()
reg = reg.fit(X_train,y_train)


results = pd.Series(reg.coef_, reg.feature_names_in_)
results.sort_values().plot.bar()

#%% Cross-validation


Niter = 100

frac_withhold = 0.2
N = len(df_g1s_balanced)

models = []
random_models = []
R_sq = np.zeros(Niter)

for i in tqdm(range(Niter)):
    
    num_withold = np.round(frac_withhold * N).astype(int)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=frac_withhold)
    
    reg = linear_model.RidgeCV()
    this_model = reg.fit(X_train,y_train)
    models.append(this_model)
    
    # predict on the withheld data
    ypred = this_model.predict(X_test)
    Rsq[i] = metrics.r2_score(y_test,ypred)
    
plt.hist(Rsq)

mlr = this_model
result = permutation_importance(
    mlr, X_test, y_test, n_repeats=100, random_state=42, n_jobs=2
)
mlr_importances = pd.Series(result.importances_mean, index=df_g1s.drop(columns=['time_g1s']).columns)

forest_importances.plot.bar(yerr=result.importances_std)
fig.tight_layout()
plt.show()

#%% Random forest regression

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

Niter = 100
sum_res = np.zeros(Niter)
Rsq = np.zeros(Niter)
importance = np.zeros((Niter,df_g1s.shape[1]-1))

for i in tqdm(range(Niter)):
    
    forest = RandomForestRegressor(n_estimators=100, random_state=i)
    
    X = df_g1s.drop(columns='time_g1s'); y = df_g1s['time_g1s']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
    
    forest.fit(X_train,y_train)
    
    y_pred = forest.predict(X_test)
    residuals = y_pred - y_test
    sum_res[i] = residuals.sum()
    Rsq[i] = np.corrcoef(y_pred,y_test)[0,1]
    importance[i,:] = forest.feature_importances_
    
plt.hist(Rsq)

imp = pd.DataFrame(importance)
imp.columns = df_g1s.columns.drop('time_g1s')

plt.figure()
sb.barplot(data=imp.melt(value_vars=imp.columns),x='variable',y='value');
plt.xticks(rotation=45);plt.ylabel('Importance')

#%% Use Region1 -> Pred Region2

df1_ = df_g1s[df_['Region'] == 1]
df2_ = df_g1s[df_['Region'] == 2]

X_train = df1_.drop(columns='time_g1s')
y_train = df1_['time_g1s']
X_test = df2_.drop(columns='time_g1s')
y_test = df2_['time_g1s']

model_region1 = smf.ols(f'time_g1s ~ ' + str.join(' + ',
                                      df1_.columns.drop(['time_g1s'])),data=df1_).fit()

# model_region1 = smf.ols(f'sgr ~ ' + str.join(' + ',
#                                       df1_.columns.drop(['sgr'])),data=df1_).fit_regularized('sqrt_lasso')

ypred_mlr = model_region1.predict(X_test)

forest_r1 = RandomForestRegressor(n_estimators=100, random_state=42)
forest_r1.fit(X_train,y_train)

ypred_rf = forest_r1.predict(X_test)



results = pd.DataFrame()
results['Measured'] = y_test
results['MLR pred'] = ypred_mlr
results['RF pred'] = ypred_rf
sb.regplot(data = results, x='Measured',y='MLR pred')
sb.regplot(data = results, x='Measured',y='RF pred')

R2_mlr = np.corrcoef(results['Measured'],results['MLR pred'])[0,1]
print(f'R2_mlr = {R2_mlr}')
R2_rf = np.corrcoef(results['Measured'],results['RF pred'])[0,1]
print(f'R2_rf = {R2_rf}')
plt.legend(['MLR','MLR mean','MLR confint','RandForest','RF mean','RF conf int'])



