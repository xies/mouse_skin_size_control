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
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale 

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_g1s.csv',index_col=0)

#%% Robust LM for smoothed specific growth rate

from numpy.linalg import eig

############### OLS for specific growth rate ###############
model_rlm = smf.rlm(f'sgr ~ ' + str.join(' + ',
                                      df_g1s.columns.drop(['sgr'])),data=df_g1s).fit()
print(model_rlm.summary())

# model_rlm_ridge = smf.ols(f'sgr ~ ' + str.join(' + ',
#                                       df_g1s.columns.drop(['sgr'])),data=df_g1s).fit_regularized('sqrt_lasso')

print(model_rlm.summary())
C = model_rlm.cov_params()
sb.heatmap(C,xticklabels=True,yticklabels=True)
L,D = eig(C)

print(f'Covariance eigenvalue ratio: {L.max()/L.min()}')

plt.figure()

plt.scatter(model_rlm.params[model_rlm.params > 0],-np.log10(model_rlm.pvalues[model_rlm.params > 0]),color='b')
plt.scatter(model_rlm.params[model_rlm.params < 0],-np.log10(model_rlm.pvalues[model_rlm.params < 0]),color='r')
sig_params = model_rlm.pvalues.index[model_rlm.pvalues < 0.05]

for p in sig_params:
    plt.text(model_rlm.params[p] + 0.01, -np.log10(model_rlm.pvalues[p]), p)

plt.hlines(-np.log10(0.05),xmin=-1,xmax=1,color='r')
plt.xlabel('Regression coefficient')
plt.ylabel('-Log(P)')

#%% Plot important parameters

from scipy.stats import stats

params = pd.DataFrame()

# Total corrcoef
X,Y = nonan_pairs(model_rlm.predict(df_g1s), df_g1s['sgr'])
R,P = stats.pearsonr(X,Y)
Rsqfull = R**2

params['var'] = model_rlm.params.index
params['coef'] = model_rlm.params.values
params['li'] = model_rlm.conf_int()[0].values
params['ui'] = model_rlm.conf_int()[1].values
params['pvals'] = model_rlm.pvalues.values

params['err'] = params['ui'] - params['coef'] 
params['effect size'] = np.sqrt(params['coef']**2 /(1-Rsqfull))

order = np.argsort( np.abs(params['coef']) )[::-1][0:10]
params = params.iloc[order]

# plt.bar(range(len(params)),params['coef'],yerr=params['err'])
params.plot.bar(y='coef',yerr='err',x='var')
plt.ylabel('Regression coefficients')



#%% Cross-validation on the same dataset

# from numpy import random

# Niter = 100
# N,P = df_g1s.shape
# frac_withhold = 0.1

# models = []
# MSE = np.zeros(Niter)
# Rsq = np.zeros(Niter)
# Rsq_random = np.zeros(Niter)

# coefficients = np.zeros((Niter,P-1))
# pvalues = np.zeros((Niter,P-1))

# for i in tqdm(range(Niter)):
    
#     # Withold data
#     num_withold = np.round(frac_withhold * N).astype(int)
#     idx_subset = random.choice(N, size = num_withold, replace=False)
#     Iwithheld = np.zeros(N).astype(bool)
#     Iwithheld[idx_subset] = True
#     Isubsetted = ~Iwithheld
#     df_subsetted = df_g1s.loc[Isubsetted]
#     df_withheld = df_g1s.loc[Iwithheld]
    
#     this_model = smf.rlm(f'sgr ~ ' + str.join(' + ',df_subsetted.columns.drop('sgr')),
#                          data=df_subsetted).fit()
#     models.append(this_model)
    
#     # predict on the withheld data
#     ypred = this_model.predict(df_withheld)
#     res = df_withheld['sgr'] - ypred
#     MSE[i] = np.nansum( res ** 2 )

#     R = np.corrcoef(*nonan_pairs(ypred, df_withheld['sgr']))[0,1]
#     Rsq[i] = R**2
    
    
#     # Generate a 'random' model
#     df_rand = df_subsetted.copy()
#     for col in df_rand.columns.drop('sgr'):
#         df_rand[col] = random.randn(N-num_withold)
        
#     random_model = smf.rlm(f'sgr ~ ' + str.join(' + ',df_rand.columns.drop('sgr')),
#                            data=df_rand).fit()
    
#     # predict on the withheld data
#     ypred = random_model.predict(df_withheld)
#     res = df_withheld['sgr'] - ypred
#     MSE[i] = np.nansum( res ** 2 )
    
#     R = np.corrcoef(*nonan_pairs(ypred, df_withheld['sgr']))[0,1]
#     Rsq_random[i] = R**2
    
#     coefficients[i,:] = this_model.params.drop('Intercept')
#     pvalues[i,:] = this_model.pvalues.drop('Intercept')

# coefficients = pd.DataFrame(coefficients,columns=df_g1s.columns.drop('sgr'))
# pvalues = pd.DataFrame(pvalues,columns=df_g1s.columns.drop('sgr'))
    
# # Plot R2
# plt.hist(Rsq.flatten())
# plt.hist(Rsq_random.flatten())
# plt.legend(['Empirical','Random']),plt.xlabel('R2 on test set')

# # Plot coefficient variance
# plt.figure()
# plt.errorbar(coefficients.mean(axis=0),y=-np.log10(pvalues).mean(axis=0),
#              xerr = coefficients.std(axis=0),yerr= -np.log10(pvalues).std(axis=0),
#              fmt='b*')
# sig_params = coefficients.columns[pvalues.mean(axis=0) < 0.05]
# for p in sig_params:
#     plt.text(coefficients[p].mean() + 0.1, -np.log10(pvalues[p]).mean() + 0.01, p)
# plt.hlines(-np.log10(0.05),xmin=-1,xmax=1,color='r')
# plt.xlabel('Reg coefficient');plt.ylabel('-Log10(P)')

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
    
    X = df_g1s.drop(columns='sgr'); y = df_g1s['sgr']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
    
    forest.fit(X_train,y_train)
    
    y_pred = forest.predict(X_test)
    residuals = y_pred - y_test
    sum_res[i] = residuals.sum()
    Rsq[i] = np.corrcoef(y_pred,y_test)[0,1]
    importance[i,:] = forest.feature_importances_
    
plt.hist(Rsq)

imp = pd.DataFrame(importance)
imp.columns = df_g1s.columns.drop('G1S_logistic')

plt.figure()
sb.barplot(data=imp.melt(value_vars=imp.columns),x='variable',y='value');
plt.xticks(rotation=45);plt.ylabel('Importance')

#%% Use Region1 -> Pred Region2

df1_ = df_g1s[df_['Region'] == 1]
df2_ = df_g1s[df_['Region'] == 2]

X_train = df1_.drop(columns='sgr')
y_train = df1_['sgr']
X_test = df2_.drop(columns='sgr')
y_test = df2_['sgr']

model_region1 = smf.glm(f'sgr ~ ' + str.join(' + ',
                                      df1_.columns.drop(['sgr'])),data=df1_).fit()

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



