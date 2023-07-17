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
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
from basicUtils import *
from os import path

from numpy import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_g1s.csv',index_col=0)


X = df_g1s.drop(columns='sgr')
y = df_g1s['sgr']

#%% Model conditioning

from sklearn.covariance import EmpiricalCovariance, MinCovDet

Cemp = EmpiricalCovariance().fit(X)
Ccd = MinCovDet().fit(X)
plt.figure(); plt.subplot(1,2,1)
sb.heatmap(Cemp.covariance_,xticklabels=True,yticklabels=True)
plt.subplot(1,2,2)
sb.heatmap(Ccd.covariance_,xticklabels=True,yticklabels=True)
L,D = eig(Cemp.covariance_)
print(f'Covariance eigenvalue ratio (empirical): {L.max()/L.min()}')
L,D = eig(Ccd.covariance_)
print(f'Covariance eigenvalue ratio (mincovdet): {L.max()/L.min()}')

#%% Multilinear regression

Nsplit = 100
R2_mlr = np.zeros(Nsplit)
for i in range(Nsplit):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    lin_model = LinearRegression()
    
    lin_model.fit(X_train,y_train)
    ypred = lin_model.predict(X_test)
    R2_mlr[i] = r2_score(y_test,ypred)

print(f'Mean Rsq for MLR = {R2_mlr.mean()}')

#%% Random forest regression


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

Niter = 100
sum_res = np.zeros(Niter)
Rsq_rf = np.zeros(Niter)
importance = np.zeros((Niter,df_g1s.shape[1]-1))

for i in tqdm(range(Niter)):
    
    forest = RandomForestRegressor(n_estimators=100, random_state=i)
    
    X = df_g1s.drop(columns='sgr'); y = df_g1s['sgr']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
    
    forest.fit(X_train,y_train)
    
    y_pred = forest.predict(X_test)
    residuals = y_pred - y_test
    sum_res[i] = residuals.sum()
    Rsq_rf[i] = np.corrcoef(y_pred,y_test)[0,1]
    importance[i,:] = forest.feature_importances_
    
plt.hist(Rsq_rf)
print(f'Mean Rsq for RF = {Rsq_rf.mean()}')

imp = pd.DataFrame(importance)
imp.columns = df_g1s.columns.drop('G1S_logistic')

plt.figure()
sb.barplot(data=imp.melt(value_vars=imp.columns),x='variable',y='value');
plt.xticks(rotation=45);plt.ylabel('Importance')

#%% Depricated-- using SM; Robust LM for smoothed specific growth rate

# from numpy.linalg import eig

# ############### OLS for specific growth rate ###############
# model_rlm = smf.rlm(f'sgr ~ ' + str.join(' + ',
#                                       df_g1s.columns.drop(['sgr'])),data=df_g1s).fit()
# print(model_rlm.summary())

# # model_rlm_ridge = smf.ols(f'sgr ~ ' + str.join(' + ',
# #                                       df_g1s.columns.drop(['sgr'])),data=df_g1s).fit_regularized('sqrt_lasso')



# plt.figure()
# plt.scatter(model_rlm.params[model_rlm.params > 0],-np.log10(model_rlm.pvalues[model_rlm.params > 0]),color='b')
# plt.scatter(model_rlm.params[model_rlm.params < 0],-np.log10(model_rlm.pvalues[model_rlm.params < 0]),color='r')
# sig_params = model_rlm.pvalues.index[model_rlm.pvalues < 0.05]

# for p in sig_params:
#     plt.text(model_rlm.params[p] + 0.01, -np.log10(model_rlm.pvalues[p]), p)

# plt.hlines(-np.log10(0.05),xmin=-1,xmax=1,color='r')
# plt.xlabel('Regression coefficient')
# plt.ylabel('-Log(P)')

# %% Plot important parameters

# from scipy.stats import stats

# params = pd.DataFrame()

# # Total corrcoef
# X,Y = nonan_pairs(model_rlm.predict(df_g1s), df_g1s['sgr'])
# R,P = stats.pearsonr(X,Y)
# Rsqfull = R**2

# params['var'] = model_rlm.params.index
# params['coef'] = model_rlm.params.values
# params['li'] = model_rlm.conf_int()[0].values
# params['ui'] = model_rlm.conf_int()[1].values
# params['pvals'] = model_rlm.pvalues.values

# params['err'] = params['ui'] - params['coef'] 
# params['effect size'] = np.sqrt(params['coef']**2 /(1-Rsqfull))

# order = np.argsort( np.abs(params['coef']) )[::-1][0:10]
# params = params.iloc[order]

# # plt.bar(range(len(params)),params['coef'],yerr=params['err'])
# params.plot.bar(y='coef',yerr='err',x='var')
# plt.ylabel('Regression coefficients')


#%% Use Region1 -> Pred Region2

# df1_ = df_g1s[df_['Region'] == 1]
# df2_ = df_g1s[df_['Region'] == 2]

# X_train = df1_.drop(columns='sgr')
# y_train = df1_['sgr']
# X_test = df2_.drop(columns='sgr')
# y_test = df2_['sgr']

# model_region1 = smf.glm(f'sgr ~ ' + str.join(' + ',
#                                       df1_.columns.drop(['sgr'])),data=df1_).fit()

# # model_region1 = smf.ols(f'sgr ~ ' + str.join(' + ',
# #                                       df1_.columns.drop(['sgr'])),data=df1_).fit_regularized('sqrt_lasso')

# ypred_mlr = model_region1.predict(X_test)

# forest_r1 = RandomForestRegressor(n_estimators=100, random_state=42)
# forest_r1.fit(X_train,y_train)

# ypred_rf = forest_r1.predict(X_test)



# results = pd.DataFrame()
# results['Measured'] = y_test
# results['MLR pred'] = ypred_mlr
# results['RF pred'] = ypred_rf
# sb.regplot(data = results, x='Measured',y='MLR pred')
# sb.regplot(data = results, x='Measured',y='RF pred')

# R2_mlr = np.corrcoef(results['Measured'],results['MLR pred'])[0,1]
# print(f'R2_mlr = {R2_mlr}')
# R2_rf = np.corrcoef(results['Measured'],results['RF pred'])[0,1]
# print(f'R2_rf = {R2_rf}')
# plt.legend(['MLR','MLR mean','MLR confint','RandForest','RF mean','RF conf int'])



