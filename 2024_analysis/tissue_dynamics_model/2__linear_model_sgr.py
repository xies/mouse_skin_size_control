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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import scale

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

def run_cross_validation(X,y,split_ratio,model,random_state=42,plot=False,run_permute=False):
    
    if run_permute:
        # X = X.sample(len(X),replace=False)
        X = random.randn(*X.shape)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=split_ratio)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    Rsq = r2_score(y_test,y_pred)
    Rsq_insample = r2_score(y_train,model.predict(X_train))
    MSE = mean_squared_error(y_test,y_pred)
    if plot:
        plt.scatter(y_test,y_pred,color='b',alpha=0.05)
    
    return Rsq,MSE,Rsq_insample

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_g1s.csv',index_col=0)

df_g1s = df_g1s.drop(columns=['cellID','time_g1s','G1S_logistic'])

X = df_g1s.drop(columns='sgr')
X['Intercept'] = 1
y = df_g1s['sgr']

#%% Multilinear regression

Niter = 100
Rsq_mlr = np.zeros(Niter)
MSE_mlr = np.zeros(Niter)
Rsq_mlr_insample = np.zeros(Niter)
Rsq_random = np.zeros(Niter)
MSE_random = np.zeros(Niter)
Rsq_random_insample = np.zeros(Niter)

for i in range(Niter):
    
    model = Ridge()
    Rsq_mlr[i],MSE_mlr[i],Rsq_mlr_insample[i] = run_cross_validation(X,y,0.1,model,random_state=i,plot=True)
    
    model_random = Ridge()
    Rsq_random[i],MSE_random[i],Rsq_random_insample[i] = run_cross_validation(X,y,0.1,model_random,random_state=i,run_permute=True)
    
print(f'Insample Rsq for MLR = {Rsq_mlr_insample.mean()}')
print(f'Mean Rsq for MLR = {Rsq_mlr.mean()}')
print(f'Mean Rsq for random model = {Rsq_random.mean()}')
print(f'Insample Rsq for random model = {Rsq_random_insample.mean()}')

#RFE
from sklearn.feature_selection import RFE

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=i)
selector = RFE(estimator=LinearRegression(), n_features_to_select=1)
results = selector.fit(X_train,y_train)

#%% Random forest regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import plot_tree

Niter = 100
Rsq_rf = np.zeros(Niter)
MSE_rf = np.zeros(Niter)
Rsq_rf_insample = np.zeros(Niter)
Rsq_random = np.zeros(Niter)
MSE_random = np.zeros(Niter)
Rsq_random_insample = np.zeros(Niter)
importance = np.zeros((Niter,df_g1s.shape[1]-1))

for i in tqdm(range(Niter)):
    
    forest = RandomForestRegressor(n_estimators=100, random_state=i)
    Rsq_rf[i],MSE_rf[i],Rsq_rf_insample[i] = run_cross_validation(X,y,0.1,forest,random_state=i,plot=True)
    forest_random = RandomForestRegressor(n_estimators=100, random_state=i)
    Rsq_random[i],MSE_random[i],_ = run_cross_validation(X,y,0.1,forest_random,random_state=i,run_permute=True)

plt.figure()
plt.hist(Rsq_rf)
print('---')
print(f'Insample Rsq for RF = {Rsq_rf_insample.mean()}')
print(f'Mean Rsq for RF = {Rsq_rf.mean()}')
print(f'Insample Rsq for random = {Rsq_random_insample.mean()}')
print(f'Mean Rsq for random = {Rsq_random.mean()}')

#%% RF: Permutation importance

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=frac_withheld,random_state=42)

forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(X_train,y_train)
result = permutation_importance(forest,X_test,y_test,n_repeats=100,random_state=42,n_jobs=2)
forest_importances = pd.Series(result.importances_mean, index=X_train.columns)

top_forest_imp = forest_importances.iloc[forest_importances.argsort()][-10:][::-1]
top_forest_imp_std = result.importances_std[forest_importances.argsort()][-10:][::-1]
top_forest_imp.plot.bar(yerr=top_forest_imp_std)
plt.ylabel("Mean accuracy decrease")
plt.xlabel('Permuted feature')
plt.tight_layout()
plt.show()

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



