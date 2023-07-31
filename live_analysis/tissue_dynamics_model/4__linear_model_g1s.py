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

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_g1s.csv',index_col=0)
df_g1s = df_g1s.drop(columns=['age','G1S_logistic'])

#Trim out G2 cells
df_g1s = df_g1s[df_g1s['time_g1s'] >= 0]


X = df_g1s.drop(columns=['time_g1s'])
y = df_g1s['time_g1s']

#Add interaction effects ?
X['vol*sgr'] = z_standardize(X['sgr'] * X['vol_sm'])

#%% Establish the theoretical maximum R2 based on time resolution alone

from numpy import random, corrcoef

Nsample = 10000

x = random.lognormal(mean=np.log(48),sigma=sqrt(0.03),size=(Nsample))

bins = np.arange(0,10*24,12)

x_hat = bins[np.digitize(x,bins)]

plt.scatter(x,x_hat)
max_exp_Rsq = corrcoef(x,x_hat)[0,1]**2
plt.title(f'Maximum expected Rsq = {max_exp_Rsq}')

#%%

from sklearn import linear_model
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

reg = linear_model.RidgeCV()
reg = reg.fit(X_train,y_train)


results = pd.Series(reg.coef_, reg.feature_names_in_)
results.sort_values().plot.bar()
plt.tight_layout()
plt.ylabel('Effect size')
plt.title('Linear regression for G1S timing')

#%% Cross-validation using MLR

Niter = 100

frac_withhold = 0.2
N = len(df_g1s)

models = []
random_models = []
Rsq = np.zeros(Niter)

for i in tqdm(range(Niter)):
    
    num_withold = np.round(frac_withhold * N).astype(int)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=frac_withhold)
    
    reg = linear_model.RidgeCV()
    this_model = reg.fit(X_train,y_train)
    models.append(this_model)
    
    # predict on the withheld data
    ypred = this_model.predict(X_test)
    Rsq[i] = metrics.r2_score(y_test,ypred)
    
plt.figure()
plt.hist(Rsq)
plt.xlabel('R^2')
plt.title('Linear regression G1/S timing')

mlr = this_model
result = permutation_importance(mlr, X_test, y_test, n_repeats=100, random_state=42, n_jobs=2)
mlr_importances = pd.Series(result.importances_mean, index=X.columns)

plt.figure()
mlr_importances.plot.bar(yerr=result.importances_std)
plt.title('Linear regression G1/S timing')
plt.ylabel('Permutation importance')

#%% Random forest regression

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

Niter = 100
sum_res = np.zeros(Niter)
Rsq = np.zeros(Niter)
importance = np.zeros((Niter,len(X.columns)))

for i in tqdm(range(Niter)):
    
    forest = RandomForestRegressor(n_estimators=100, random_state=i)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
    
    forest.fit(X_train,y_train)
    
    y_pred = forest.predict(X_test)
    residuals = y_pred - y_test
    sum_res[i] = residuals.sum()
    Rsq[i] = np.corrcoef(y_pred,y_test)[0,1]
    importance[i,:] = forest.feature_importances_
    
plt.hist(Rsq,15); plt.xlabel('R-squared'); plt.ylabel('Counts'); plt.title('RF regression model of G1/S timing')
plt.vlines(max_exp_Rsq,0,15,color='r',linestyle='dashed')
plt.vlines(Rsq.mean(),0,15,color='k',linestyle='dashed')
plt.xlim([0,1])
plt.legend(['Expected maximum R2 based on 12h resolution',f'Mean RF R2: {Rsq.mean()}','RF R2'])

imp = pd.DataFrame(importance)
imp.columns = X.columns

plt.figure()
sb.barplot(data=imp.melt(value_vars=imp.columns),x='variable',y='value');
plt.xticks(rotation=90);plt.ylabel('Impurity importance')
plt.title('RF regression model of G1/S timing')
plt.tight_layout()

#%%

# Plot permutation importance
result = permutation_importance(
    forest, X_test, y_test, n_repeats=100, random_state=42, n_jobs=2
)

forest_importances = pd.Series(result.importances_mean, index=df_g1s.drop(columns=['time_g1s']).columns)

plt.figure()
top_forest_imp = forest_importances.iloc[forest_importances.argsort()][-5:]
top_forest_imp_std = result.importances_std[forest_importances.argsort()][-5:]
top_forest_imp.plot.bar(yerr=top_forest_imp_std)
plt.ylabel("Mean accuracy decrease")
plt.tight_layout()
plt.show()

#%% Use Region1 -> Pred Region2
