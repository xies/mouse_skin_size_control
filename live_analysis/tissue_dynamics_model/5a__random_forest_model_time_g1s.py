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
from basicUtils import *

from numpy import random
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_g1s.csv',index_col=0)
df_g1s = df_g1s.drop(columns=['age','G1S_logistic'])

#Trim out G2 cells
df_g1s = df_g1s[df_g1s['time_g1s'] >= 0]

df_g1s = df_g1s.drop(columns=['cellID'])
dt = 0.111727

X = df_g1s.drop(columns=['time_g1s'])
X['Intercept'] = 1
y = df_g1s['time_g1s']

#Add interaction effects ?
X['vol*sgr'] = z_standardize(X['sgr'] * X['vol_sm'])
df_g1s = df_g1s.drop(columns=['fucci_int_24h','fucci_int_12h'])

#%% Establish the theoretical maximum R2 based on time resolution alone

Nsample = 100

true_g1s_times = random.lognormal(mean=np.log(1),sigma=np.sqrt(0.03),size=(Nsample))
# observed_g1s_times = (true_g1s_times // dt)*dt

# x = []
# x_hat = []
# for i in range(Nsample):
#     true_cumulative_wait_times = np.arange(0,true_g1s_times[i],dt) - true_g1s_times[i]
#     observed_cumulative_wait_times = np.arange(0,observed_g1s_times[i],dt) - observed_g1s_times[i]
    
#     if len(observed_cumulative_wait_times) < len(true_cumulative_wait_times):
#         observed_cumulative_wait_times = np.hstack((observed_cumulative_wait_times,0))
    
#     x.extend(true_cumulative_wait_times)
#     x_hat.extend(observed_cumulative_wait_times)
#     assert(len(x) == len(x_hat))
# x = -np.array(x).flatten()
# x_hat = -np.array(x_hat).flatten()

errors = random.uniform(high = dt,size=len(y))

# plt.hist(y,8,density=True,histtype='step');plt.hist(x,density=True,histtype='step')
MSE_theo = mean_squared_error(np.zeros(len(y)),errors)

R2_theo = r2_score(y,y+errors)

print(f'Minimum R2 = {R2_theo}')
print(f'Minimum MSE = {MSE_theo}')

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
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
    forest.fit(X_train,y_train)
    
    y_pred = forest.predict(X_test)
    residuals = y_pred - y_test
    sum_res[i] = residuals.sum()
    Rsq[i] = np.corrcoef(y_pred,y_test)[0,1]**2
    importance[i,:] = forest.feature_importances_
    # r2_score(y_pred,y_test)
    
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

from sklearn.inspection import permutation_importance

# Plot permutation importance
result = permutation_importance(
    forest, X_test, y_test, n_repeats=100, random_state=42, n_jobs=2
)

forest_importances = pd.Series(result.importances_mean, index=X.columns)

plt.figure()
top_forest_imp = forest_importances.iloc[forest_importances.argsort()][-5:]
top_forest_imp_std = result.importances_std[forest_importances.argsort()][-5:]
top_forest_imp.plot.bar(yerr=top_forest_imp_std)
plt.ylabel("Mean accuracy decrease")
plt.tight_layout()
plt.show()

#%% Feature drop -> Rsq

Niter = 100
rsq_max = np.zeros(Niter)
rsq_full = np.zeros(Niter)
rsq_random = np.zeros(Niter)
rsq_no_vol_sm = np.zeros(Niter)
rsq_no_other = np.zeros((5,Niter))

mse_max = np.zeros(Niter)
mse_full = np.zeros(Niter)
mse_random = np.zeros(Niter)
mse_no_vol_sm = np.zeros(Niter)
mse_no_other = np.zeros((5,Niter))

for i in tqdm(range(Niter)):
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
    model_full = RandomForestRegressor(n_estimators=100, random_state=i)
    model_full.fit(X_train,y_train)
    y_pred_full = model_full.predict(X_test)
    rsq_full[i] = r2_score(y_test,y_pred_full)
    mse_full[i] = mean_squared_error(y_test,y_pred_full)
    
    X_train,X_test,y_train,y_test = train_test_split(X.drop(columns='vol_sm'),y,test_size=0.2)
    model_no_vol_sm = RandomForestRegressor(n_estimators=100, random_state=i)
    model_no_vol_sm.fit(X_train,y_train)
    y_pred_no_vol_sm = model_no_vol_sm.predict(X_test)
    rsq_no_vol_sm[i] = r2_score(y_test,y_pred_no_vol_sm)
    mse_no_vol_sm[i] = mean_squared_error(y_test,y_pred_no_vol_sm)
    
    X_train,X_test,y_train,y_test = train_test_split(random.randn(*X.shape),y,test_size=0.2)
    model_random = RandomForestRegressor(n_estimators=100, random_state=i)
    model_random.fit(X_train,y_train)
    y_pred_random = model_random.predict(X_test)
    rsq_random[i] = r2_score(y_test,y_pred_random)
    mse_random[i] = mean_squared_error(y_test,y_pred_random)
    
    for j,f in enumerate(forest_importances.sort_values()[::-1][1:6].index):
        X_train,X_test,y_train,y_test = train_test_split(X.drop(columns=f),y,test_size=0.2)
        model_other = RandomForestRegressor(n_estimators=100, random_state=i)
        model_other.fit(X_train,y_train)
        y_pred_no_other = model_other.predict(X_test)
        rsq_no_other[j,i] = r2_score(y_test,y_pred_no_other)
        mse_no_other[j,i] = mean_squared_error(y_test,y_pred_no_other)

plt.figure()
plt.hist(rsq_full,density=True,histtype='step')
plt.hist(rsq_random,density=True,histtype='step')
plt.hist(rsq_no_vol_sm,density=True,histtype='step')
plt.hist(rsq_no_other.flatten(),density=True,histtype='step')
plt.legend(['Full','Random','No cell volume','No other feature'])

plt.xlabel('Rsq')

plt.figure()
# plt.vline()
plt.hist(mse_full,density=True,histtype='step')
plt.hist(mse_random,density=True,histtype='step')
plt.hist(mse_no_vol_sm,density=True,histtype='step')
plt.hist(mse_no_other.flatten(),density=True,histtype='step')
plt.legend(['Full','Random','No cell volume','No other feature'])

plt.xlabel('MSE')

#%% Use Region1 -> Pred Region2
