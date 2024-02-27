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
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

def run_pca(df,ncomp):
    
    pca = PCA(n_components = Ncomp)
    X_ = pca.fit_transform(df)
    df_pca = pd.DataFrame(X_,columns = [f'PC{str(i)}' for i in range(Ncomp)])
    components = pd.DataFrame(pca.components_,columns=df.columns)
    
    return df_pca, components, pca

def plot_principle_component(loadings,which_comp):
    loadings = loadings.iloc[which_comp]
    loadings = loadings[np.abs(loadings).argsort()]
    plt.figure()
    np.abs(loadings).plot.bar()
    plt.ylabel('Magnitude of loading')
    plt.xlabel('Original dimensions');plt.title(f'PC {which_comp}')
    plt.tight_layout()

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
    
    return Rsq,MSE,Rsq_insample,[y_test,y_pred]

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_g1s.csv',index_col=0)
df_g1s = df_g1s.drop(columns=['age','region','cellID'])

# De-standardize and note down stats
std = 34.54557205301856
mean = -75.85760517799353
df_g1s['time_g1s'] = df_g1s['time_g1s'] * std
df_g1s['time_g1s'] = df_g1s['time_g1s'] + mean

#Trim out G2 cells
df_g1s = df_g1s[df_g1s['G1S_logistic'] == 0]

# Re-standardize
# std = df_g1s['time_g1s'].std()
# mean = df_g1s['time_g1s'].mean()
# df_g1s['time_g1s'] = (df_g1s['time_g1s'] - mean)/std

#Add interaction effects ?
df_g1s = df_g1s.drop(columns=['fucci_int_12h'])

X = df_g1s.drop(columns=['time_g1s'])
X['Intercept'] = 1
y = df_g1s['time_g1s']
X['vol*sgr'] = z_standardize(X['sgr'] * X['vol_sm'])

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

Niter = 1000

frac_withheld = 0.1

Rsq_lr = pd.DataFrame()
MSE_lr = pd.DataFrame()

for i in tqdm(range(Niter)):
        
    model = linear_model.LinearRegression()
    rsq,mse,rsq_in,_ = run_cross_validation(X,y,frac_withheld,model)
    Rsq_lr.at[i,'Out'] = rsq
    Rsq_lr.at[i,'In'] = rsq_in
    model_random = linear_model.LinearRegression()
    rsq,mse,rse_in,_ = run_cross_validation(X,y,frac_withheld,model,run_permute=True)
    Rsq_lr.at[i,'Random'] = rsq
    
sb.histplot(Rsq_lr.melt(),x='value',hue='variable')

#%%

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
mlr = linear_model.LinearRegression().fit(X_train,y_train)

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

from sklearn.inspection import permutation_importance

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
