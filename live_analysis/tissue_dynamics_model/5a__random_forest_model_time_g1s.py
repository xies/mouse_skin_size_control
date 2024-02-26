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
    
    return Rsq,MSE,Rsq_insample


df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_g1s.csv',index_col=0)
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
df_g1s = df_g1s.drop(columns=['fucci_int_12h'])

#%% Random forest regression

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

Niter = 100
sum_res = np.zeros(Niter)
Rsq_rf = np.zeros(Niter)
Rsq_rf_insample = np.zeros(Niter)
MSE_rf = np.zeros(Niter)
Rsq_random = np.zeros(Niter)
Rsq_random_insample = np.zeros(Niter)
MSE_random = np.zeros(Niter)

for i in tqdm(range(Niter)):
    
    forest = RandomForestRegressor(n_estimators=100, random_state=i)
    Rsq_rf[i],MSE_rf[i],Rsq_rf_insample[i] = run_cross_validation(X,y,0.1,forest,random_state=i,plot=True)
    forest_random = RandomForestRegressor(n_estimators=100, random_state=i)
    Rsq_random[i],MSE_random[i],_ = run_cross_validation(X,y,0.1,forest_random,random_state=i,run_permute=True)

print('---')
print(f'Insample Rsq for RF = {Rsq_rf_insample.mean()}')
print(f'Mean Rsq for RF = {Rsq_rf.mean()}')
print(f'Insample Rsq for random = {Rsq_random_insample.mean()}')
print(f'Mean Rsq for random = {Rsq_random.mean()}')

#%% Random forest regression on PCA

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

Niter = 100
Ncomp = 20

sum_res = np.zeros(Niter)
Rsq_rf = np.zeros(Niter)
Rsq_rf_insample = np.zeros(Niter)
MSE_rf = np.zeros(Niter)
Rsq_random = np.zeros(Niter)
Rsq_random_insample = np.zeros(Niter)
MSE_random = np.zeros(Niter)

for i in tqdm(range(Niter)):
    
    pca,_,_ = run_pca(X,Ncomp)
    
    forest = RandomForestRegressor(n_estimators=100, random_state=i)
    Rsq_rf[i],MSE_rf[i],Rsq_rf_insample[i] = run_cross_validation(pca,y,0.1,forest,random_state=i,plot=True)
    forest_random = RandomForestRegressor(n_estimators=100, random_state=i)
    Rsq_random[i],MSE_random[i],_ = run_cross_validation(pca,y,0.1,forest_random,random_state=i,run_permute=True)

print('---')
print(f'Insample Rsq for RF = {Rsq_rf_insample.mean()}')
print(f'Mean Rsq for RF = {Rsq_rf.mean()}')
print(f'Insample Rsq for random = {Rsq_random_insample.mean()}')
print(f'Mean Rsq for random = {Rsq_random.mean()}')

#%% Permutation importances

from sklearn.inspection import permutation_importance

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

# Plot permutation importance
forest = RandomForestRegressor(n_estimators=100, random_state=i).fit(X_train,y_train)
result = permutation_importance(forest,X_test,y_test,n_repeats=10,random_state=42,n_jobs=2)
forest_importances = pd.Series(result.importances_mean, index=X.columns)

plt.figure()
top_forest_imp = forest_importances.iloc[forest_importances.argsort()][-10:][::-1]
top_forest_imp_std = result.importances_std[forest_importances.argsort()][-10:][::-1]
top_forest_imp.plot.bar(yerr=top_forest_imp_std)
plt.ylabel("Mean accuracy decrease")
plt.tight_layout()
plt.show()

#%% Drop single features -> Rsq

Niter = 100
other_top_features_to_try = 5

Rsq = pd.DataFrame()
MSE = pd.DataFrame()

for i in tqdm(range(Niter)):
    
    forest = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,_ = run_cross_validation(X,y,0.1,forest,random_state=i)
    Rsq.at[i,'Full'] = rsq
    MSE.at[i,'Full'] = mse
    
    # No volume
    forest_no_vol = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,_ = run_cross_validation(X.drop(columns='vol_sm'),y,0.1,forest_no_vol,random_state=i)
    Rsq.at[i,'No volume'] = rsq
    MSE.at[i,'No volume'] = mse
    
    forest_random = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,_ = run_cross_validation(X,y,0.1,forest_random,random_state=i,run_permute=True)
    Rsq.at[i,'Random'] = rsq
    MSE.at[i,'Random'] = mse
    
    for j,f in enumerate(forest_importances.sort_values()[::-1][1:1+other_top_features_to_try].index):
        # X_train,X_test,y_train,y_test = train_test_split(X.drop(columns=f),y,test_size=0.2)
        forest_no_other = RandomForestRegressor(n_estimators=100, random_state=i)
        rsq,mse,_ = run_cross_validation(X.drop(columns=f),y,0.1,forest_no_other,random_state=i)
        Rsq.at[i,f'No {f}'] = rsq
        MSE.at[i,f'No {f}'] = mse

print('---')

Rsq_ = pd.melt(Rsq)
Rsq_['Category'] = Rsq_['variable']
Rsq_.loc[(Rsq_['variable'] != 'Full') & (Rsq_['variable'] != 'Random') & (Rsq_['variable'] != 'No volume'),'Category'] = 'No other'
Rsq_.groupby('Category').mean()

sb.histplot(Rsq_,x='value',hue='Category',stat='probability',element='poly',common_norm=False,fill=False)
plt.vlines(Rsq_.groupby('Category').mean(),0,0.25)

#%% Singleton features

Niter = 100
other_top_features_to_try = 5

Rsq = pd.DataFrame()
MSE = pd.DataFrame()

for i in tqdm(range(Niter)):
    
    forest = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,_ = run_cross_validation(X,y,0.1,forest,random_state=i)
    Rsq.at[i,'Full'] = rsq
    MSE.at[i,'Full'] = mse
    
    # No volume
    forest_no_vol = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,_ = run_cross_validation(X[['vol_sm','Intercept']],y,0.1,forest_no_vol,random_state=i)
    Rsq.at[i,'Only volume'] = rsq
    MSE.at[i,'Only volume'] = mse
    
    forest_random = RandomForestRegressor(n_estimators=100, random_state=i)
    X_ = X; X_['Dummy'] = random.randn(len(X))
    rsq,mse,_ = run_cross_validation(X[['Dummy','Intercept']],y,0.1,forest_random,random_state=i,run_permute=True)
    Rsq.at[i,'Random'] = rsq
    MSE.at[i,'Random'] = mse
    
    for j,f in enumerate(forest_importances.sort_values()[::-1][1:1+other_top_features_to_try].index):
        forest_no_other = RandomForestRegressor(n_estimators=100, random_state=i)
        rsq,mse,_ = run_cross_validation(X[[f,'Intercept']],y,0.1,forest_no_other,random_state=i)
        Rsq.at[i,f'Only {f}'] = rsq
        MSE.at[i,f'Only {f}'] = mse

Rsq_ = pd.melt(Rsq)
Rsq_['Category'] = Rsq_['variable']
Rsq_.loc[(Rsq_['variable'] != 'Full') & (Rsq_['variable'] != 'Random') & (Rsq_['variable'] != 'Only volume'),'Category'] = 'Only other'
Rsq_.groupby('Category').mean()

sb.histplot(Rsq_,x='value',hue='Category',bins=30,stat='probability',element='poly',common_norm=False,fill=False)
plt.vlines(Rsq_.groupby('Category').mean(),0,0.25)

#%% Use Region1 -> Pred Region2
#@todo: un-standardize data for display

# df_g1s = df_g1s.drop(columns=['cellID'])
# dt = 0.111727

# X = df_g1s.drop(columns=['time_g1s'])
# X['Intercept'] = 1
# y = df_g1s['time_g1s']

# #Add interaction effects ?
# X['vol*sgr'] = z_standardize(X['sgr'] * X['vol_sm'])
# df_g1s = df_g1s.drop(columns=['fucci_int_24h','fucci_int_12h'])


X_train = X[X.region == 1]
y_train = y[X.region == 1]
X_test = X[X.region == 2]
y_test = y[X.region == 2]
forest = RandomForestRegressor(n_estimators=1000, random_state=i)

forest.fit(X_train,y_train)

y_pred = forest.predict(X_test)
plt.scatter(y_test,y_pred,alpha=0.1)

plt.xlabel('Measured: hours until G1/S')
plt.ylabel('Predicted: hours until G1/S')

# residuals = y_pred - y_test
# sum_res[i] = residuals.sum()
# Rsq[i] = np.corrcoef(y_pred,y_test)[0,1]**2
# importance[i,:] = forest.feature_importances_
# r2_score(y_pred,y_test)
