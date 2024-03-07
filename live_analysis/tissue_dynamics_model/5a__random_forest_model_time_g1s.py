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

df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_g1s.csv',index_col=0)
df_g1s = df_g1s.drop(columns=['age','cellID','region','G1S_logistic','fucci_int_12h'])

# De-standardize and note down stats
corr_std = 34.54557205301856
corr_mean = -75.85760517799353
df_g1s['time_g1s'] = df_g1s['time_g1s'] * corr_std
df_g1s['time_g1s'] = df_g1s['time_g1s'] + corr_mean

#Trim out G2 cells
df_g1s = df_g1s[df_g1s['time_g1s'] <= 0]

# Re-standardize
std = df_g1s['time_g1s'].std()
mean = df_g1s['time_g1s'].mean()
df_g1s['time_g1s'] = (df_g1s['time_g1s'] - mean)/std

X = df_g1s.drop(columns=['time_g1s'])
X['Intercept'] = 1
y = df_g1s['time_g1s']
X['vol*sgr'] = z_standardize(X['sgr'] * X['vol_sm'])

#%% Random forest regression

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

Niter = 100
Rsq_rf = pd.DataFrame()
MSE_rf = pd.DataFrame()

ytests = []; ypreds = []

for i in tqdm(range(Niter)):
    
    forest = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,rsq_in,prediction = run_cross_validation(X,y,0.1,forest,random_state=i,plot=True)
    Rsq_rf.at[i,'Out sample'] = rsq
    Rsq_rf.at[i,'In sample'] = rsq_in
    MSE_rf.at[i,'Out sample'] = mse
    forest_random = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,_,_ = run_cross_validation(X,y,0.1,forest_random,random_state=i,run_permute=True)
    Rsq_rf.at[i,'Random'] = rsq
    MSE_rf.at[i,'Random'] = mse
    
    ytests.append(prediction[0]); ypreds.append(prediction[1])
        
sb.histplot(Rsq_rf[['Out sample','Random']].melt(),bins=25,x='value',hue='variable',element='poly',fill=False,stat='probability',)
plt.vlines(Rsq_rf[['Out sample','Random']].mean(),0,0.15)

#%%

predictions = pd.DataFrame()
predictions['y test'] = np.hstack(ytests)*corr_std
predictions['y pred'] = np.hstack(ypreds)*corr_std
predictions = predictions + corr_mean

plt.figure()
sb.histplot(-predictions,x='y test',y='y pred',bins=len(predictions['y test'].unique()),cbar=True,stat='probability')
plt.tight_layout()
plt.xlabel('Observed wait time until G1/S (h)')
plt.ylabel('Predicted wait time until G1/S (h)')
plt.gca().set_aspect('equal','box')

plt.figure()
plt.scatter(-predictions['y test'],-predictions['y pred'],alpha=0.01)
plt.xlabel('Observed wait time until G1/S (h)')
plt.ylabel('Predicted wait time until G1/S (h)')

#%%

#%% Random forest regression on PCA

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

Niter = 10
Ncomp = 20

Rsq_pca = pd.DataFrame()
MSE_pca = pd.DataFrame()

for i in tqdm(range(Niter)):
    
    pca,_,_ = run_pca(X,Ncomp)
    
    forest = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,rsq_in,_ = run_cross_validation(pca,y,0.1,forest,random_state=i,plot=True)
    Rsq_pca.at[i,'Out'] = rsq
    Rsq_pca.at[i,'In'] = rsq_in
    MSE_pca.at[i,'Full'] = mse
    forest_random = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,_,_ = run_cross_validation(pca,y,0.1,forest_random,random_state=i,run_permute=True)
    Rsq_pca.at[i,'Random'] = rsq
    MSE_pca.at[i,'Random'] = mse

plt.figure()
sb.histplot(Rsq_pca.melt(),x='value',hue='variable',common_norm=False,bins=20)

#%% Permutation importances

from mathUtils import total_std

Niter = 10
mean_importances = []
std_importances = []

for i in tqdm(range(Niter)):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
    
    # Plot permutation importance
    forest = RandomForestRegressor(n_estimators=100, random_state=i).fit(X_train,y_train)
    result = permutation_importance(forest,X_test,y_test,n_repeats=10,random_state=42,n_jobs=2)
    mean = pd.DataFrame(result.importances_mean, index=X.columns).T
    std = pd.DataFrame(result.importances_std, index=X.columns).T
    mean_importances.append(mean)
    std_importances.append(std)

mean_importances = pd.concat(mean_importances)
std_importances = pd.concat(std_importances)

perm_importances = pd.DataFrame()
perm_importances['Mean'] = mean_importances.mean()
# perm_importances['Std'] = [ total_std(std_importances[f],mean_importances[f],np.ones(Niter)*100) for f in X.columns ] # not sure if this is right...
perm_importances['Std'] = mean_importances.std()

perm_importances.sort_values('Mean',ascending=False).plot(kind='bar',yerr='Std')
plt.tight_layout()

#%% Drop single features -> Rsq

Niter = 100
other_features2try = perm_importances.sort_values('Mean',ascending=False).index[1:10]

Rsq = pd.DataFrame()
MSE = pd.DataFrame()

for i in tqdm(range(Niter)):
    
    forest = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,_,_ = run_cross_validation(X,y,0.1,forest,random_state=i)
    Rsq.at[i,'Full'] = rsq
    MSE.at[i,'Full'] = mse
    
    # No volume
    forest_no_vol = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,_,_ = run_cross_validation(X.drop(columns='vol_sm'),y,0.1,forest_no_vol,random_state=i)
    Rsq.at[i,'No volume'] = rsq
    MSE.at[i,'No volume'] = mse
    
    forest_random = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,_,_ = run_cross_validation(X,y,0.1,forest_random,random_state=i,run_permute=True)
    Rsq.at[i,'Random'] = rsq
    MSE.at[i,'Random'] = mse
    
    for j,f in enumerate(other_features2try):
        # X_train,X_test,y_train,y_test = train_test_split(X.drop(columns=f),y,test_size=0.2)
        forest_no_other = RandomForestRegressor(n_estimators=100, random_state=i)
        rsq,mse,_,_ = run_cross_validation(X.drop(columns=f),y,0.1,forest_no_other,random_state=i)
        Rsq.at[i,f'No {f}'] = rsq
        MSE.at[i,f'No {f}'] = mse

print('---')


Rsq_ = pd.melt(Rsq)
Rsq_['Category'] = Rsq_['variable']
Rsq_.loc[(Rsq_['variable'] != 'Full') & (Rsq_['variable'] != 'Random') & (Rsq_['variable'] != 'No volume'),'Category'] = 'No other'
Rsq_.groupby('Category').mean()

sb.histplot(Rsq_,x='value',hue='Category',stat='probability',element='poly',common_norm=False,fill=False)
plt.vlines(Rsq_.groupby('Category').mean(),0,0.25)

plt.figure()
sb.barplot(Rsq_,y='value',x='variable')
plt.xticks(rotation=90);plt.tight_layout();

#%% Singleton features

Niter = 100
other_features2try = perm_importances.sort_values('Mean',ascending=False).index[1:10]

Rsq = pd.DataFrame()
MSE = pd.DataFrame()

for i in tqdm(range(Niter)):
    
    forest = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,_,_ = run_cross_validation(X,y,0.1,forest,random_state=i)
    Rsq.at[i,'Full'] = rsq
    MSE.at[i,'Full'] = mse
    
    # Only volume
    forest_no_vol = RandomForestRegressor(n_estimators=100, random_state=i)
    rsq,mse,_,_ = run_cross_validation(X[['vol_sm','Intercept']],y,0.1,forest_no_vol,random_state=i)
    Rsq.at[i,'Only volume'] = rsq
    MSE.at[i,'Only volume'] = mse
    
    
    forest_random = RandomForestRegressor(n_estimators=100, random_state=i)
    X_ = X; X_['Dummy'] = random.randn(len(X))
    rsq,mse,_,_ = run_cross_validation(X[['Dummy','Intercept']],y,0.1,forest_random,random_state=i,run_permute=True)
    Rsq.at[i,'Random'] = rsq
    MSE.at[i,'Random'] = mse
    
    for j,f in enumerate(other_features2try):
        forest_no_other = RandomForestRegressor(n_estimators=100, random_state=i)
        rsq,mse,_,_ = run_cross_validation(X[[f,'Intercept']],y,0.1,forest_no_other,random_state=i)
        Rsq.at[i,f'Only {f}'] = rsq
        MSE.at[i,f'Only {f}'] = mse
    
Rsq_ = pd.melt(Rsq)
Rsq_['Category'] = Rsq_['variable']
Rsq_.loc[(Rsq_['variable'] != 'Full') & (Rsq_['variable'] != 'Random') & (Rsq_['variable'] != 'Only volume'),'Category'] = 'Only other'
Rsq_.groupby('Category').mean()

# sb.histplot(Rsq_,x='value',hue='Category',bins=30,stat='probability',element='poly',common_norm=False,fill=False)
# plt.vlines(Rsq_.groupby('Category').mean(),0,0.25)

plt.figure()
sb.barplot(Rsq_,y='value',x='variable')
plt.xticks(rotation=90);plt.tight_layout();

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
