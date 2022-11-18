#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:06:05 2022

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from basicUtils import *
from os import path

from numpy import random
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale 
from sklearn.cross_decomposition import PLSCanonical

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

#%% Load features from training + test set

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
df1 = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)
df1_ = df1[df1['Phase'] != '?']
df1_['Region'] = 1

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'
df2 = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)
df2_ = df2[df2['Phase'] != '?']
df1_['Region'] = 2

df_ = pd.concat((df1_,df2_),ignore_index=True)
# N,P = df_.shape

#%% Sanitize field names for smf

features_list = { # Cell geometry
                'Age':'age'
                # ,'Z_x':'z','Y_x':'y','X_x':'x'
                ,'Volume (sm)':'vol_sm'
                ,'Axial component':'axial_moment'
                ,'Nuclear volume':'nuc_vol'
                # ,'Nuclear surface area':'nuc_sa'
                ,'Nuclear axial component':'nuc_axial_moment'
                # ,'Nuclear solidity':'nuc_solid'
                ,'Nuclear axial angle':'nuc_angle'
                ,'Planar eccentricity':'planar_ecc'
                ,'Axial eccentricity':'axial_ecc'
                ,'Axial angle':'axial_angle'
                # ,'Planar component 1':'planar_component_1'
                # ,'Planar component 2':'planar_component_2'
                ,'Relative nuclear height':'rel_nuc_height'
                ,'Surface area':'sa'
                # ,'SA to vol':'ratio_sa_vol'
                # ,'Time to G1S':'time_g1s'
                ,'Basal area':'basal'
                ,'Apical area':'apical'
                
                # Growth rates
                ,'Specific GR b (sm)':'sgr'
                ,'Height to BM':'height_to_bm'
                ,'Mean curvature':'mean_curve'
                # ,'Gaussian curvature':'gaussian_curve'
                
                # Neighbor topolgy and
                # ,'Coronal angle':'cor_angle'
                # ,'Coronal density':'cor_density'
                ,'Cell alignment':'cell_align'
                ,'Mean neighbor nuclear volume':'mean_neighb_nuc_vol'
                ,'Mean neighbor dist':'mean_neighb_dist'
                ,'Neighbor mean height frame-1':'neighb_height_12h'
                ,'Neighbor mean height frame-2':'neighb_height_24h'
                # ,'Num diff neighbors':'neighb_diff'
                # ,'Num planar neighbors':'neighb_plan'
                ,'Collagen fibrousness':'col_fib'
                ,'Collagen alignment':'col_align'}

df_g1s = df_.loc[:,list(features_list.keys())]
df_g1s = df_g1s.rename(columns=features_list)
df_g1s['G1S_logistic'] = (df_['Phase'] == 'SG2').astype(int)

df_g1s_test = df_test_.loc[:,list(features_list.keys())]
df_g1s_test = df_g1s_test.rename(columns=features_list)
df_g1s_test['G1S_logistic'] = (df_test_['Phase'] == 'SG2').astype(int)

# Standardize
for col in df_g1s.columns:
    df_g1s[col] = z_standardize(df_g1s[col])

for col in df_g1s_test.columns:
    df_g1s_test[col] = z_standardize(df_g1s_test[col])

df_g1s = df_g1s.dropna()
# Count NANs
print(df_g1s.isnull().sum(axis=0))
print(len(df_g1s))

#%% Random forest regression

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

Niter = 100
sum_res = np.zeros(Niter)
Rsq = np.zeros(Niter)
importance = np.zeros((Niter,22))

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


sb.barplot(data=imp.melt(value_vars=imp.columns),x='variable',y='value')


