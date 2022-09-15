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
from scipy.special import expit
from basicUtils import *
from os import path

from numpy import random
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale 
from sklearn.cross_decomposition import PLSCanonical

def plot_logit_model(model,field):
    mdpoint = - model.params['Intercept'] / model.params[field]

    print(f'Mid point is: {mdpoint}')
    x = df_g1s[field].values
    y = df_g1s['G1S_logistic'].values
    plt.figure()
    plt.scatter( x, jitter(y,0.1) )
    sb.regplot(data = df_g1s,x=field,y='G1S_logistic',logistic=True,scatter=False)
    plt.ylabel('G1/S transition')
    plt.vlines([mdpoint],0,1)
    expitvals = expit( (x * model.params[field]) + model.params['Intercept'])
    I = np.argsort(expitvals)
    plt.plot(x[I],expitvals[I],'b')

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

#%%

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
df = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)

df_ = df[df['Phase'] != '?']

#%% Sanitize field names for smf

features_list = { # Cell geometry
                'Age':'age'
                ,'Volume (sm)':'vol_sm'
                # ,'Nuclear volume':'nuc_vol'
                ,'Planar eccentricity':'planar_ecc'
                ,'Axial eccentricity':'axial_ecc'
                ,'Axial component':'axial_moment'
                ,'Axial angle':'axial_angle'
                ,'Planar component 1':'planar_component_1'
                ,'Planar component 2':'planar_component_2'
                ,'Relative nuclear height':'rel_nuc_height'
                # ,'Surface area':'sa'
                
                # Growth rates
                # ,'Specific GR b (sm)':'sgr'
                ,'Growth rate b (sm)':'gr'
                ,'Height to BM':'height_to_bm'
                ,'Mean curvature':'mean_curve'
                
                # Neighbor topolgy and
                # ,'Coronal density':'cor_density'
                ,'Cell alignment':'cell_align'
                ,'Mean neighbor dist':'mean_neighb_dist'
                ,'Neighbor mean height frame-1':'neighb_height_12h'
                # ,'Neighbor mean height frame-2':'neighb_height_24h'
                ,'Num diff neighbors':'neighb_diff'
                ,'Num planar neighbors':'neighb_plan'}

df_g1s = df_.loc[:,list(features_list.keys())]
df_g1s = df_g1s.rename(columns=features_list)

# Standardize
for col in df_g1s.columns[df_g1s.columns != 'G1S_logistic']:
    df_g1s[col] = z_standardize(df_g1s[col])

df_g1s['G1S_logistic'] = (df_['Phase'] == 'SG2').astype(int)

#%%

from firthlogist import FirthLogisticRegression, load_sex2
import statsmodels.api as sm

df_g1s_ = df_g1s.dropna()
y = df_g1s_['G1S_logistic']
df_X = df_g1s_[df_g1s_.columns[df_g1s.columns != 'G1S_logistic']]
X = sm.add_constant(df_X)
feature_names = X.columns

fl = FirthLogisticRegression( pl_max_iter=1000)
fl.fit(X, y)

fl.summary(xname = feature_names)

#%%OLS for smoothed specific growth rate

############### OLS for specific growth rate ###############
model_ols = smf.rlm(f'sgr ~ ' + str.join(' + ',
                                      df_g1s.columns[(df_g1s.columns != 'sgr') &
                                                     (df_g1s.columns != 'gr')]),data=df_g1s).fit()
print(model_ols.summary())


############### GLM for specific growth rate ###############
model_glm = smf.glm(f'sgr ~ ' + str.join(' + ',
                                      df_g1s.columns[(df_g1s.columns != 'sgr') &
                                                     (df_g1s.columns != 'gr')]),data=df_g1s).fit()
print(model_glm.summary())



#%% Logistic for G1/S transition

############### G1S logistic as function of age ###############
model = smf.logit('G1S_logistic ~ ' + str.join(' + ',df_g1s.columns[df_g1s.columns != 'G1S_logistic']),
                  data=df_g1s).fit()
print(model.summary())

#%% Leave one out feature selection





