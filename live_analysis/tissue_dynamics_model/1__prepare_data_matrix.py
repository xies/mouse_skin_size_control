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
from sklearn.covariance import EmpiricalCovariance
from basicUtils import *
from os import path

from numpy import random
from sklearn.preprocessing import scale 
from numpy.linalg import eig

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

#%% Load features from training + test set

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
df1 = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)
df1['Region'] = 1
df1_ = df1[df1['Phase'] != '?']

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'
df2 = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)
df2['Region'] = 2
df2_ = df2[df2['Phase'] != '?']

df_ = pd.concat((df1_,df2_),ignore_index=True)
df_['UniqueID'] = df_['basalID'].astype(str) + '_' + df_['Region'].astype(str)
N,P = df_.shape

# Sanitize field names for smf

features_list = { # Cell geometry
                'Age':'age'
                ,'Region':'region'
                # ,'Differentiating':'diff'
                ,'Y_x':'y','X_x':'x'
                # ,'Z_x':'z'
                ,'Volume':'vol_sm'
                # ,'Axial component':'axial_moment'
                # ,'Nuclear volume':'nuc_vol'
                ,'NC ratio':'nc_ratio'
                # ,'Nuclear surface area':'nuc_sa'
                # ,'Nuclear axial component':'nuc_axial_moment'
                ,'Nuclear solidity':'nuc_solid'
                ,'Nuclear axial angle':'nuc_angle'
                # ,'Planar eccentricity':'planar_ecc'
                ,'Axial eccentricity':'axial_ecc'
                # ,'Nuclear axial eccentricity':'nuc_axial_ecc'
                # ,'Nuclear planar eccentricity':'nuc_planar_ecc'
                ,'Axial angle':'axial_angle'
                # ,'Planar component 1':'planar_component_1'
                # ,'Planar component 2':'planar_component_2'
                ,'Relative nuclear height':'rel_nuc_height'
                # ,'Surface area':'sa'
                ,'Time to G1S':'time_g1s'
                ,'Basal area':'basal'
                ,'Apical area':'apical'
                
                # Growth rates and other central cell dynamics
                ,'Specific GR spl':'sgr'
                ,'Height to BM':'height_to_bm'
                
                ,'FUCCI bg sub frame-1':'fucci_int_12h'
                ,'FUCCI bg sub frame-2':'fucci_int_24h'
                
                # Neighbor topolgy and geometry
                ,'Coronal angle':'cor_angle'
                # ,'Coronal density':'cor_density'
                ,'Cell alignment':'cell_align'
                ,'Mean curvature':'mean_curve'
                ,'Num diff neighbors':'neighb_diff'
                ,'Num planar neighbors':'neighb_plan'
                ,'Collagen fibrousness':'col_fib'
                ,'Collagen alignment':'col_align'
                # ,'Gaussian curvature':'gaussian_curve'
                
                # Cell dynamics
                ,'Delta curvature':'delta_curvature'
                ,'Delta height':'delta_height'
                
                # Neighborhood dynamics (averages of previous frame)
                ,'Mean neighbor nuclear volume normalized':'mean_neighb_nuc_vol'
                ,'Neighbor mean nuclear volume frame-1':'mean_neighb_nuc_vol_12h'
                # ,'Neighbor mean nuclear volume frame-2':'mean_neighb_nuc_vol_24h'
                ,'Std neighbor nuclear volume normalized':'std_neighb_nuc_vol'
                # ,'Max neighbor nuclear volume normalized':'max_neighb_nuc_vol'
                # ,'Min neighbor nuclear volume normalized':'min_neighb_nuc_vol'
                ,'Mean neighbor dist':'mean_neighb_dist'
                ,'Mean neighbor FUCCI intensity':'mean_neighb_fucci_int'
                ,'Frac neighbor FUCCI high':'frac_neighb_fucci_high'

                ,'Frac neighbor FUCCI high':'frac_neighb_fucci_high'
                ,'Neighbor mean height frame-1':'neighb_height_12h'
                # ,'Neighbor mean height frame-2':'neighb_height_24h'

                }

df_g1s = df_.loc[:,list(features_list.keys())]
df_g1s = df_g1s.rename(columns=features_list)

# Standardize
for col in df_g1s.columns:
    df_g1s[col] = z_standardize(df_g1s[col])

df_g1s['G1S_logistic'] = (df_['Phase'] == 'SG2').astype(int)


# Count NANs
print(np.isnan(df_g1s).sum(axis=0))
print('----')

print(f'Num features: {df_g1s.shape[1]}')
print('----')
#% Print some dataframe summaries
print(df_.groupby('Region').count()['basalID'])
print('----')
print('# unique basal cells'); print(df_['UniqueID'].unique().shape)
print('----')
print(df_.groupby('Phase').count()['Region'])

Inan = df_g1s.isnull().any(axis=1).values
df_ = df_[~Inan]
df_g1s = df_g1s[~Inan]

C = EmpiricalCovariance().fit(df_g1s)
sb.heatmap(C.covariance_,xticklabels=df_g1s.columns,yticklabels=df_g1s.columns)
L,D = eig(C.covariance_)

print('----')
print(f'Condition number: {L.max() / L.min()}')


df_.to_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_.csv')
df_g1s.to_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_g1s.csv')


