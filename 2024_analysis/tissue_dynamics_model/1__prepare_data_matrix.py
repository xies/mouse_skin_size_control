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
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from os import path

from statsmodels.stats.outliers_influence import variance_inflation_factor

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

# Make ratios some of the correlated components
df_['Height to BM relative to cell height'] = df_['Height to BM'] / df_['Height']
df_['NC ratio (sm)'] = df_['Volume (sm)'] / df_['Nuclear volume (sm)']

df_['CV neighbor cell volume'] = df_['Std neighbor cell volume'] / df_['Mean neighbor cell volume']
df_['CV neighbor apical area'] = df_['Std neighbor apical area'] / df_['Mean neighbor apical area']
df_['CV neighbor basal area'] = df_['Std neighbor basal area'] / df_['Mean neighbor basal area']

df_['Neighbor CV cell volume frame-1'] = df_['Neighbor std cell volume frame-1'] / df_['Neighbor mean cell volume frame-1']

df_['G1S_logistic'] = (df_['Phase'] == 'SG2').astype(int)

# Sanitize field names for smf

features_list = { # Cell identity, position
                'Age':'age'
                ,'Differentiating':'diff'
                ,'Y':'y','X':'x'
                # ,'Z':'z'
                ,'Height to BM relative to cell height':'rel_height_to_bm'
                
                # Cell geometry
                ,'Volume (sm)':'vol_sm'
                ,'Nuclear volume (sm)':'nuc_vol_sm'
                # ,'Surface area':'sa'
                ,'SA to vol':'sa_to_vol'
                # ,'Axial component':'axial_moment'
                # ,'Nuclear volume':'nuc_vol'
                ,'NC ratio':'nc_ratio'
                # ,'Nuclear surface area':'nuc_sa'
                # ,'Nuclear axial component':'nuc_axial_moment'
                ,'Nuclear solidity':'nuc_solid'
                # ,'Nuclear axial angle':'nuc_angle'
                ,'Planar eccentricity':'planar_ecc'
                ,'Axial eccentricity':'axial_ecc'
                # ,'Nuclear axial eccentricity':'nuc_axial_ecc'
                # ,'Nuclear planar eccentricity':'nuc_planar_ecc'
                ,'Axial angle':'axial_angle'
                # ,'Planar component 1':'planar_component_1'
                # ,'Planar component 2':'planar_component_2'
                ,'Relative nuclear height':'rel_nuc_height'
                ,'Time to G1S':'time_g1s'
                ,'Basal area':'basal'
                ,'Apical area':'apical'
                
                # Central cell relation to neighborhood
                ,'Coronal eccentricity':'cor_eccentricity'
                # ,'Coronal density':'cor_density'
                ,'Cell alignment to corona':'cell_align_to_coro'
                ,'Mean curvature':'mean_curve'
                # ,'Collagen fibrousness':'col_fib'
                ,'Collagen alignment':'col_align'
                ,'Distance to closest macrophage':'macrophage'
                # ,'Gaussian curvature':'gaussian_curve'
                ,'Delta curvature':'delta_curvature'
                ,'Delta height':'delta_height'
                
                # Current-frame neighborhood stats
                ,'Num diff neighbors':'num_neighb_diff'
                ,'Num planar neighbors':'num_neighb_plan'
                # ,'Mean neighbor dist':'mean_neighb_dist'
                ,'Mean neighbor cell volume':'mean_neighb_vol'
                ,'CV neighbor cell volume':'cv_neighb_vol'
                # ,'Mean neighbor apical area':'mean_neighb_apical'
                ,'CV neighbor apical area':'cv_neighb_apical'
                # ,'Mean neighbor basal area':'mean_neighb_basal'
                ,'CV neighbor basal area':'cv_neighb_basal'
                # ,'Mean neighbor cell height':'mean_neighb_height'
                ,'Max neighbor height from BM':'max_neigb_height_to_bm'
                
                ,'Mean neighbor collagen alignment':'mean_neighb_collagen_alignment'
                ,'Mean neighbor FUCCI intensity':'mean_neighb_fucci_int'
                # ,'Frac neighbor FUCCI high':'frac_neighb_fucci_high'
                
                # Growth rates and other central cell dynamics
                ,'Specific GR spl':'sgr'
                # ,'Exponential growth rate':'exp_gr'
                ,'FUCCI bg sub frame-1':'fucci_int_12h'
                # ,'FUCCI bg sub frame-2':'fucci_int_24h'
                # ,'Volume frame-1':'vol_12h'
                # ,'Volume frame-2':'vol_24h'
                ,'Collagen alignment-1':'collagen_alignment_12h'
                # ,'Collagen alignment-2':'collagen_alignment_24h'
                
                # Neighborhood stats from previous frame(s)
                # ,'Neighbor mean dist frame-1':'mean_neighb_dist_12h'
                # ,'Neighbor mean dist frame-2':'mean_neighb_dist_24h'
                ,'Neighbor mean cell volume frame-1':'mean_neighb_vol_12h'
                # ,'Neighbor mean cell volume frame-2':'mean_neighb_vol_24h'
                ,'Neighbor std cell volume frame-1':'std_neighb_vol_12h'
                # ,'Neighbor std cell volume frame-2':'std_neighb_vol_24h'
                # ,'Neighbor mean height from BM frame-1':'mean_neighb_height_to_bm_12h'
                # ,'Neighbor mean height from BM frame-2':'mean_neighb_height_to_bm_24h'
                ,'Neighbor max height from BM frame-1':'max_neighb_height_to_bm_12h'
                # ,'Neighbor max height from BM frame-2':'max_neighb_height_to_bm_24h'
                ,'Neighbor mean collagen alignment frame-1':'mean_neighb_collagen_align_12h'
                # ,'Neighbor mean collagen alignment frame-2':'mean_neighb_collagen_align_24h'
                ,'Neighbor planar number frame-1':'num_planar_neighb_12h'
                # ,'Neighbor planar number frame-2':'num_planar_neighb_24h'
                ,'Neighbor diff number frame-1':'num_diff_neighb_12h'
                # ,'Neighbor diff number frame-2':'num_diff_neighb_24h'
                # ,'Neighbor mean FUCCI int frame-1':'mean_neighb_fucci_int_12h'
                
                # Bookkeeping for LMM groups or drop for OLS
                ,'basalID':'cellID'
                ,'Region':'region'
                }


df_g1s = df_.loc[:,list(features_list.keys())]
df_g1s = df_g1s.rename(columns=features_list)

# Standardize
for col in df_g1s.columns.drop(['region','cellID']):
    df_g1s[col] = z_standardize(df_g1s[col])

df_g1s['G1S_logistic']= df_['G1S_logistic']

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

df2plot = df_g1s.drop(columns=['region','cellID','G1S_logistic','diff','time_g1s'])

C = MinCovDet().fit(df2plot)
plt.figure()
L,D = eig(C.covariance_)
plt.title(f'MinCovDet, C = {L.max() / L.min():.2f}')
sb.heatmap(C.covariance_,xticklabels=df2plot.columns,yticklabels=df2plot.columns,cmap='vlag',center=0)
plt.gca().set_aspect('equal', 'box')
plt.tight_layout()

print('----')
print(f'Condition number (MinCovDet): {L.max() / L.min():.2f}')

df_.to_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_.csv')
df_g1s.to_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_g1s.csv')


#%%

plt.figure()
df2plot = df_g1s.drop(columns=['nuc_vol_sm','cellID','region','G1S_logistic','time_g1s','diff'])
C = MinCovDet().fit(df2plot)
L,D = eig(C.covariance_)
stats = pd.Series({df2plot.columns[i]:variance_inflation_factor(df2plot,i) for i in range(len(df2plot.columns))},name='VIF')
stats.sort_values().plot.bar()
plt.tight_layout()
plt.title(f'Cell volume, C = {L.max() / L.min():.2f}')

plt.figure()
df2plot = df_g1s.drop(columns=['vol_sm','cellID','region','G1S_logistic','time_g1s','diff'])
C = MinCovDet().fit(df2plot)
L,D = eig(C.covariance_)
stats = pd.Series({df2plot.columns[i]:variance_inflation_factor(df2plot,i) for i in range(len(df2plot.columns))},name='VIF')
stats.sort_values().plot.bar()
plt.tight_layout()
plt.title(f'Nuc volume C = {L.max() / L.min():.2f}')

#%%



