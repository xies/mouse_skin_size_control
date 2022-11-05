#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:04:34 2022

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb

from os import path
from glob import glob
from tqdm import tqdm

from sklearn.decomposition import PCA
from mathUtils import z_standardize

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
df = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)

def plot_principle_component(df,pca,comp):
    
    I = np.argsort( np.abs(pca.components_[comp,:]) )
    plt.figure();plt.bar(range(P),np.abs(pca.components_[comp,I]));
    plt.ylabel('Original dimensions');plt.xlabel('PC 1')
    plt.xticks(range(P),rotation=45, ha='right')
    plt.gca().set_xticklabels(df.columns[I])
    order = np.argsort(pca.components_[1,:])[::-1]
    print(f'Top 5 PC1 dimensions: {df_pca.columns[order[0:10]]}')


#%% NaN check

df_pca = df.drop(columns = ['basalID','CellposeID','G1S frame','Phase','Border','Differentiating',
                            # 'Mean diff neighbor height','Neighbor mean height frame-2',
                            'Volume','Growth rate f','Growth rate b',
                            'Collagen orientation','Basal orientation','Coronal angle','Nuclear planar orientation'
                            'Z_y','X-pixels_x','Y-pixels_x','X-pixels_y','Y-pixels_y','X_y','Y_y','Z_x','X_x','Y_x'
                            ])

for col in df_pca.columns[df_pca.columns != 'G1S_logistic']:
    df_pca[col] = z_standardize(df_pca[col])

print(df_pca.columns.values[ np.any(np.isnan(df_pca.values),axis=0)])
print(np.isnan(df_pca.values).sum(axis=0))

X = df_pca.values
I = np.all(~np.isnan(X),axis=1)
df_pca = df_pca.loc[I]

N,P = X.shape

#%%

X = df_pca.values
pca = PCA(n_components = 5)

pca.fit(X)

#%%
plt.figure()
plt.bar(range(5),pca.explained_variance_ratio_); plt.ylabel('% variance explained');plt.xlabel('Components')
# plt.figure(); plt.plot(pca.singular_values_); plt.ylabel('Singular value');plt.xlabel('Components')

# plot_principle_component(df_pca,pca,0)
plot_principle_component(df_pca,pca,1)
# plot_principle_component(df_pca,pca,2)
# plot_principle_component(df_pca,pca,19)


#%% Sanitize field names for smf

features_list = { # Cell geometry
                'Age':'age'
                # ,'Z_x':'z','Y_x':'y','X_x':'x'
                ,'Volume (sm)':'vol_sm'
                ,'Axial component':'axial_moment'
                ,'Nuclear volume':'nuc_vol'
                ,'Nuclear surface area':'nuc_sa'
                ,'Nuclear axial component':'nuc_axial_moment'
                ,'Nuclear solidity':'nuc_solid'
                ,'Nuclear axial angle':'nuc_angle'
                ,'Planar eccentricity':'planar_ecc'
                ,'Axial eccentricity':'axial_ecc'
                ,'Axial angle':'axial_angle'
                ,'Planar component 1':'planar_component_1'
                ,'Planar component 2':'planar_component_2'
                ,'Relative nuclear height':'rel_nuc_height'
                ,'Surface area':'sa'
                ,'SA to vol':'ratio_sa_vol'
                ,'Time to G1S':'time_g1s'
                ,'Basal area':'basal'
                ,'Apical area':'apical'
                ,'Basal orientation':'basal_orien'
                
                # Growth rates
                ,'Specific GR b (sm)':'sgr'
                ,'Height to BM':'height_to_bm'
                ,'Mean curvature':'mean_curve'
                ,'Gaussian curvature':'gaussian_curve'
                
                # Neighbor topolgy and
                ,'Coronal angle':'cor_angle'
                ,'Coronal density':'cor_density'
                ,'Cell alignment':'cell_align'
                ,'Mean neighbor dist':'mean_neighb_dist'
                ,'Neighbor mean height frame-1':'neighb_height_12h'
                ,'Neighbor mean height frame-2':'neighb_height_24h'
                ,'Num diff neighbors':'neighb_diff'
                ,'Num planar neighbors':'neighb_plan'
                ,'Collagen fibrousness':'col_fib'
                ,'Collagen alignment':'col_align'}

df_g1s = df_.loc[:,list(features_list.keys())]
df_g1s = df_g1s.rename(columns=features_list)
# df_g1s['G1S_logistic'] = (df_['Phase'] == 'SG2').astype(int)

df_g1s_test = df_test_.loc[:,list(features_list.keys())]
df_g1s_test = df_g1s_test.rename(columns=features_list)
df_g1s_test['G1S_logistic'] = (df_test_['Phase'] == 'SG2').astype(int)

# Standardize
for col in df_g1s.columns:
    df_g1s[col] = z_standardize(df_g1s[col])

for col in df_g1s_test.columns:
    df_g1s_test[col] = z_standardize(df_g1s_test[col])

# Count NANs
print(np.isnan(df_g1s).sum(axis=0))

#%% PCA regression

from sklearn.decomposition import PCA

N = 20
df_g1s_nonan = df_g1s.dropna()

pca = PCA(n_components = N)
X_ = pca.fit_transform(df_g1s_nonan.drop(columns='sgr'))

names = [f'PC{x}' for x in range(N)]
X_ = pd.DataFrame(X_,columns=names)
X_['sgr'] = df_g1s_nonan['sgr']

model_pca = smf.rlm('sgr ~ ' + str.join(' + ', X_.columns.drop('sgr')),
                    data=X_).fit()
print(model_pca.summary())

C = model_pca.cov_params()
sb.heatmap(C,xticklabels=True,yticklabels=True)
L,D = eig(C)

print(f'Covariance eigenvalue ratio: {L.max()/L.min()}')

def plot_principle_component(df,pca,comp):
    P = df_g1s_nonan.shape[1]-1
    I = np.argsort( np.abs(pca.components_[comp,:]) )
    plt.figure();plt.bar(range(P),np.abs(pca.components_[comp,I]));
    plt.ylabel('Original dimensions');plt.xlabel('fPC {comp}')
    plt.xticks(range(P),rotation=45, ha='right')
    plt.gca().set_xticklabels(df.columns[I])
    order = np.argsort(pca.components_[1,:])[::-1]
    print(f'Top 5 PC1 dimensions: {df.columns[order[0:10]]}')

plot_principle_component(df_g1s_nonan.drop(columns='sgr'),pca,8)
