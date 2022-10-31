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

