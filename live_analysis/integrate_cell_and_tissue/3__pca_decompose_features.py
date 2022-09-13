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


dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
df = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)

#%% NaN check

df_pca = df.drop(columns = ['basalID','CellposeID','G1S frame','Phase','Border','Differentiating',
                            'Mean diff neighbor height','Neighbor mean height frame-2','Volume','Growth rate'])


print(df_pca.columns.values[ np.any(np.isnan(df_pca.values),axis=0)])
print(np.isnan(df_pca.values).sum(axis=0))

X = df_pca.values
I = np.all(~np.isnan(X),axis=1)
df_pca = df_pca.loc[I]

#%%

X = df_pca.values
pca = PCA(n_components = 20)

pca.fit(X)

plt.plot(pca.explained_variance_ratio_); plt.ylabel('% variance explained');plt.xlabel('Components')
# plt.figure(); plt.plot(pca.singular_values_); plt.ylabel('Singular value');plt.xlabel('Components')

plt.figure();plt.plot(pca.components_[0,:]); plt.ylabel('Original dimensions');plt.xlabel('PC 1')
order = np.argsort(pca.components_[0,:])[::-1]
print(f'Top 5 PC1 dimensions: {df_pca.columns[order[0:10]]}')

