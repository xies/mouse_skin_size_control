#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:59:27 2024

@author: xies
"""

import pandas as pd
import pickle as pkl
import numpy as np

#Load df from pickle
r1 = pd.read_pickle('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/2020 CB analysis/exports/dataframe.pkl')
r2 = pd.read_pickle('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/2020 CB analysis/exports/dataframe.pkl')

r5 = pd.read_pickle('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R5/tracked_cells/dataframe.pkl')
r5f = pd.read_pickle('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R5-full/tracked_cells/dataframe.pkl')
df = pd.concat((r1,r2,r5,r5f))

df.to_pickle('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/cell_summary.pkl')

# Load growth curves from pickle
with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/2020 CB analysis/exports/collated_manual.pkl','rb') as f:
    c1 = pkl.load(f,encoding='latin-1')
with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/2020 CB analysis/exports/collated_manual.pkl','rb') as f:
    c2 = pkl.load(f,encoding='latin-1')
with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R5/2020 CB analysis/exports/collated_manual.pkl','rb') as f:
    c5 = pkl.load(f,encoding='latin-1')
with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R5-full/tracked_cells/collated_manual.pkl','rb') as f:
    c5f = pkl.load(f,encoding='latin-1')
collated = c1+c2+c5+c5f

# with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/time_series.pkl','wb') as f:
#     pkl.dump(collated,f)

#df = df[~df.Mitosis]
Ncells = len(df)


# Filter for phase-annotated cells in collated
collated_filtered = [c for c in collated if c.iloc[0]['Phase'] != '?']

# Filter for cells that have daughter data
df_has_daughter = df[~np.isnan(df['Division volume interpolated'])]

# Concatenate all collated cells into dfc
dfc = pd.concat(collated_filtered)
cells = [c for _,c in dfc.groupby('CellID')]

#%%

import seaborn as sb

subsampled = [c.iloc[np.random.randint(low=0,high=2)::2] for c in cells]

bsize = np.array([c.iloc[0]['Volume'] for c in subsampled])

g1size = np.ones(len(subsampled)) * np.nan
for i,c in enumerate(subsampled):
    
    I = np.where(c.Phase == 'SG2')[0]
    if I.sum() > 0:
        g1size[i] = c.iloc[I[0]]['Volume']

g1growth = g1size - bsize

df24 = pd.DataFrame()
df24['Birth nuc vol 24h'] = bsize
df24['G1 nuc grown 24h'] = g1growth

sb.regplot(df,x='Birth nuc volume',y='G1 nuc grown')
sb.regplot(df24,x='Birth nuc vol 24h',y='G1 nuc grown 24h')


        