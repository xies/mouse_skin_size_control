#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:57:59 2019

@author: xies
"""

import pandas as pd
import pickle as pkl
import numpy as np


#Load df from pickle
r1 = pd.read_pickle('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/exports/dataframe.pkl')
r2 = pd.read_pickle('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/exports/dataframe.pkl')

# r5 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/Mesa et al/W-R5/tracked_cells/dataframe.pkl')
# r5f = pd.read_pickle('/Users/xies/Box/Mouse/Skin/Mesa et al/W-R5-full/tracked_cells/dataframe.pkl')
df = pd.concat((r1,r2))

# df.to_pickle('/Users/xies/Box/Mouse/Skin/Mesa et al/cell_summary.pkl')

# Load growth curves from pickle
with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/exports/collated_manual.pkl','rb') as f:
    c1 = pkl.load(f,encoding='latin-1')
with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/exports/collated_manual.pkl','rb') as f:
    c2 = pkl.load(f,encoding='latin-1')
# with open('/Users/xies/Box/Mouse/Skin/Mesa et al/W-R5/exports/collated_manual.pkl','rb') as f:
#     c5 = pkl.load(f,encoding='latin-1')
# with open('/Users/xies/Box/Mouse/Skin/Mesa et al/W-R5-full/exports/collated_manual.pkl','rb') as f:
#     c5f = pkl.load(f,encoding='latin-1')
collated = c1+c2

# with open('/Users/xies/Box/Mouse/Skin/Mesa et al/time_series.pkl','wb') as f:
#     pkl.dump(collated,f)

#df = df[~df.Mitosis]
Ncells = len(df)


# Filter for phase-annotated cells in collated
collated_filtered = [c for c in collated if c.iloc[0]['Phase'] != '?']

# Filter for cells that have daughter data
df_has_daughter = df[~np.isnan(df['Division volume interpolated'])]

# Concatenate all collated cells into dfc
dfc = pd.concat(collated_filtered)

#%% Alternatively, load all the series

with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/tracked_data_collated/cell_summary.pkl','rb' ) as f:
    df = pkl.load(f)

with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/tracked_data_collated/time_series.pkl','rb' ) as f:
    ts = pkl.load(f)
    
    
