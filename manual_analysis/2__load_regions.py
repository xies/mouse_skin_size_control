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
r1 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R1/tracked_cells/dataframe.pkl')
r2 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/dataframe.pkl')
r5 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R5/tracked_cells/dataframe.pkl')
r5f = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R5-full/tracked_cells/dataframe.pkl')
df = pd.concat((r1,r2,r5,r5f))

# Load growth curves from pickle
with open('/Users/xies/Box/Mouse/Skin/W-R1/tracked_cells/collated_manual.pkl','rb') as f:
    c1 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/collated_manual.pkl','rb') as f:
    c2 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R5/tracked_cells/collated_manual.pkl','rb') as f:
    c5 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R5-full/tracked_cells/collated_manual.pkl','rb') as f:
    c5f = pkl.load(f)
collated = c1+c2+c5+c5f

#df = df[~df.Mitosis]
Ncells = len(df)


# Filter for phase-annotated cells in collated
collated_filtered = [c for c in collated if c.iloc[0]['Phase'] != '?']

# Filter for cells that have daughter data
df_has_daughter = df[~np.isnan(df['Division volume interpolated'])]

# Concatenate all collated cells into dfc
dfc = pd.concat(collated_filtered)
