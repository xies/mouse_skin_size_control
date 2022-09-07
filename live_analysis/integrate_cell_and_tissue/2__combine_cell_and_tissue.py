#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 19:58:28 2022

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb

from os import path
from glob import glob
from tqdm import tqdm
import pickle as pkl


dirname = dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
ZZ = 72
XX = 460
T = 15

#%% Load

with open(path.join(dirname,'basal_no_daughters.pkl'),'rb') as f:
    collated = pkl.load(f)
    
cell_ts = pd.concat(collated,ignore_index=True)

tissue = pd.read_csv(path.join(dirname,'tissue_dataframe.csv'),index_col = 0)
df = pd.merge(cell_ts, tissue, how='inner', on='basalID')

#%%

sb.pairplot(df,vars=['Nuclear volume','Volume','Height to BM',
                     'Mean curvature','Mean neighbor dist','Nuclear axial angle'
                     ,'Axial angle','Coronal area'])