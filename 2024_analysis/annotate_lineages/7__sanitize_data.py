#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 15:07:15 2025

@author: xies
"""


# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb

# General utils
from tqdm import tqdm
from os import path

from sklearn import preprocessing, model_selection
from sklearn import ensemble, metrics

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated_lookback.csv'),
                     index_col=['Frame','TrackID'])

#%% Designate the bookkeeping columns

meta_cols = ['TrackID','LineageID','Left','Right','Division','Terminus',
             'Mother','Sister','Daughter a','Daughter b','Cell type','Reviewed',
             'Cutoff','Complete cycle','Will divide','Divide next frame',
             'Differentiated','Will differentiate',
             'Delaminate next frame','Time to differentiation',
             'Keep until first differentiation']

Imeta = np.isin(all_df.columns,meta_cols)
metadata_index = {True:'Meta',False:'Measurement'}
metadata_index = [metadata_index[x] for x in Imeta]

new_cols = pd.DataFrame()
new_cols['Metadata'] = metadata_index
new_cols['Name'] = all_df.columns

all_df.columns = pd.MultiIndex.from_frame(new_cols)

