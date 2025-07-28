#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 12:59:52 2025

@author: xies
"""

import numpy as np
import pandas as pd
from os import path
from skimage import io
from tqdm import tqdm

dirname ='/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1/'

cyto_seg = io.imread(path.join(dirname,'Mastodon/tracked_cyto.tif'))
all_df = pd.read_pickle(path.join(dirname,'Mastodon/single_timepoints_dynamics.pkl'))

df = all_df[all_df['Fate known','Meta']]
df = df[ ~df['Border','Meta']]
df = df[ df['Cell type','Meta'] == 'Basal']

from imageUtils import colorize_segmentation

#%% Self-fate: diff v. divide

colorized = []
for t in tqdm(df.reset_index().Frame.unique()):
    _df = df.loc[t,:]
    
    colorized.append(colorize_segmentation(cyto_seg[t,...],
                                           (_df['Will differentiate','Meta']+1).to_dict(),
                                           dtype=np.uint8))
colorized = np.stack(colorized)
io.imsave(path.join(dirname,'colorized/will_differentiate.tif'),
          colorized.astype(np.uint8))

#%% Daughter fate: DivDiv, DivDiff, DiffDiff

daughter_known = all_df[ ~np.isnan(all_df['Num daughter differentiated','Meta'])]

colorized = []
for t in tqdm(daughter_known.reset_index().Frame.unique()):
    _df = daughter_known.loc[t,:]
    
    colorized.append(colorize_segmentation(cyto_seg[t,...],
                                           (_df['Num daughter differentiated','Meta']+2).to_dict(),
                                           dtype=np.uint8))
colorized = np.stack(colorized)
io.imsave(path.join(dirname,'colorized/num_daughter_differentiated.tif'),
          colorized.astype(np.uint8))



#%% Cell cycle: G1 v SG2

colors = {'G1':1,'SG2':2,'NA':0}


colorized = []
for t in tqdm(df.reset_index().Frame.unique()):
    _df = df.loc[t,:]
    
    colorized.append(colorize_segmentation(cyto_seg[t,...],
                                           (_df['Cell cycle phase','Meta']).map(colors).to_dict(),
                                           dtype=np.uint8))
colorized = np.stack(colorized)
io.imsave(path.join(dirname,'colorized/cellcycle_g1_sg2.tif'),
          colorized.astype(np.uint8))

plot_track(tracks[637].droplevel(level=1,axis=1),x='Frame',
           y='Mean FUCCI intensity',celltypefield='Cell type')
