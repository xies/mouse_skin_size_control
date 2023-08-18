#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 19:42:06 2023

@author: xies
"""

import numpy as np
import pandas as pd
from os import path
from skimage import io

import pickle as pkl
from tqdm import tqdm

from statsmodels.robust import scale

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/R1/'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R2'

field2plot = 'Specific GR'

#%%

# Load dataframe
with open(path.join(dirname,'manual_tracking/Ablation_R4_Ablation_dense.pkl'),'rb') as f:
    tracks = pkl.load(f)
ts = pd.concat(tracks,ignore_index=True)
ts = ts.dropna(subset=field2plot)
    
# Load segmentation
segmentation = io.imread(path.join(dirname,'manual_tracking/Ablation_R4_Ablation.tif'))
[TT,ZZ,YY,XX] = segmentation.shape

# normalize
values2plot = ts[field2plot].values
values2plot = values2plot/scale.mad(values2plot)

cutoff = 3

values2plot[values2plot < -cutoff ] = cutoff
values2plot[values2plot > cutoff ] = cutoff

# change all
values2plot = values2plot / cutoff

#%%

visualization = np.zeros_like(segmentation,dtype=float)
for (i,(_,row)) in enumerate(tqdm(ts.iterrows())):

    frame = row['Frame']
    cellID = row['CellID']
    mask = segmentation[frame,...] == cellID
    
    visualization[frame,mask] = values2plot[i]
    
#%%

im = visualization.copy()
im = (im * 65535/2) + 65535/2
im = im.astype(np.uint16)

io.imsave(path.join(dirname,f'Visualizations/{field2plot}.tif'), im)

