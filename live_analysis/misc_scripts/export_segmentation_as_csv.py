#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 21:26:50 2023

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, measure
from os import path

from tqdm import tqdm

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

#%%

T = 15

_tmp = []
for t in tqdm(range(T)):
    
    im = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}_basal.tif'))
    
    df = pd.DataFrame(measure.regionprops_table(im,properties=['label','centroid']))
    df['T'] = t
    
    df = df.rename(columns={'centroid-0':'Z','centroid-1':'Y','centroid-2':'X','label':'CellposeID'})
    df['Y'] /= 4
    df['X'] /= 4
    
    _tmp.append(df)
    
df = pd.concat(_tmp,ignore_index=True)

df.to_csv(path.join(dirname,'3d_nuc_seg/cellpose_cleaned_manual/zyxt.csv'))

