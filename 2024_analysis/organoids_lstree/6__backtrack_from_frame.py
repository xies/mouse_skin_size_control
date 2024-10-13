#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:52:56 2024

@author: xies
"""

import pandas as pd
import numpy as np
from os import path

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'
dx = 0.26
dz = 2

df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features.csv'),index_col=0)

#%%

df_by_frame = {frame:_df for frame,_df in df.groupby('Frame')}
tracks = {ID:t for ID,t in df.groupby('trackID')}

for trackID,track in tracks.items():
    
    track['']