#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 23:07:46 2023

@author: xies
"""

import pandas as pd
import numpy as np
import networkx as nx
from skimage import io

from os import path
from tqdm import tqdm

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

seg_notrack = io.imread(path.join(dirname,'3d_nuc_seg/manual_seg_no_track.tif'))

#%% Convert excel annotation into image

mother_annotations = pd.read_excel(path.join(dirname,'manual_basal_tracking_mothers/mother_annotations.xlsx'),index_col=0)

basalIDs = mother_annotations.index
manual_basal_tracking_mothers = np.zeros_like(seg_notrack)

for bID in tqdm(basalIDs):
    
    frame = int(mother_annotations.loc[bID]['Mother frame'])
    motherID = mother_annotations.loc[bID]['Mother cellposeID']
    
    mask = seg_notrack[frame,...] == motherID
    
    manual_basal_tracking_mothers[frame,mask] = bID
    
io.imsave(path.join(dirname,'manual_basal_tracking_mothers/manual_basal_tracking_mothers.tif'),manual_basal_tracking_mothers)

#%% Load mother, cell, and daughters

manual_basal_tracking_mothers = 
manual_basal_tracking_daughters = 