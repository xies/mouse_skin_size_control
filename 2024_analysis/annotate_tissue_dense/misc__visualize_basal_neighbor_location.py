#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:55:05 2024

@author: xies
"""


import numpy as np
from skimage import io, measure, draw, util, morphology
import pandas as pd

from basicUtils import euclidean_distance

import matplotlib.pylab as plt
from imageUtils import draw_labels_on_image, draw_adjmat_on_image, most_likely_label, colorize_segmentation

from tqdm import tqdm
from os import path

dx = 0.25
XX = 460

SAVE = True
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

df = []

for t in range(15):

    nuc_seg = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
    manual_tracks = io.imread(path.join(dirname,f'manual_basal_tracking/sequence/t{t}.tif'))
    df_manual = pd.DataFrame(measure.regionprops_table(manual_tracks,intensity_image = nuc_seg,
                                                        properties = ['label'],
                                                        extra_properties = [most_likely_label]))
    df_manual = df_manual.rename(columns={'label':'basalID','most_likely_label':'CellposeID'})
    adj_dict = np.load(path.join(dirname,f'Image flattening/flat_adj_dict/adjdict_t{t}.npy'),allow_pickle=True).item()
    
    # Find the corresponding manually segmented ID/volume
    cells2highlight = []
    for _,this_cell in df_manual.iterrows():
         # Find if all the adjacent cells are cyto annotated
         cells2highlight.extend(adj_dict[this_cell['CellposeID']])
    
    # print(cells2highlight)
    im2save = np.zeros_like(nuc_seg)
    for ID in cells2highlight:
        im2save[nuc_seg == ID] = ID
        
    io.imsave(path.join(dirname,f'manual_basal_tracking/neighborhood_cellposeIDs/t{t}.tif'),im2save)

