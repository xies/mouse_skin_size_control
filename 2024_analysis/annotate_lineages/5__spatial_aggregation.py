#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 15:43:28 2025

@author: xies
"""

# Core libraries
import numpy as np
from skimage import io, measure, morphology
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pylab as plt

# Specific utils
from imageUtils import draw_labels_on_image, draw_adjmat_on_image, \
    most_likely_label, colorize_segmentation
from mathUtils import get_neighbor_idx, argsort_counter_clockwise

# General utils
from tqdm import tqdm
from os import path
from basicUtils import nonans

dx = 0.25
dz = 1

# Filenames
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

def find_touching_labels(labels, centerID, threshold, selem=morphology.disk(3)):
    this_mask = labels == centerID
    this_mask_dil = morphology.binary_dilation(this_mask,selem)
    touchingIDs,counts = np.unique(labels[this_mask_dil],return_counts=True)
    touchingIDs[counts > threshold] # should get rid of 'conrner touching'

    touchingIDs = touchingIDs[touchingIDs > 2] # Could touch background pxs
    touchingIDs = touchingIDs[touchingIDs != centerID] # nonself
    
    return touchingIDs
    

#% Reconstruct adj network from cytolabels that touch
def get_adjdict_from_2d_segmentation(seg2d:np.array, touching_threshold = 2):
    
    #@todo: OK for 3D segmentation?
    # assert(seg2d.ndim == 2) # only works with 2D images for now
    
    A = {centerID:find_touching_labels(seg2d, centerID, touching_threshold)
         for centerID in np.unique(seg2d)[1:]}
    
    return A

def process(token):
    return token['text']

from collections.abc import Callable
def aggregate_over_adj(adjmat: dict, aggregators: dict[str,Callable],
                       df = pd.DataFrame, fields2aggregate=list[str]):
    
    df_aggregated = pd.DataFrame()
    for centerID,neighborIDs in adjmat.items():
        neighbors = df.loc[neighborIDs]
        
        for agg_name, agg_func in aggregators.items():
            df[fields2aggregate].apply(agg_func)
            
    
    return df_aggregated

#%%

all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints.csv'),index_col=['Frame','TrackID'])

seg2d = [io.imread(path.join(dirname,f'Image flattening/flat_cyto_seg_manual/t{t}.tif'))
         for t in range(15)]
seg3d = [io.imread(path.join(dirname,f'Image flattening/flat_cyto_seg_manual/t{t}.tif'))
         for t in range(15)]

adjDicts = [get_adjdict_from_2d_segmentation(seg) for seg in basal_segmentations]
[process(token) for token in tqdm(adjDicts)]

# Connect the frame, segID into TrackID




aggregators = {'mean',np.mean}
fields2aggregate = ['Nuclear volume']

df = all_df
adjmat = adjMats

# all_df['Time'] = all_df['Frame'] * 12
# tracks = [ _df for _,_df in all_df.groupby('TrackID')]





#% Label transfer from nuc3D -> cyto2D
