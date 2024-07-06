#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:16:49 2024

@author: xies
"""

import numpy as np
from skimage import io, measure, draw, util, morphology
from scipy.spatial import distance, Voronoi, Delaunay
import pandas as pd

from trimesh import Trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure, sphere_ball_intersection
from basicUtils import euclidean_distance

import matplotlib.pylab as plt
# from matplotlib.path import Path
# from roipoly import roipoly
from imageUtils import draw_labels_on_image, draw_adjmat_on_image, most_likely_label, colorize_segmentation
from mathUtils import get_neighbor_idx, surface_area, parse_3D_inertial_tensor, argsort_counter_clockwise

from tqdm import tqdm
from glob import glob
from os import path

dx = 0.25
XX = 460
Z_SHIFT = 10

# Differentiating thresholds
centroid_height_cutoff = 3.5 #microns above BM

SAVE = True
VISUALIZE = True
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

# FUCCI threshold (in stds)
alpha_threshold = 1
#NB: idx - the order in array in dense segmentation

#%% Merge with dense cortical segmentation

df = pd.read_csv(path.join(dirname,'nuc_dataframe.csv'),index_col=0)

df_cyto = pd.read_csv(path.join(dirname,'cyto_dataframe.csv'),index_col=0)
df = pd.merge(df,df_cyto,on=['Frame','CellposeID'],how='left')

    
df.to_csv(path.join(dirname,'tissue_dataframe.csv'))
print(f'Saved to: {dirname}')
