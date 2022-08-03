#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:43:56 2022

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, filters, morphology, segmentation
from skimage.feature import peak_local_max
from os import path
from glob import glob

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import label
from ifUtils import delete_border_objects

dirname = '/Users/xies/OneDrive - Stanford/Skin/Confocal/07-19-2022 Skin/DS 06-25-22 H2B-Cerulean FUCCI2 Phall647'

#%%

wt = io.imread(path.join(dirname,'KO3/KO3.tif'))

th = filters.threshold_li(wt)

mask = wt > th
io.imsave(path.join(dirname,'KO3/KO3_mask.tif'),mask)

#%% 2D seed finding

# basal slice should be the 'densest'
basalZ = mask.sum(axis=1).sum(axis=1).argmax()
basal_mask = mask[basalZ,:,:]
basal_mask = morphology.binary_erosion(basal_mask,selem=morphology.disk(15))


# Use distance transform on mask to get seeds
bwmap_basal = distance_transform_edt(basal_mask)

# local_peaks = peak_local_max(bwmap_basal, min_distance=20)
Ncells = len(local_peaks)

#%%
seeds = np.zeros_like(mask)
seeds[ tuple( np.vstack( (np.ones((1,Ncells))*basalZ,local_peaks.T)).astype(int) ) ] = 1

# seeds = np.zeros_like(mask)
# seeds[basalZ,:,:] = morphology.binary_erosion(basal_mask,selem=morphology.disk(18))
# seeds[basalZ,:,:] = morphology.skeletonize(seeds[basalZ,:,:])

seeds,_ = label(seeds)

#%% 3D watershed

bwmap = distance_transform_edt(mask)
labels = segmentation.watershed(-bwmap, seeds, mask = mask)
labels = delete_border_objects(labels)

io.imsave(path.join(dirname,'KO3/KO3_labels.tif'),labels)
