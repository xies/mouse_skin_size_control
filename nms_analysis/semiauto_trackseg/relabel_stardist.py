#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 20:40:45 2021

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io,morphology
import seaborn as sb
from os import path
from glob import glob

import pickle as pkl

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-03-2021 Rb-fl/M1 WT/R1/'

#%% Reindex stardist labels

tiff_list = glob(path.join(dirname,'stardist/round2/training/[1-9].tif'))
training = map(io.imread, tiff_list )

for fname, img in zip(tiff_list,training):
    print(f'{fname}')
    labels = img[:,0,...]
    
    all_labels = np.unique(labels)
    all_labels = all_labels[all_labels > 0]
    
    for i,l in enumerate(all_labels):
        labels[labels == l] = i
        
    io.imsave(path.join( path.dirname(fname), f'{path.splitext(path.basename(fname))[0]}_idx.tif'), labels.astype(np.int16))
    

#%% Convert indexed label images into fore/background labels

# First need to erode each object 1 px away from border ( avoid overlap )

# Load prediction by stardist (indexed labels in channel 1)
tiff_list = glob(path.join(dirname,'stardist/round2/training/*.tif'))
training = map(io.imread, tiff_list )

for fname, img in zip(tiff_list,training):
    print(f'{fname}')
    labels = img[:,0,...]
    fg = np.zeros_like(labels,dtype=bool)
    
    all_labels = np.unique(labels)
    all_labels = all_labels[all_labels > 0]
    for l in all_labels:
        this_object = labels == l
        fg = fg | morphology.erosion(this_object)
        
    io.imsave(path.join( path.dirname(fname), f'{path.splitext(path.basename(fname))[0]}_fg.tif'), fg)
    