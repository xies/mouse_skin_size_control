#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 17:00:09 2021

@author: xies
"""

import numpy as np
import matplotlib.pylab as plt

from scipy.ndimage.morphology import distance_transform_edt
from skimage import io, morphology, filters, measure, segmentation, util

from os import path
from glob import glob

import pickle as pkl
B+B
dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-03-2021 Rb-fl/M1 WT/R1/'

#%% Generate masks

tiff_list = glob(path.join(dirname,'stardist/round2/training/[1-9].tif'))
training = map(io.imread, tiff_list )

for fname, img in zip(tiff_list,training):
    print(f'{fname}')
    labels = img[:,0,...]
    raw = img[:,1,...]
    raw_ = filters.gaussian(raw,sigma=0.5)
    
    th = filters.threshold_otsu(raw_)
    mask = raw_ > th
     
    io.imsave(path.join( path.dirname(fname), f'{path.splitext(path.basename(fname))[0]}_otsu.tif'), mask.astype(np.int16))
    
    #% Generate seeds from stardist
    
    # skel = morphology.skeletonize(labels)
    props = measure.regionprops(labels)
    centroids = np.array([ p['Centroid'] for p in props],dtype = int)
    
    
    seeds = np.zeros_like(labels)
    
    for i,ct in enumerate(centroids): # Can't figure out multiindexing
        seeds[ct[0],ct[1],ct[2]] = i
    
    D = distance_transform_edt(mask)
    
    io.imsave(path.join( path.dirname(fname), f'{path.splitext(path.basename(fname))[0]}_dist.tif'), D.astype(np.int16))
    
    seg  = segmentation.watershed(-D,seeds)
    
    io.imsave(path.join( path.dirname(fname), f'{path.splitext(path.basename(fname))[0]}_seg.tif'), seg.astype(np.int16))
    
