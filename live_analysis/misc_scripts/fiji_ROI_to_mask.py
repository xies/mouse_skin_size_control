#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:16:47 2022

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io
from os import path
from re import findall
from glob import glob

from twophoton_util import parse_unaligned_channels
dirname = '/Users/xies/OneDrive - Stanford/Skin/06-25-2022/M6 RBKO/R1/'

#%%

def draw_mask_from_ROI(im,X,Y,Z,label):
    assert(X.max() < im.shape[1])
    assert(Y.max() < im.shape[0])
    
    for i,z in enumerate(Z[0]):
        
        im[z,Y[i],X[i]] = 
        
def apply_transform_to_ROI(matrix,X,Y,z):
    
        
#%%

reg_reg_list = parse_unaligned_channels(dirname)

xfiles = sorted(glob(path.join(dirname,'manual_track/*/*/*.xpts.txt')))
yfiles = sorted(glob(path.join(dirname,'manual_track/*/*/*.ypts.txt')))
zfiles = sorted(glob(path.join(dirname,'manual_track/*/*/*.zpts.txt')))
file_tuple = zip(xfiles,yfiles,zfiles)

# Load transformation matrices



for fx,fy,fz in file_tuple:
    
    # Manually load list because line size is ragged
    with open(fx) as f:
        X = f.readlines()
    with open(fy) as f:
        Y = f.readlines()
    with open(fz) as f:
        Z = f.readlines()
        
    X = [np.array(line.strip('\n').split(','),dtype=int) for line in X]
    Y = [np.array(line.strip('\n').split(','),dtype=int) for line in Y]
    Z = np.array(Z[0].strip('\n').split(','),dtype=int)
    
    # parse timestamp
    t = int(findall('t(\d+)\.',path.basename(fx))[0])
    cellID = int(path.split(path.split(fx)[0])[1].split('.')[1])
    
    im_filename = reg_reg_list['G'].iloc[t]
    
    im = io.imread(im_filename)
    
    Nstack = len(Z)
    
    for i,z in enumerate(Z):
        
        thisX = X[i]
        thisY = Y[i]
        
        mask = draw_mask_from_ROI(im,thisX,thisY,z,cellID)
        
        
        
        
    