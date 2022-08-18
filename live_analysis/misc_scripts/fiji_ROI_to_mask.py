#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:16:47 2022

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, transform
from os import path
from re import findall
import pickle as pkl
from glob import glob


from twophoton_util import parse_unaligned_channels, parse_timecourse_directory
dirname = '/Users/xies/OneDrive - Stanford/Skin/06-25-2022/M1 WT/R1/'

#%%

def draw_mask_from_ROI(im,t,coords,Z,matrix,label):
    
    for i,coord in enumerate(coords):
        coord = np.array(coord).T
        coord = transform.matrix_transform(coords = coord, matrix = matrix)
        
        X = coord[:,0]
        Y = coord[:,1]
        X[X.max() >= im.shape[3]] = im.shape[3]
        Y[X.min() < 0] = 0
        Y[Y.max() >= im.shape[2]] = im.shape[2]
        Y[Y.min() < 0] = 0
        im[t,(Z[i] * np.ones(len(X))).astype(int),Y.astype(int),X.astype(int)] = label
        
    return im
        
        
#%%

XX = 1024
ZZ = 88
T = 20

# reg_reg_list = parse_unaligned_channels(dirname)
# align_list = parse_timecourse_directory(dirname)

xfiles = sorted(glob(path.join(dirname,'manual_track/*/*/*.xpts.txt')))
yfiles = sorted(glob(path.join(dirname,'manual_track/*/*/*.ypts.txt')))
zfiles = sorted(glob(path.join(dirname,'manual_track/*/*/*.zpts.txt')))
coordinate_file_tuple = zip(xfiles,yfiles,zfiles)

# Load transformation matrices
_tmp = pkl.load(open(path.join(dirname,'alignment_information.pkl'),'rb'))
dZ = np.array(_tmp[0]) - _tmp[2]
XY_matrices = _tmp[1]

labeled_image = np.zeros((T,ZZ,XX,XX))

for fx,fy,fz in tqdm(coordinate_file_tuple):
    # parse timestamp
    t = int(findall('t(\d+)\.',path.basename(fx))[0])
    cellID = int(path.split(path.split(fx)[0])[1].split('.')[1])
        
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
    Z_ = Z + dZ[t]
    this_matrix = XY_matrices[t]
    
    coords = zip(X,Y)
    labeled_image = draw_mask_from_ROI(labeled_image,t,coords,Z_,this_matrix,cellID)
    
    
    
        
        
        
    