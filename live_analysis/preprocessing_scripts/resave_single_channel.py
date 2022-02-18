#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:49:45 2022

@author: xies
"""

import numpy as np
from skimage import io
from glob import glob
from os import path


#%%

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/10-20-2021/WT/R1/reg'

imgfiles = glob(path.join(dirname,'*.tif'))

for f in imgfiles:
    im = io.imread(f)
    channel = im[...,1]
    
    path.splitext(f)
    
    io.imsave(path.splitext(f)[0]+'_chan2.tif',channel)
    