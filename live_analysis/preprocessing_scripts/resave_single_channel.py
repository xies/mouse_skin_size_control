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

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/Shared/20210322_K10 revisits/20220322_female4/area1'

imgfiles = glob(path.join(dirname,'*.tif'))

for f in imgfiles:
    im = io.imread(f)
    channel = im[...,2]
    
    path.splitext(f)
    
    io.imsave(path.splitext(f)[0]+'_chan3.tif',channel)
    