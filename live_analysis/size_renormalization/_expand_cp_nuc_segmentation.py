#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:06:37 2023

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io,morphology, filters

from os import path
from glob import glob

import seaborn as sb

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

#%%

seg = io.imread(path.join(dirname,'3d_nuc_seg/manual_seg_no_track.tif'))

#%%

t = 0

this_seg = seg[t,...]

footprint = morphology.cube(5)

this_seg_dilated = morphology.dilation(this_seg,footprint=footprint)

im = io.imread(path.join(dirname,f'im_seq/t{t}.tif'))[...,2]

#%%

