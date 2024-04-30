#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:00:22 2023

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from skimage import io, measure, filters
from scipy.ndimage import distance_transform_edt
from glob import glob
from os import path
from tqdm import tqdm

from pyvoro import compute_voronoi
from re import match

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

#%%



vor_seg = io.imread(path.join(dirname,'Image flattening/voro_seg.tif'))

#%%

for t in range(15):
    im = io.imread(path.join(dirname,f'Image flattening/flat_B/t{t}.tif'))[...,2]
    plt.plot(im.sum(axis=1).sum(axis=1))

#%%

im_th = im > filters.threshold_otsu(im)

