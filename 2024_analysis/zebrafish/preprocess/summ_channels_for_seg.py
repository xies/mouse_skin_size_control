#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:00:02 2024

@author: xies
"""


import pandas as pd
from os import path
from glob import glob
from natsort.natsort import natsorted
from skimage import io, transform, util, registration
import numpy as np
from tqdm import tqdm

from scipy import ndimage
from pystackreg import StackReg

dirname = '/Users/xies/OneDrive - Stanford/In vitro/CV from snapshot/zebrafish_ditalia/osx_fucci_26hpp_11_4_17/'

#%%

mch_files = natsorted(glob(path.join(dirname,'aligned_stacks/*_mch_*.tif')))
venus_files = natsorted(glob(path.join(dirname,'aligned_stacks/*_venus_*.tif')))

file_tuple = zip(mch_files,venus_files)

for t,(m,v) in enumerate(file_tuple):
    mCh = io.imread(m)
    venus = io.imread(v)
    
    io.imsave(path.join(dirname,f'aligned_stacks/aligned_stacks_summed_t{t:02d}.tif'),mCh+venus)
    