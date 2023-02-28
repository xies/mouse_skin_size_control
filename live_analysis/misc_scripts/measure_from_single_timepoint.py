#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 18:06:43 2023

@author: xies
"""

from skimage import measure, io
from tqdm import tqdm
from glob import glob
from os import path

#%%

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/11-17-2022 RB-KO tam control/M*/R1/'

im_filenames = glob(path.join(dirname,''

seg_filenames