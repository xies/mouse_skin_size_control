#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 9

@author: xies
"""

import numpy as np
from cellpose import models
from cellpose import io
from scipy import ndimage

from glob import glob
from os import path
from shutil import move
from os import mkdir
from time import time
from tqdm import tqdm

model = models.Cellpose(model_type='nuclei')

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Drosophila/Gut unwrap/2021-05-14_starved_1-172_slider'

diameter = 20 #27 OK for 1.5x BE basal cells at 1.4 zoomin
anisotropy = 4.0
cellprob_threshold = -0.1

# Load the raw image (RGB,Z,X,Y)
# filenames = []
# for dirname in dirnames:
# 	filenames = filenames + glob(path.join(dirname,'t*.tif'))

OVERWRITE = False

f = path.join(dirname,'h2b.tif')
stack = io.imread(f)
TT = stack.shape[0]

for t in tqdm(range(TT)):
    
    im = stack[t,...]

    tic = time()
    print(f'Predicting on {t}')
    
    # 3D gaussian blur the image
    im = ndimage.gaussian_filter(im,sigma=1)
    
    masks,flows,styles,diams = model.eval(im,diameter=None, do_3D=True,
    				cellprob_threshold=cellprob_threshold, anisotropy=anisotropy)
    io.masks_flows_to_seg(im, masks,flows,diams,f)
    # annoyingly, need to manually move
    #move(path.join(d, basename + '_seg.npy'), path.join(output_dir,basename + '_seg.npy'))
    
    # Also resave the mask as standalone .tif for convenience
    io.imsave(path.join(dirname, f'nuc_masks/t{t}_masks.tif'),masks)
    
    toc = time()
    print(f'Processed frame={t} in {toc- tic:0.4f} seconds')


