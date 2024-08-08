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

dirname = '/Users/xies/OneDrive - Stanford/In vitro/p107 dynamics/NIH3T3 p107 knockdown/08-01-2024 3T3 Anti-107'

diameter = 25 #27 OK for 1.5x BE basal cells at 1.4 zoomin
cellprob_threshold = -0.1

filelist = glob(path.join(dirname,'*/Pos*/*channel001*.tif'))

OVERWRITE = False

for f in tqdm(filelist):

    basename = path.splitext(f)[0] # i.e. 't9'
    subdir = path.dirname(f) # i.e. 't9'

    im = io.imread(f)

    tic = time()
    print(f'Predicting on {f}')

    # 2D gaussian blur the image
    im = ndimage.gaussian_filter(im,sigma=1)
    masks,flows,styles,diams = model.eval(im,diameter=None,
					cellprob_threshold=cellprob_threshold)
    io.masks_flows_to_seg(im, masks,flows,diams,f)

	# Also resave the mask as standalone .tif for convenience
    io.imsave(path.join(subdir, basename + '_masks.tif'),masks)

    toc = time()
    print(f'Saved to {subdir}')
    print(f'Processed in {toc- tic:0.4f} seconds')
