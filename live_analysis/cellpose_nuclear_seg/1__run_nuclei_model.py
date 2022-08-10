#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 9

@author: xies
"""

import numpy as np
from cellpose import models
from cellpose import io

from glob import glob
from os import path

model = models.Cellpose(model_type='nuclei')

dirnames = ['/home/xies/data/Skin/06-25-2022/M6 RBKO/R1/im_seq']
dirnames.append('/home/xies/data/Skin/06-25-2022/M1 WT/R1/im_seq')

diameter = 30 #OK for 1.5x BE basal cells at 1.4 zoomin

# Load the raw image (RGB,Z,X,Y)
filenames = []
for dirname in dirnames:
	filenames = filenames + glob(path.join(dirname,'t*[!seg].tif'))
	filenames = filenames + glob(path.join(dirname,'t?.tif'))
channels = [0,2]

for f in filenames:
	if path.exists(path.splitext(f)[0] + '_seg.npy'):
		continue
	im = io.imread(f)
	masks,flows,styles,diams = model.eval(im,diameter=None,channels=channels)
	f = path.splitext(f)[0]
	io.masks_flows_to_seg(im, masks,flows,diams,f,channels)
