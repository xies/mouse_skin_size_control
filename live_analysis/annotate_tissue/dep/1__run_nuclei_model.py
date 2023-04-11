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
from shutil import move
from os import mkdir

model = models.Cellpose(model_type='nuclei')

dirnames = ['/home/xies/data/Skin/06-25-2022/M6 RBKO/R1/im_seq']
dirnames.append('/home/xies/data/Skin/06-25-2022/M1 WT/R1/im_seq')

diameter = 30 #OK for 1.5x BE basal cells at 1.4 zoomin

# Load the raw image (RGB,Z,X,Y)
filenames = []
for dirname in dirnames:
	filenames = filenames + glob(path.join(dirname,'t*.tif'))
channels = [0,2]

OVERWRITE = True

for f in filenames:
	d = path.dirname(f)
	basename = path.splitext(path.basename(f))[0] # i.e. 't9'
	output_dir = path.join(d,basename)
	if path.exists( output_dir ) and not OVERWRITE:
		print(f'Skipping {f}')
		continue
	else:
		mkdir(output_dir)

	print(f'Predicting on {f}')
	im = io.imread(f)
	masks,flows,styles,diams = model.eval(im,diameter=None,channels=channels, do_3d=True)
	io.masks_flows_to_seg(im, masks,flows,diams,f,channels)
	# annoyingly, need to manually move
	move(path.join(d, basename + '_seg.npy'), path.join(output_dir,basename + '_seg.npy'))
	
	# Also resave the mask as standalone .tif for convenience
	io.imsave(path.join(output_dir, basename + '_masks.tif'),masks)

