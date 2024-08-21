#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 21:52:47 2024

@author: xies
"""

import numpy as np
from skimage import io
from stardist.models import StarDist2D
from os import path
from csbdeep.utils import normalize

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/diTalia_zebrafish/osx_fucci_26hpp_11_4_17/stardist/'

stack = io.imread(path.join(dirname,'test_images/t00_summed.tif'))

# prints a list of available models
StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

labels = []
for im in stack:
    l, _ = model.predict_instances(normalize(im))
    labels.append(l)

io.imsave(path.join(dirname,'test_images/t00_summed_2d_labels.tif'), np.stack(labels))
