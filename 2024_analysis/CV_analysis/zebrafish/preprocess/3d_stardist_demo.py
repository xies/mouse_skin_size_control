#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 06:39:57 2024

@author: xies
"""

from skimage import io, exposure, util
from stardist.models import StarDist3D
from os import path
from csbdeep.utils import normalize

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/diTalia_zebrafish/osx_fucci_26hpp_11_4_17/stardist/training'

stack = io.imread(path.join(dirname,'t65_summed.tif'))

# prints a list of available models
StarDist3D(None, name='stardist', basedir='models')
s
# model = StarDist2D(None, name='stardist', basedir='models')
# creates a pretrained model
model = StarDist3D.from_pretrained('3D_demo')

labels,_ = model.predict_instances(normalize(stack))

io.imsave(path.join(dirname,'t65_summed_3d_labels.tif'), labels)
