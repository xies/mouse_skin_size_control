#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:47:41 2022

Deletes the intermediate files *_reg.tif, results of 1__



@author: xies
"""

from os import path, remove
from glob import glob

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-08-2022/'

#%%

filelist = glob(path.join(dirname,'*/R*/Day*/ZSeries*/B_reg.tif'))
for f in filelist:
    remove(f)

filelist = glob(path.join(dirname,'*/R*/Day*/ZSeries*/G_reg.tif'))
for f in filelist:
    remove(f)
    
filelist = glob(path.join(dirname,'*/R*/Day*/ZSeries*/R_reg.tif'))
for f in filelist:
    remove(f)
        
filelist = glob(path.join(dirname,'*/R*/Day*/ZSeries*/R_shg_reg.tif'))
for f in filelist:
    remove(f)