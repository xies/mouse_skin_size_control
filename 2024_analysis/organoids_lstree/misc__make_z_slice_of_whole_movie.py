#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 22:44:41 2025

@author: xies
"""

import numpy as np
from skimage import io
from os import path
from glob import glob
from natsort import natsort

dirname ='/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'

c00 = natsort.natsorted(glob(path.join(dirname,'Channel0-Deconv/*.tif')))[:80]
c01 = natsort.natsorted(glob(path.join(dirname,'Channel1-Deconv/*.tif')))[:80]
c02 = natsort.natsorted(glob(path.join(dirname,'Channel2-Deconv/*.tif')))[:80]

z2slice = 20

stack = np.zeros((3,80,404,396))
for t,(h2b,cdt,gem) in enumerate(zip(c00,c01,c02)):
   
    h2b = io.imread(h2b)
    cdt = io.imread(cdt)
    gem = io.imread(gem)
    
    stack[0,t,...] = h2b[z2slice,...]
    stack[1,t,...] = cdt[z2slice,...]
    stack[2,t,...] = gem[z2slice,...]
    
io.imsave('/Users/xies/Desktop/ch00.tif', stack[0,...])
io.imsave('/Users/xies/Desktop/ch01.tif', stack[1,...])
io.imsave('/Users/xies/Desktop/ch02.tif', stack[2,...])

stack = np.zeros((3,80,404,396))
for t,(h2b,cdt,gem) in enumerate(zip(c00,c01,c02)):
   
    h2b = io.imread(h2b)
    cdt = io.imread(cdt)
    gem = io.imread(gem)
    
    stack[0,t,...] = h2b.max(axis=0)
    stack[1,t,...] = cdt.max(axis=0)
    stack[2,t,...] = gem.max(axis=0)
    
io.imsave('/Users/xies/Desktop/mip00.tif', stack[0,...])
io.imsave('/Users/xies/Desktop/mip01.tif', stack[1,...])
io.imsave('/Users/xies/Desktop/mip02.tif', stack[2,...])


#%%ß

from tqdm import tqdm

dirname ='/Volumes/Seagate Backup Plus Drive/20200303_194709_09/14_crop/'

T = 200
c00 = natsort.natsorted(glob(path.join(dirname,'decon/FUCCIC/FUCCI*.tif')))[:T]
c01 = natsort.natsorted(glob(path.join(dirname,'decon/Lgr5GFP/Lgr5GFP*.tif')))[:T]

z2slice = 120

stack = np.zeros((2,T,500,420),dtype=np.int32)
MIP = np.zeros((2,T,500,420),dtype=np.int32)
for t,(h2b,cdt) in tqdm(enumerate(zip(c00,c01))):
   
    h2b = io.imread(h2b)
    cdt = io.imread(cdt)
    
    stack[0,t,...] = h2b[z2slice,:500,:420]
    stack[1,t,...] = cdt[z2slice,:500,:420]

    MIP[0,t,...] = h2b[:,:500,:420].max(axis=0)
    MIP[1,t,...] = cdt[:,:500,:420].max(axis=0)
    
io.imsave('/Users/xies/Desktop/ch00.tif', stack[0,...])
io.imsave('/Users/xies/Desktop/ch01.tif', stack[1,...])


io.imsave('/Users/xies/Desktop/mip00.tif', MIP[0,...])
io.imsave('/Users/xies/Desktop/mip01.tif', MIP[1,...])

