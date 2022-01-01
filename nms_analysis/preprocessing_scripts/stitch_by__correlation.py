#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:48:25 2021

@author: xies
"""

import numpy as np
import seaborn as sb
import pandas as pd
from skimage import io,filters,util
from scipy import signal
from os import path
from glob import glob
from pystackreg import StackReg

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/10-20-2021/WT/R1'

#%% Functions
def find_max_corr(z_reg,consecutive_frames):

    max_corr = dict()
    T = len(z_reg)
    
    for t in range(T-1):
        
        #Make sure z is first dim for list comp
        this_frame = z_reg[t]
        next_frame = z_reg[t+1]
        
        # Kernel-smooth
        this_frame = filters.gaussian(this_frame,sigma=0.5)
        next_frame = filters.gaussian(next_frame,sigma=0.5)
    
        C = np.zeros((this_frame.shape[0],next_frame.shape[0]))
    
        for z,ref_z in enumerate(this_frame):
            ref_z = this_frame[z,...]
        
            for i,current_slice in enumerate(next_frame):
                C[z,i] = signal.correlate2d(ref_z, current_slice, mode='valid')
    
        [source_z, target_z] = np.unravel_index(C.argmax(),C.shape)
        max_corr[consecutive_frames[t]] = (source_z,target_z)

    return max_corr


def collate_z_registration(filelist,max_corr=None,ref_z=None):
    T = len(filelist)
    if ref_z == None:
        #Propagate the 'reference in z' so that it's a simple list of z-ref instead of corresponding pairs
        diff_in_ref = np.diff(np.array(list(max_corr.values())),axis=1)
        
        ref_z = np.hstack((0,np.cumsum(diff_in_ref)))
        # Make sure no ref_Z is negative
        if min(ref_z) < 0:
            ref_z = ref_z - ref_z.min()
    
    lengths = np.array([ io.imread(x).shape[0] for x in filelist ])
    bottom_ref_z = lengths - ref_z
    
    top_Z = max(ref_z)
    bottom_Z = max( bottom_ref_z)
    
    # z_reg = np.zeros((T,top_Z+bottom_Z,1024,1024,3))
    z_reg = []
    for t in range(T):
        
        print(f'Z registering t = {t}')
        
        this_frame = io.imread(filelist[t])[:,:,:,:]
        this_reg = np.zeros((top_Z+bottom_Z,1024,1024,3))
        
        top_half = this_frame[0:ref_z[t],...]
        bottom_half = this_frame[ref_z[t]:,...]
        this_reg[top_Z - ref_z[t]: top_Z,...] = top_half
        this_reg[top_Z : top_Z + lengths[t] - ref_z[t]] = bottom_half
        
        z_reg.append(this_reg)
        
    return z_reg, top_Z, ref_z

def stack_reg_consecutive_frames(z_reg,top_Z):
    T = len(z_reg)
    Z,_,_ = z_reg[0].shape
    
    for t in range(T-1):
        print(f'XY registering t = {t}')
        
        ref = z_reg[t][top_Z,:,:,2]
        target_img = z_reg[t+1][top_Z,:,:,2]
        
        sr = StackReg(StackReg.RIGID_BODY)
        reg_matrix = sr.register(ref,target_img)
        
        # # Use registration matrix on whole stack
        
        for z in range(Z):
            for c in range(3):
                z_reg[t+1][z,:,:,c] = sr.transform( z_reg[t+1][z,:,:,c] )
        
    return z_reg

#%% Load filelist

filelist = glob(path.join(dirname,'Day */day*.tif'))
T = len(filelist)

z_reg = list(map(io.imread,filelist))
z_reg = [im[:,:,:,2] for im in z_reg]
consecutive_frames = list(zip(np.arange(0,T-1),np.arange(1,T)))

#%% Automatic iterative registration
""" 1) First find the most similar z-slice between pairs of stacks
    2) Construct a stitched together 4D t-z-stack
    3) Go through consecutive frames and XY-register the most similar z-stack
    
    Repeat for Niters
"""

Niters = 1
ref_z = [4,6,8,3,9,7,12,10,7,9,5,3,2,8,6,5,4,5,6]

for i in range(Niters):
    print(f'--- Iteration {i} ---')
    # max_corr = find_max_corr(z_reg,consecutive_frames)
    # z_reg,top_Z,ref_z = collate_z_registration(filelist,max_corr=max_corr)
    z_reg,top_Z,ref_z = collate_z_registration(filelist,ref_z=ref_z)
    
    z_reg = stack_reg_consecutive_frames(z_reg[:10],top_Z)
    
    
#%% Save stack

for idx,img in enumerate(z_reg):
    io.imsave(f'/Users/xies/Desktop/z_reg_t{idx}.tif',img.astype(np.int16))

#%% Manual adjustment

t = 8

ref = z_reg[t-1][ref_z[t-1],...]
target_img = z_reg[t][ref_z[t],...]

sr = StackReg(StackReg.RIGID_BODY)
sr.register(ref,target_img)
target_reg = sr.transform(z_reg[t][ref_z[t],...])

plt.figure(); io.imshow(ref)
plt.figure(); io.imshow(target_reg-target_reg.min())
# for z in range(z_reg[t].shape[0]):
#     z_reg[t][z,...] = sr.transform( z_reg[t][z,...])
    
# io.imsave(f'/Users/xies/Desktop/z_reg_t{t}.tif',z_reg[t].astype(np.int16))


