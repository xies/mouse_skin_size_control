#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:22:11 2024

@author: xies
"""


import numpy as np
import pandas as pd
from skimage import io, transform, filters, util
from os import path
from re import match
from glob import glob
from pystackreg import StackReg
from tqdm import tqdm
from mathUtils import normxcorr2
import matplotlib.pylab as plt
import pickle as pkl

from twophotonUtils import parse_unaligned_channels

# dirname = '/Volumes/T7/11-07-2023 DKO/M3 p107homo Rbfl/Right ear/Post Ethanol/R3'
# dirname = '/Volumes/T7/01-13-2023 Ablation K14Cre H2B FUCCI/Black unclipped less leaky DOB 06-30-2023/R2'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Old mice/01-24-2024 12month old mice/F1 DOB 12-18-2022/R1'

filelist = parse_unaligned_channels(dirname,folder_str='*.*/')

#%%

'''
Step three: automatically register the two timepoints
'''
def _update_timepoints_on_file_change(widget):
    """Called whenever the file picker is changed. Will look into that directory,
    and return the available timepoints as defined by parse_unregistered_channels
    """
    dirname = widget.dirname.value
    choices = None
    filelist = parse_unregistered_channels(dirname)
    if len(filelist) > 0:
        choices = filelist.index
    else:
        show_warning(f'Directory {dirname} is not a region directory.')
    if choices is not None:
        widget.timepoints_to_register.choices = choices

def auto_register_b_and_rshg():
    '''
    Gets dirname from picker and populates the available timepoints to register.
    See _update_timepoints_on_file_change for the populator
    '''
    @magicgui(call_button='Register channels (R_shg and B)',
              timepoints_to_register={'widget_type':'Select',
                                      'choices':DEFAULT_CHOICES,
                                      'label':'Time points to register'},
              dirname={'label':'Image region to load:','mode':'d'})
    def widget(
        dirname=Path.home(),
        timepoints_to_register=(0),
        OVERWRITE: bool=False,
        ):
    
        filelist = parse_unregistered_channels(dirname)
        
        for t in progress(timepoints_to_register):
        
            # Check for overwriting
            output_dir = path.split(path.dirname(filelist.loc[t,'R']))[0]
            if path.exists(path.join(path.dirname(filelist.loc[t,'R']),'R_reg_reg.tif'))  and not OVERWRITE:
            # and path.exists(path.join(path.dirname(B_tifs[t]),'B_reg_reg.tif'))  and not OVERWRITE:
                print(f'Skipping t = {t} because its R_reg_reg.tif already exists')
                continue
                
            print(f'\n--- Started t = {t} ---')
            B = io.imread(filelist.loc[t,'B'])
            R_shg = io.imread(filelist.loc[t,'R_shg'])
            # G = io.imread(filelist.loc[t,'G'])
            R = io.imread(filelist.loc[t,'R'])    
            print('Done reading images')
                
            # Find the slice with maximum mean value in R_shg channel
            z_ref = R_shg.mean(axis=2).mean(axis=1).argmax()
            print(f't = {t}: R_shg max std at {z_ref}')
            R_ref = R_shg[z_ref,...]
            R_ref = filters.gaussian(R_ref,sigma=0.5)
            z_moving = find_most_likely_z_slice_using_CC(R_ref,B)
            print(f'Cross correlation done and target Z-slice set at: {z_ref}')
            target = filters.gaussian(B[z_ref,...],sigma=0.5)
        
            #NB: Here, move the R channel wrt the B channel
            print('StackReg + transform')
            sr = StackReg(StackReg.RIGID_BODY)
            T = sr.register(target/target.max(),R_ref) #Obtain the transformation matrices
            T = EuclideanTransform(T)
            R_transformed = np.zeros_like(R).astype(float)
            R_shg_transformed = np.zeros_like(R).astype(float)
            for i, R_slice in enumerate(R):
                R_transformed[i,...] = warp(R_slice,T)
                R_shg_transformed[i,...] = warp(R_shg[i,...],T)
            
            # z-pad
            R_padded = z_translate_and_pad(B,R_transformed,z_ref,z_moving)
            R_shg_padded = z_translate_and_pad(B,R_shg_transformed,z_ref,z_moving)
                
            output_dir = path.dirname(filelist.loc[t,'G'])
        
            print('Saving')
            io.imsave(path.join(output_dir,'R_reg_reg.tif'),util.img_as_uint(R_padded/R_padded.max()),check_contrast=False)
            io.imsave(path.join(output_dir,'R_shg_reg_reg.tif'),util.img_as_uint(R_shg_padded/R_shg_padded.max()),check_contrast=False)
        
    @widget.dirname.changed.connect
    def update_timepoints_on_file_change(event=None):
        _update_timepoints_on_file_change(widget)
    return widget

    