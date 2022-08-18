#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:55:26 2022

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, transform, filters
from os import path
from re import match
from glob import glob
from pystackreg import StackReg
from tqdm import tqdm
import matplotlib.pylab as plt

dirname = '/Users/xies/OneDrive - Stanford/Skin/06-25-2022/M1 WT/R1'
dirname = '/Users/xies/OneDrive - Stanford/Skin/06-25-2022/M6 RBKO/R1'

#%% Reading the first ome-tiff file using imread reads entire stack


def sort_by_day(filename):
    day = match('\d+. Day (\d+\.?5?)',path.split(path.split(path.split(filename)[0])[0])[1])
    day = day.groups()[0]
    return float(day)

# Grab all registered B/R tifs
B_tifs = sorted(glob(path.join(dirname,'*Day*/ZSeries*/B_reg_reg.tif')),key=sort_by_day)
G_tifs = sorted(glob(path.join(dirname,'*Day*/ZSeries*/G_reg_reg.tif')),key=sort_by_day)
R_shg_tifs = sorted(glob(path.join(dirname,'*Day*/ZSeries*/R_shg_reg_reg.tif')),key=sort_by_day)
R_tifs = sorted(glob(path.join(dirname,'*Day*/ZSeries*/R_reg_reg.tif')),key=sort_by_day)

assert(len(B_tifs) == len(R_tifs))
assert(len(G_tifs) == len(R_shg_tifs))

########################################################################################
# Author: Ujash Joshi, University of Toronto, 2017                                     #
# Based on Octave implementation by: Benjamin Eltzner, 2014 <b.eltzner@gmx.de>         #
# Octave/Matlab normxcorr2 implementation in python 3.5                                #
# Details:                                                                             #
# Normalized cross-correlation. Similiar results upto 3 significant digits.            #
# https://github.com/Sabrewarrior/normxcorr2-python/master/norxcorr2.py                #
# http://lordsabre.blogspot.ca/2017/09/matlab-normxcorr2-implemented-in-python.html    #
########################################################################################

from scipy.signal import fftconvolve

def xcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return np.abs(out)
                  
                  
#%% Correlate each R_shg timepoint with first time point
# R_shg is best channel to use bc it only has signal in the collagen layer.
# Therefore it's easy to identify which z-stack is most useful.

XX = 1024

OVERWRITE = False

XY_reg = True
MANUAL_Z = False
APPLY = False

ref_T = 0

# Find the slice with maximum mean value in R_shg channel
R_shg_ref = io.imread( R_shg_tifs[ref_T] )
Z_ref = R_shg_ref.shape[ref_T]
Imax_ref = R_shg_ref.std(axis=2).std(axis=1).argmax() # Find max contrast slice
ref_img = R_shg_ref[Imax_ref,...]


# variables to save:
z_pos_in_original = np.zeros(len(B_tifs))
z_pos_in_original[ref_T] = Imax_ref
z0_in_refframe = Z_ref
XY_matrices = []

for t in tqdm( np.arange(3,len(B_tifs)) ): # 0-indexed
    
    if t == ref_T:
        continue
    
    output_dir = path.split(path.dirname(R_tifs[t]))[0]
    if APPLY and not OVERWRITE and path.exists(path.join(path.dirname(B_tifs[t]),'B_align.tif')):
        print(f'Skipping t = {t}')
        continue
    
    print(f'Working on {R_shg_tifs[t]}')
    #Load the target
    R_shg_target = io.imread(R_shg_tifs[t]).astype(float)
    
    # Find simlar in the next time point
    # If specified, use the manually determined ref_z
    if MANUAL_Z:
        Imax_target = 64
        print(f'Target z-slice manually set at {Imax_target}')
    else:
        # 1. Use xcorr2 to find the z-slice on the target that has max CC with the reference
        CC = np.zeros(R_shg_target.shape[0])
        for z,im in enumerate(R_shg_target):
            CC[z] = xcorr2(ref_img, im).max()
        Imax_target = CC.argmax()
        print(f'Target z-slice automatically determined to be {Imax_target}')
    z_pos_in_original[t] = Imax_target
    
    # Perform transformations
    B = io.imread(B_tifs[t])
    G = io.imread(G_tifs[t])
    R = io.imread(R_tifs[t])
    
    B_transformed = B.copy(); R_transformed = R.copy(); G_transformed = G.copy(); R_shg_transformed = R_shg_target.copy();
    
    if XY_reg:
        moving_img = R_shg_target[Imax_target,...]
        print('\n Starting stackreg')
        # Use StackReg to 'align' the two z slices
        sr = StackReg(StackReg.RIGID_BODY)
        T = sr.register(ref_img,moving_img) #Obtain the transformation matrices
        XY_matrices.append(T)
        
        if APPLY:
            print('Applying transformation matrices')
            # Apply transformation matrix to each stacks
            
            for i, B_slice in enumerate(B):
                B_transformed[i,...] = sr.transform(B_slice.astype(float),tmat=T)
                G_transformed[i,...] = sr.transform(G[i,...].astype(float),tmat=T)
            for i, R_slice in enumerate(R):
                R_transformed[i,...] = sr.transform(R_slice.astype(float),tmat=T)
                R_shg_transformed[i,...] = sr.transform(R_shg_target[i,...].astype(float),tmat=T)
        
    if APPLY:    
        # Z-pad the time point in reference to t - 1
        Z_target = R_shg_target.shape[0]
    
        print('Padding')
        top_padding = Imax_ref - Imax_target
        if top_padding > 0: # the needs padding
            R_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),R_transformed), axis= 0)
            G_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),G_transformed), axis= 0)
            B_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),B_transformed), axis= 0)
            R_shg_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),R_shg_transformed), axis= 0)
            
        elif top_padding < 0: # then needs trimming
            R_padded = R_transformed[-top_padding:,...]
            G_padded = G_transformed[-top_padding:,...]
            B_padded = B_transformed[-top_padding:,...]
            R_shg_padded = B_transformed[-top_padding:,...]
            
        elif top_padding == 0:
            R_padded = R_transformed
            G_padded = G_transformed
            B_padded = B_transformed
            R_shg_padded = R_shg_transformed
            
        delta_ref = Z_ref - Imax_ref
        delta_target = Z_target - Imax_target
        bottom_padding = delta_ref - delta_target
        if bottom_padding > 0: # the needs padding
            R_padded = np.concatenate( (R_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
            G_padded = np.concatenate( (G_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
            B_padded = np.concatenate( (B_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
            R_shg_padded = np.concatenate( (R_shg_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
            
        elif bottom_padding < 0: # then needs trimming
            R_padded = R_padded[0:bottom_padding,...]
            G_padded = G_padded[0:bottom_padding,...]
            B_padded = B_padded[0:bottom_padding,...]
            R_shg_padded = R_shg_padded[0:bottom_padding,...]
    
        print('Saving')
        output_dir = path.dirname(B_tifs[t])
        io.imsave(path.join(output_dir,'B_align.tif'),B_padded.astype(np.int16),check_contrast=False)
        io.imsave(path.join(output_dir,'G_align.tif'),G_padded.astype(np.int16),check_contrast=False)
        
        output_dir = path.dirname(R_tifs[t])
        io.imsave(path.join(output_dir,'R_align.tif'),R_padded.astype(np.int16),check_contrast=False)
        io.imsave(path.join(output_dir,'R_shg_align.tif'),R_shg_padded.astype(np.int16),check_contrast=False)


#%% Manually reconstruct transformation matrix if there is problematic time point



#%% Sort filenames by time (not alphanumeric) and then assemble 'master stack'
        
# But exclude R_shg since 4-channel tifs are annoying to handle for FIJI loading.

T = len(B_tifs)-1

filelist = pd.DataFrame()
filelist['B'] = sorted(glob(path.join(dirname,'*Day*/ZSeries*/B_align.tif')), key = sort_by_day)
filelist['G'] = sorted(glob(path.join(dirname,'*Day*/ZSeries*/G_align.tif')), key = sort_by_day)
filelist['R'] = sorted(glob(path.join(dirname,'*Day*/ZSeries*/R_align.tif')), key = sort_by_day)
filelist.index = np.arange(0,T)

# # t= 0 has no '_align'
s = pd.Series({'B': glob(path.join(dirname,'*Day 0/ZSeries*/B_reg_reg.tif'))[0],
                  'G': glob(path.join(dirname,'*Day 0/ZSeries*/G_reg_reg.tif'))[0],
                  'R': glob(path.join(dirname,'*Day 0/ZSeries*/R_reg_reg.tif'))[0]}, name=0)

filelist = filelist.append(s)
filelist = filelist.sort_index()

# Save individual day*.tif

MAX = 2**16-1

def fix_image_range(im, max_range):
    
    im = im.copy().astype(float)
    im[im == 0] = np.nan
    im = im - np.nanmin(im)
    im = im / np.nanmax(im) * max_range
    im[np.isnan(im)] = 0
    return im.astype(np.uint16)

for t in tqdm(range(T+1)):
    # stack = np.zeros((Z_ref,3,XX,XX))
    
    R = io.imread(filelist.loc[t,'R'])
    G = io.imread(filelist.loc[t,'G'])
    B = io.imread(filelist.loc[t,'B'])
    R_ = fix_image_range(R,MAX)
    G_ = fix_image_range(G,MAX)
    B_ = fix_image_range(B,MAX)
    
    # Do some image range clean up
    
    stack = np.stack((R_,G_,B_))
    io.imsave(path.join(dirname,f'im_seq/t{t}.tif'),stack.astype(np.uint16),check_contrast=False)

#%% Manually input any thing and save

z_pos_in_original[11] = 65

#@todo: figure out how to get translation matrix (also sign of transform)
# from scipy.spatial import transform
# r = transform.Rotation.from_euler('z',3,degrees=True).as_matrix()
# r = r + transform.Translation

import pickle as pkl
with open(path.join(dirname,'alignment_information.pkl'),'wb') as f:
    pkl.dump([z_pos_in_original,XY_matrices,z0_in_refframe],f)

#%% Save master stack
# Load file and concatenate them appropriately
# FIJI default: CZT XY, but this is easier for indexing
stack = np.zeros((T,Z_ref,3,XX,XX))

for t in range(T):
    R = io.imread(filelist.loc[t,'R'])
    G = io.imread(filelist.loc[t,'G'])
    B = io.imread(filelist.loc[t,'B'])
    
    stack[t,:,0,:,:] = R
    stack[t,:,1,:,:] = G
    stack[t,:,2,:,:] = B
    
# io.imsave(path.join(dirname,'master_stack.tif'),stack.astype(np.int16))



