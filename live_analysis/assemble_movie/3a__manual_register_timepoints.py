#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 22:29:22 2022

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

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-08-2022/F2 WT/R1'

#%% Reading the first ome-tiff file using imread reads entire stack

# Grab all registered B/R tifs
B_tifs = glob(path.join(dirname,'Day*/ZSeries*/B_reg_reg.tif'))
G_tifs = glob(path.join(dirname,'Day*/ZSeries*/G_reg_reg.tif'))
R_shg_tifs = glob(path.join(dirname,'Day*/ZSeries*/R_shg_reg_reg.tif'))
R_tifs = glob(path.join(dirname,'Day*/ZSeries*/R_reg_reg.tif'))


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

def normxcorr2(template, image, mode="full"):
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
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return np.abs(out)
                  
                  
#%% Correlate each R_shg timepoint with subsequent timepoint (Nope, using first time point instead)
# R_shg is best channel to use bc it only has signal in the collagen layer.
# Therefore it's easy to identify which z-stack is most useful.

XX = 1024
OVERWRITE = True

assert(len(B_tifs) == len(R_tifs))
assert(len(G_tifs) == len(R_shg_tifs))

ref_T = 2
target_T = 9

R_shg_ref = io.imread( R_shg_tifs[3] )

###

# Find the slice with maximum mean value in R_shg channel
Imax_ref = R_shg_ref.std(axis=2).std(axis=1).argmax() # Find max contrast slice
ref_img = R_shg_ref[Imax_ref,...]
Z_ref = R_shg_ref.shape[0]

CC = np.zeros(len(R_tifs))
Iz_target_slice = np.zeros(len(R_tifs))

output_dir = path.split(path.dirname(R_tifs[target_T]))[0]

#Load the target
R_shg_target = io.imread(R_shg_tifs[target_T])

# Find simlar in the next time point
Imax_target = R_shg_target.std(axis=2).std(axis=1).argmax()
target_img = R_shg_target[Imax_target,...]

print('\n Starting stackreg')
# Use StackReg to 'align' the two z slices
sr = StackReg(StackReg.RIGID_BODY)
T = sr.register(ref_img,target_img) #Obtain the transformation matrices

B = io.imread(B_tifs[target_T])
G = io.imread(G_tifs[target_T])
R = io.imread(R_tifs[target_T])

T = transform.SimilarityTransform(matrix=T)

print('Applying transformation matrices')
# Apply transformation matrix to each stacks
B_transformed = np.zeros_like(B)
G_transformed = np.zeros_like(G)
R_transformed = np.zeros_like(R)
R_shg_transformed = np.zeros_like(R)
for i, B_slice in enumerate(B):
    B_transformed[i,...] = transform.warp(B_slice.astype(float),T)
    G_transformed[i,...] = transform.warp(G[i,...].astype(float),T)
    R_transformed[i,...] = transform.warp(R[i,...].astype(float),T)
    R_shg_transformed[i,...] = transform.warp(R_shg_target[i,...].astype(float),T)
    
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
    R_padded = R[-top_padding:,...]
    G_padded = G[-top_padding:,...]
    B_padded = B[-top_padding:,...]
    R_shg_padded = R_shg_target[-top_padding:,...]
    
elif top_padding == 0:
    R_padded = R
    G_padded = G
    B_padded = B
    R_shg_padded = R_shg_target
    
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
output_dir = path.dirname(B_tifs[target_T])
io.imsave(path.join(output_dir,'B_align.tif'),B_padded.astype(np.int16))
io.imsave(path.join(output_dir,'G_align.tif'),G_padded.astype(np.int16))

output_dir = path.dirname(R_tifs[target_T])
io.imsave(path.join(output_dir,'R_align.tif'),R_padded.astype(np.int16))
io.imsave(path.join(output_dir,'R_shg_align.tif'),R_shg_padded.astype(np.int16))
    
#%% Sort filenames by time (not alphanumeric) and then assemble 'master stack'
# But exclude R_shg since 4-channel tifs are annoying to handle for FIJI loading.

T = len(B_tifs)-1
# Use a function to regex the Day number and use that to sort
def sort_by_number(filename):
    day = match('Day (\d+)',path.split(path.split(path.split(filename)[0])[0])[1])
    day = day.groups()[0]
    return int(day)

filelist = pd.DataFrame()
filelist['B'] = sorted(glob(path.join(dirname,'Day*/ZSeries*/B_align.tif')), key = sort_by_number)
filelist['G'] = sorted(glob(path.join(dirname,'Day*/ZSeries*/G_align.tif')), key = sort_by_number)
filelist['R'] = sorted(glob(path.join(dirname,'Day*/ZSeries*/R_align.tif')), key = sort_by_number)
filelist.index = np.arange(1,T+1)

# t= 0 has no '_align'
s = pd.Series({'B': glob(path.join(dirname,'Day 3.5/ZSeries*/B_reg_reg.tif'))[0],
                 'G': glob(path.join(dirname,'Day 3.5/ZSeries*/G_reg_reg.tif'))[0],
                 'R': glob(path.join(dirname,'Day 3.5/ZSeries*/R_reg_reg.tif'))[0]}, name=0)

filelist = filelist.append(s)
filelist = filelist.sort_index()

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
    
io.imsave(path.join(dirname,'master_stack.tif'),stack.astype(np.int16))



