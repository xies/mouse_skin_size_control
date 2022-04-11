#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 21:41:31 2022

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, filters, util, transform
from os import path
from glob import glob


dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/03-24-2022 power series 24h/M8 WT/R5 940nm_pw150 1020nm_pw225'


#%%

########################################################################################
# Author: Ujash Joshi, University of Toronto, 2017                                     #
# Based on Octave implementation by: Benjamin Eltzner, 2014 <b.eltzner@gmx.de>         #
# Octave/Matlab normxcorr2 implementation in python 3.5                                #
# Details:                                                                             #
# Normalized cross-correlation. Similiar results upto 3 significant digits.            #
# https://github.com/Sabrewarrior/normxcorr2-python/master/norxcorr2.py                #
# http://lordsabre.blogspot.ca/2017/09/matlab-normxcorr2-implemented-in-python.html    #
########################################################################################

import numpy as np
from skimage.filters import gaussian
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


#%% Reading the first ome-tiff file using imread reads entire stack

# Grab all registered B/R tifs
B_tifs = glob(path.join(dirname,'Day*/ZSeries*/B_reg.tif'))
G_tifs = glob(path.join(dirname,'Day*/ZSeries*/G_reg.tif'))
R_shg_tifs = glob(path.join(dirname,'Day*/ZSeries*/R_shg_reg.tif'))
R_tifs = glob(path.join(dirname,'Day*/ZSeries*/R_reg.tif'))

#%%

XX = 1024

assert(len(B_tifs) == len(R_tifs))

for t in range(len(B_tifs)):
    
    output_dir = path.split(path.dirname(R_tifs[t]))[0]
    if path.exists(path.join(output_dir,'stack_reg.tif')):
        print(f'Skipping t = {t}')
        continue
    
    print(f'--- Started t = {t} ---')
    B = io.imread(B_tifs[t])
    R_shg = io.imread(R_shg_tifs[t])
    G = io.imread(G_tifs[t])
    R = io.imread(R_tifs[t])
    R = R - R.min()
    
    # Find the slice with maximum mean value in R_shg channel
    Imax = R_shg.mean(axis=2).mean(axis=1).argmax()
    R_ref = R_shg[Imax,...]
    R_ref = gaussian(R_ref,sigma=0.5)
    
    # Iteratively find maximum x-corr (2D) for each B channel slice
    
    CC = np.zeros((B.shape[0],XX * 2 - 1,XX * 2 -1))
    
    for i,B_slice in enumerate(B):
        B_slice = gaussian(B_slice,sigma=0.5)
        CC[i,...] = normxcorr2(R_ref,B_slice,mode='full')

    [Iz,y_shift,x_shift] = np.unravel_index(CC.argmax(),CC.shape) # Iz refers to B channel
    y_shift = XX - y_shift
    x_shift = XX - x_shift
    
    B_transformed = np.zeros_like(B)
    G_transformed = np.zeros_like(G)
    T = transform.SimilarityTransform(translation=(-x_shift,-y_shift))
    
    for i, B_slice in enumerate(B):
        B_transformed[i,...] = transform.warp(B_slice.astype(float),T)
        G_transformed[i,...] = transform.warp(G[i,...].astype(float),T)
        
    G_transformed -= G_transformed.min()
    B_transformed -= B_transformed.min()
    

    output_dir = path.dirname(B_tifs[t])
    io.imsave(path.join(output_dir,'B_reg_reg.tif'),B_transformed.astype(np.int16))
    io.imsave(path.join(output_dir,'G_reg_reg.tif'),B_transformed.astype(np.int16))
    
    # Z-pad the red + red_shg channel using Imax and Iz
    bottom_padding = Iz - Imax
    if bottom_padding > 0: # the needs padding
        R_padded = np.concatenate( (np.zeros((bottom_padding,XX,XX)),R), axis= 0)
        R_shg_padded = np.concatenate( (np.zeros((bottom_padding,XX,XX)),R_shg), axis= 0)
    elif bottom_padding < 0: # then needs trimming
        R_padded = R[bottom_padding:,...]
        R_shg_padded = R_shg[bottom_padding:,...]
    
    top_padding = B.shape[0] - R_padded.shape[0]
    if top_padding > 0: # the needs padding
        R_padded = np.concatenate( (R_padded.astype(float), np.zeros((top_padding,XX,XX))), axis= 0)
        R_shg_padded = np.concatenate( (R_shg_padded.astype(float), np.zeros((top_padding,XX,XX))), axis= 0)

    elif top_padding < 0: # then needs trimming
        R_padded = R_padded[0:top_padding,...]
        R_shg_padded = R_shg_padded[0:top_padding,...]
    
    output_dir = path.dirname(R_tifs[t])

    io.imsave(path.join(output_dir,'R_reg_reg.tif'),R_padded.astype(np.int16))
    io.imsave(path.join(output_dir,'R_shg_reg_reg.tif'),R_shg_padded.astype(np.int16))
    




