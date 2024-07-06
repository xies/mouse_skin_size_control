#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 00:30:57 2022

@author: xies
"""

import numpy as np
from skimage import io, filters, transform
from os import path
from glob import glob

from pystackreg import StackReg
from tqdm import tqdm


#dirname = '/Users/xies/Box/Mouse/Skin/Two photon/Shared/20210322_K10 revisits/20220322_female4/area3'
dirname = '/home/xies/data/two_photon_shared/20210322_K10 revisits/20220322_female4/area3'
filenames = glob(path.join(dirname,'reg*.tif'))

im_list = list(map(io.imread,filenames))

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


#%%

XX = 1024

manual_z_ref = np.array([33 ,33, 35, 35, 31, 38, 42, 35, 33]) - 1

assert(len(manual_z_ref) == len(im_list))

ref_img = im_list[0][manual_z_ref[0],...,2]
ref_img = filters.gaussian(ref_img, sigma = 1)
Z_ref = im_list[0].shape[0]

for t in tqdm( np.arange(1,len(im_list)) ):
    
    print('Registering')
    target_img = im_list[t][manual_z_ref[t],...,2]
    target_img = filters.gaussian(target_img, sigma=20) # align the "vague shapes"

    sr = StackReg(StackReg.RIGID_BODY)
    T = sr.register(ref_img,target_img) #Obtain the transformation matrices
    
    # Apply matrix to new time point
    transformed = np.zeros_like(im_list[t])
    for z,im in enumerate(im_list[t]):
        transformed[z,...] = transform.warp(im.astype(float),T)

    print('Padding')
    top_padding = manual_z_ref[0] - manual_z_ref[t]
    if top_padding > 0: # the needs padding
        padded = np.concatenate( (np.zeros((top_padding,XX,XX,3)),transformed), axis= 0)
        
    elif top_padding < 0: # then needs trimming
        padded = transformed[-top_padding:,...]
        
    elif top_padding == 0:
        padded = transformed
        
    delta_ref = Z_ref - manual_z_ref[0]
    delta_target = im_list[t].shape[0] - manual_z_ref[t]
    bottom_padding = delta_ref - delta_target
    if bottom_padding > 0: # the needs padding
        padded = np.concatenate( (padded.astype(float), np.zeros((bottom_padding,XX,XX,3))), axis= 0)
        
    elif bottom_padding < 0: # then needs trimming
        padded = padded[0:bottom_padding,...]
        
    
    print('Saving')
    io.imsave(path.join(dirname,'reg',f'reg_day{t}.tif'), padded.astype(np.int16))



