#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:24:22 2022

@author: xies
"""

import numpy as np
from skimage import io, filters, transform
from os import path
from glob import glob
from re import match
from pystackreg import StackReg

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R1'

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


#%% Reading the first ome-tiff file using imread reads entire stack

def sort_by_day(filename):
    day = match('\d+. Day (\d+\.?5?)',path.split(path.join(path.split(filename)[0])[0])[1])
    day = day.groups()[0]
    return float(day)

# Grab all registered B/R tifs
# B_tifs = sorted(glob(path.join(dirname,'*. Day*/ZSeries*/B_reg.tif')),key=sort_by_day)
G_tifs = sorted(glob(path.join(dirname,'*. Day*/ZSeries*/G_reg.tif')),key=sort_by_day)
R_shg_tifs = sorted(glob(path.join(dirname,'*. Day*/ZSeries*/R_shg_reg.tif')),key=sort_by_day)
R_tifs = sorted(glob(path.join(dirname,'*. Day*/ZSeries*/R_reg.tif')),key=sort_by_day)

#%% Hand build translation matrices

Tmatrices = dict()
Tmatrices[0] = transform.SimilarityTransform(translation=(-5,-60))
Tmatrices[t] = transform.SimilarityTransform(translation=(-12,-8))

Zshifts = dict()
Zshifts[0] = 10
Zshifts[1] = -7

#%% Transform

t = 1

G = io.imread(G_tifs[t])
R = io.imread(R_tifs[t])
R_shg = io.imread(R_tifs[t])

Zshift = Zshifts[t]

G_ref = G[G_zref,...]
R_ref = R[G_zref + Zshift,...]

# sr = StackReg(StackReg.TRANSLATION)
# T = sr.register(G_ref, R_ref)


R_transformed = np.zeros_like(R)
R_shg_transformed = np.zeros_like(R)

T = Tmatrices[t]
for i, R_slice in enumerate(R):
    R_transformed[i,...] = transform.warp(R_slice.astype(float),T)
    R_shg_transformed[i,...] = transform.warp(R_shg[i,...].astype(float),T)
    

# Z-pad the red + red_shg channel using Imax and Iz
bottom_padding = Zshift
if bottom_padding > 0: # the needs padding
    R_padded = np.concatenate( (np.zeros((bottom_padding,XX,XX)),R_transformed), axis= 0).astype(np.int16)
    R_shg_padded = np.concatenate( (np.zeros((bottom_padding,XX,XX)),R_shg_transformed), axis= 0).astype(np.int16)
elif bottom_padding < 0: # then needs trimming
    R_padded = R_transformed[-bottom_padding:,...]
    R_shg_padded = R_shg_transformed[-bottom_padding:,...]
elif bottom_padding == 0:
    R_padded = R_transformed
    R_shg_padded = R_shg_transformed

top_padding = G.shape[0] - R_padded.shape[0]
if top_padding > 0: # the needs padding
    R_padded = np.concatenate( (R_padded.astype(float), np.zeros((top_padding,XX,XX))), axis= 0).astype(np.int16)
    R_shg_padded = np.concatenate( (R_shg_padded.astype(float), np.zeros((top_padding,XX,XX))), axis= 0).astype(np.int16)
elif top_padding < 0: # then needs trimming
    R_padded = R_padded[0:top_padding,...]
    R_shg_padded = R_shg_padded[0:top_padding,...]

output_dir = path.dirname(R_tifs[t])

io.imsave(path.join(output_dir,'R_reg_reg.tif'),R_padded.astype(np.int16),check_contrast=False)
io.imsave(path.join(output_dir,'R_shg_reg_reg.tif'),R_shg_padded.astype(np.int16),check_contrast=False)

    

    
