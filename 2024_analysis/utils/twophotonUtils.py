#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:59:29 2022

@author: xies
"""

from glob import glob
import pandas as pd
from re import findall
from os import path
import matplotlib.pyplot as plt
import numpy as np

from skimage import filters
from mathUtils import normxcorr2
from tqdm import tqdm

import xml.etree.ElementTree as ET
from dateutil import parser
from collections import OrderedDict

def return_prefix(filename):

    # Use a function to regex the Day number and use that to sort
    day = findall('.(\d+)\. ',filename)
    assert(len(day) == 1)

    return int(day[0])

def parse_aligned_timecourse_directory(dirname,folder_str='*. */',SPECIAL=[]):
    # Given a directory (of Prairie Instruments time course)
    #

    B = glob(path.join(dirname,folder_str, 'B_align.tif'))
    idx = [return_prefix(f) for f in B]
    filelist = pd.DataFrame(index=idx)
    filelist.loc[idx,'B'] = B

    G = glob(path.join(dirname,folder_str, 'G_align.tif'))
    idx = [return_prefix(f) for f in G]
    filelist.loc[idx,'G'] = G

    R = glob(path.join(dirname,folder_str, 'R_align.tif'))
    idx = [return_prefix(f) for f in R]
    filelist.loc[idx,'R'] = R

    R_shg = glob(path.join(dirname,folder_str, 'R_shg_align.tif'))
    idx = [return_prefix(f) for f in R_shg]
    filelist.loc[idx,'R_shg'] = R

    # T = len(filelist)

    for t in SPECIAL:
        # t= 0 has no '_align'imp
        # s = pd.DataFrame({'B': sorted(glob(path.join(dirname,folder_str, 'B_reg.tif')))[0],
        #                   'G': sorted(glob(path.join(dirname,folder_str, 'G_reg.tif')))[0],
        #                   'R': sorted(glob(path.join(dirname,folder_str, 'R_reg_reg.tif')))[0],
        #               'R_shg': sorted(glob(path.join(dirname,folder_str, 'R_shg_reg_reg.tif')))[0]},
        #                  index=[t])

        B = glob(path.join(dirname,f'{t}. */B_reg.tif'))[0]
        filelist.loc[t,'B'] = B
        G = glob(path.join(dirname,f'{t}. */G_reg.tif'))[0]
        filelist.loc[t,'G'] = G
        R = glob(path.join(dirname,f'{t}. */R_reg_reg.tif'))[0]
        filelist.loc[t,'R'] = R
        R_shg = glob(path.join(dirname,f'{t}. */R_shg_reg_reg.tif'))[0]
        filelist.loc[t,'R_shg'] = R_shg

        # filelist = pd.concat((s,filelist))
    filelist = filelist.sort_index()

    return filelist


def parse_unregistered_channels(dirname,folder_str='*. Day*/',sort_func=return_prefix):
    # Given a directory (of Prairie Instruments time course), grab all the _reg.tifs
    # (channels are not registered to each other)
    #


    B = glob(path.join(dirname,folder_str, 'B_reg.tif'))
    idx = [return_prefix(f) for f in B]
    filelist = pd.DataFrame(index=idx)
    filelist.loc[idx,'B'] = B

    # filelist['G'] = sorted(glob(path.join(dirname,folder_str, 'G_reg.tif')), key = return_prefix)
    G = glob(path.join(dirname,folder_str, 'G_reg.tif'))
    idx = [return_prefix(f) for f in G]
    filelist.loc[idx,'G'] = G

    R = glob(path.join(dirname,folder_str, 'R_reg.tif'))
    idx = [return_prefix(f) for f in R]
    filelist.loc[idx,'R'] = R

    R_shg = glob(path.join(dirname,folder_str, 'R_shg_reg.tif'))
    idx = [return_prefix(f) for f in R_shg]
    filelist.loc[idx,'R_shg'] = R_shg

    filelist = filelist.sort_index()

    return filelist

def parse_unaligned_channels(dirname,folder_str='*. Day*/'):
    # Given a directory (of Prairie Instruments time course)
    #

    # filelist['B'] = sorted(glob(path.join(dirname,folder_str, 'B_reg.tif')), key = return_prefix)
    B = glob(path.join(dirname,folder_str, 'B_reg.tif'))
    idx = [return_prefix(f) for f in B]
    filelist = pd.DataFrame(index=idx,columns=['B','G','R','R_shg'])
    filelist.loc[idx,'B'] = B

    # filelist['G'] = sorted(glob(path.join(dirname,folder_str, 'G_reg.tif')), key = return_prefix)
    G = glob(path.join(dirname,folder_str, 'G_reg.tif'))
    idx = [return_prefix(f) for f in G]
    filelist.loc[idx,'G'] = G

    R = glob(path.join(dirname,folder_str, 'R_reg_reg.tif'))
    idx = [return_prefix(f) for f in R]
    filelist.loc[idx,'R'] = R

    R_shg = glob(path.join(dirname,folder_str, 'R_shg_reg_reg.tif'))
    idx = [return_prefix(f) for f in R_shg]
    filelist.loc[idx,'R_shg'] = R_shg

    filelist = filelist.sort_index()
    # T = len(filelist)

    return filelist

def plot_cell_volume(track,x='Frame',y='Volume'):
    t = track[x]
    y = track[y]
    if 'Mitosis' in track.columns:
        if track.iloc[0]['Mitosis']:
            t = t[:-1]
            y = y[:-1]
    plt.plot(t,y)


def parse_XML_timestamps(region_dir,subdir_str='*. Day*/',beginning=0):
    
    T = len(glob(path.join(region_dir,subdir_str)))
    timestamps = OrderedDict()

    for t in range(beginning,beginning+T):

        subfolders = glob(path.join(region_dir,f'{t}.*/ZSeries*/'))

        for d in subfolders:
            ome_tifs = glob(path.join(d,'*.ome.tif'))
            xmls = glob(path.join(d,'*.xml'))
            if len(ome_tifs) > 40:
                if len(findall('1020nm',path.split(path.split(d)[0])[1])) == 0:

                    tree = ET.parse(xmls[0])
                    timestr = tree.getroot().attrib['date']
                    timestamp = parser.parse(timestr)

        timestamps[t] = timestamp

    return timestamps


def parse_voxel_resolution_from_XML(region_dir):

    xmls = glob(path.join(region_dir,'*.*/ZSeries*/*.xml'))

    tree = ET.parse(xmls[0])
    root = tree.getroot()

    for child in root:
        if child.tag == 'PVStateShard':
            PVState = child
            break

    for child in PVState:
        if child.attrib['key'] == 'micronsPerPixel':
            microns_resolution = child

    for child in microns_resolution:
        if child.attrib['index'] == 'XAxis':
            dx = child.attrib['value']
        if child.attrib['index'] == 'ZAxis':
            dz = child.attrib['value']

    return float(dx),float(dz)

def find_most_likely_z_slice_using_CC(ref_slice,stack):
    '''

    Parameters
    ----------
    ref_slice : YxX array
        Single slice of reference image
    stack : 3xYxX array
        A z-stack image to find the slice correspoinding to ref_slice.

    Returns
    -------
    Iz : int
        index in stack of the corresponding ref_slice.

    '''
    assert(ref_slice.ndim == 2)
    assert(stack.ndim == 3)
    XX = stack.shape[1]
    CC = np.zeros((stack.shape[0],XX * 2 - 1,XX * 2 -1))

    print('Cross correlation started')
    for i,B_slice in enumerate(stack):
        B_slice = filters.gaussian(B_slice,sigma=0.5)
        CC[i,...] = normxcorr2(ref_slice,B_slice,mode='full')
    [Iz,y_shift,x_shift] = np.unravel_index(CC.argmax(),CC.shape) # Iz refers to B channel
    return Iz

def z_translate_and_pad(im_ref,im_moving,z_ref,z_moving):
    '''
    Takes two z-stacks and translate the im_moving so that z_ref and z_moving will end up
    being the same index in the translated image. Will also truncate/pad im_moving
    to be the same size as im_ref
    
    '''
    XX = im_moving.shape[1]

    # Bottom padding
    bottom_padding = z_ref - z_moving
    if bottom_padding > 0: # the needs padding
        im_padded = np.concatenate( (np.zeros((bottom_padding,XX,XX)),im_moving), axis= 0)
    elif bottom_padding < 0: # then needs trimming
        im_padded = im_moving[-bottom_padding:,...]
    elif bottom_padding == 0:
        im_padded = im_moving

    top_padding = im_ref.shape[0] - im_padded.shape[0]
    if top_padding > 0: # the needs padding
        im_padded = np.concatenate( (im_padded.astype(float), np.zeros((top_padding,XX,XX))), axis= 0)
    elif top_padding < 0: # then needs trimming
        im_padded = im_padded[0:top_padding,...]

    # Cut down the z-stack shape
    Zref = im_ref.shape[0]
    im_padded = im_padded[:Zref,...]

    assert(np.all(im_ref.shape == im_padded.shape))

    return im_padded

def z_align_ragged_timecourse(ragged_stack_list,same_Zs):
    '''
    Takes two z-stacks and translate the im_moving so that z_ref and z_moving will end up
    being the same index in the translated image. Will also truncate/pad im_moving
    to be the same size as im_ref
    
    '''
    same_Zs = same_Zs.astype(int)
    original_stack_sizes = np.array([x.shape[0] for x in ragged_stack_list])
    top_size = (original_stack_sizes - same_Zs)
    
    ragged_bottom_Z = same_Zs.max() - same_Zs
    
    TT = len(ragged_stack_list)
    XX = ragged_stack_list[0].shape[1]
    
    aligned_stack_timecourse = np.zeros((TT,int(same_Zs.max()+top_size.max()),XX,XX))

    for t in tqdm(range(TT)):
        
        # Take the 'bottom' portion of original stack
        aligned_stack_timecourse[t, ragged_bottom_Z[t]:same_Zs.max() ,:,:] \
            = ragged_stack_list[t][0:same_Zs[t],:,:]
        # Take the 'top' portion of the original stack
        aligned_stack_timecourse[t,same_Zs.max():same_Zs.max()+top_size[t],:,:] \
            = ragged_stack_list[t][same_Zs[t]:,:,:]
        
    return aligned_stack_timecourse
